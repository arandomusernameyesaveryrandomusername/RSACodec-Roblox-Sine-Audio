"""
rsc_encoder.py -- Roblox Sine Codec (RSC) Encoder  [optimized]

Format: RSC4
  Header : 23 bytes  (magic changed RSC3→RSC4, layout identical)
  Partial: 4 bytes   uint16 freq | uint16 amp
           (phase removed — decoder simulates phase continuously)

Usage:
    python rsc_encoder.py --input audio.wav --output audio.rsc
    python rsc_encoder.py --input audio.wav --output audio.rsc --partials 384 --samplerate 44100
"""

import argparse
import math
import struct
import wave

import numpy as np
from scipy.signal import find_peaks
from scipy.signal import windows


# ---------------------------------------------
#  Constants
# ---------------------------------------------
TARGET_FPS         = 60
DEFAULT_PARTIALS   = 384
DEFAULT_SAMPLERATE = 44100
RSC_EXTENSION      = ".rsc"
ANALYSIS_WIN       = 4096     # ~10.8 Hz/bin at 44100
SLOT_COOLDOWN      = 4        # frames a slot must wait after death before reuse


# ---------------------------------------------
#  WAV Loading
# ---------------------------------------------
def load_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        n_channels  = wf.getnchannels()
        sampwidth   = wf.getsampwidth()
        sample_rate = wf.getframerate()
        raw         = wf.readframes(wf.getnframes())

    if sampwidth == 1:
        s = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        s = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8)
        i = (b[0::3].astype(np.int32) | (b[1::3].astype(np.int32) << 8) |
             (b[2::3].astype(np.int32) << 16))
        i[i >= 0x800000] -= 0x1000000
        s = i.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        s = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
        s = s.reshape(-1, n_channels).mean(axis=1)
    peak = np.max(np.abs(s))
    if peak > 1e-9: s /= peak
    return s, sample_rate


# ---------------------------------------------
#  Analysis State  (precomputed ONCE per run)
# ---------------------------------------------
class AnalysisState:
    def __init__(self, sample_rate: int, analysis_win: int = ANALYSIS_WIN, NW: float = 4.0):
        self.win       = analysis_win
        self.sr        = sample_rate

        # DPSS window (Discretely Prolate Spheroidal) for minimal leakage
        self.window    = windows.dpss(analysis_win, NW, sym=False).astype(np.float32)
        self.win_scale = 2.0 / float(np.sum(self.window))  # amplitude normalization

        # FFT frequencies
        self.freqs     = np.fft.rfftfreq(analysis_win, d=1.0 / sample_rate).astype(np.float32)
        self.bin_width = float(sample_rate) / analysis_win

        # Minimum bin distance for peak detection (~25 Hz)
        self.min_dist  = max(2, int(round(25.0 / self.bin_width)))

        # Nyquist frequency
        self.nyquist   = sample_rate / 2.0

        # Zero-padding buffer for edge frames
        self.pad_buf   = np.zeros(analysis_win, dtype=np.float32)

        self.erb = 21.4 * np.log10(4.37e-3 * self.freqs + 1)


# ---------------------------------------------
#  FFT Candidate Extraction
#  Returns (freqs, amps) only — phase dropped
# ---------------------------------------------
def _fft_candidates(
    audio: np.ndarray,
    center: int,
    state: AnalysisState,
    n_candidates: int,
) -> tuple[np.ndarray, np.ndarray]:
    half = state.win // 2
    s, e = center - half, center + half
    n    = len(audio)

    if s < 0 or e > n:
        chunk = state.pad_buf.copy()
        ss, se = max(0, s), min(n, e)
        chunk[ss - s : ss - s + (se - ss)] = audio[ss:se]
    else:
        chunk = audio[s:e]

    # ─── FFT + mag ───
    spec  = np.fft.rfft(chunk.astype(np.float64) * state.window)
    mags  = np.abs(spec).astype(np.float32) * state.win_scale

    # ─── HFC score ───
    hfc_scores = mags**2 * state.erb

    # ─── True local maxima ───
    peak_idx, _ = find_peaks(mags, distance=state.min_dist, height=1e-6)

    if len(peak_idx) == 0:
        # Fallback: very flat frame — take top n_candidates by HFC directly
        peak_idx = np.argpartition(hfc_scores, -n_candidates)[-n_candidates:]

    # ─── Sort peaks by HFC descending, oversample exactly 2× ───
    peak_hfc   = hfc_scores[peak_idx]
    sort_order = np.argsort(peak_hfc)[::-1]
    oversampled = peak_idx[sort_order][:n_candidates * 2]

    # ─── Freq filter (20 Hz – nyquist) ───
    f    = state.freqs[oversampled]
    mask = (f >= 20.0) & (f <= state.nyquist - state.bin_width)
    top  = oversampled[mask][:n_candidates]

    # ─── Fallback: freq filter ate too many peaks ───
    if len(top) < n_candidates:
        extra_needed = n_candidates - len(top)
        # Start after the full oversample window to avoid duplicates
        remaining = peak_idx[sort_order][n_candidates * 2:]
        extra     = remaining[:extra_needed]
        top       = np.concatenate([top, extra])

    ca = np.clip(mags[top], 0.0, 1.0)

    return state.freqs[top], ca


# ---------------------------------------------
#  Greedy Peak Tracker  (with slot cooldown)
# ---------------------------------------------
def _track_greedy(
    cand_f: np.ndarray, cand_a: np.ndarray,
    prev_f: np.ndarray, prev_a: np.ndarray,
    n_partials: int,
    cooldowns: np.ndarray,
    tol: float = 50.0,
    cooldown_frames: int = SLOT_COOLDOWN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_f = np.zeros(n_partials, dtype=np.float32)
    out_a = np.zeros(n_partials, dtype=np.float32)

    # Tick down all cooldowns at the top of every frame
    cooldowns = np.maximum(0, cooldowns - 1)

    if len(cand_f) == 0:
        return out_f, out_a, cooldowns

    claimed = np.zeros(len(cand_f), dtype=bool)

    # Continue active partials first (sorted highest amp first)
    active = np.where(prev_a > 1e-6)[0]
    active = active[np.argsort(prev_a[active])[::-1]]

    for slot in active:
        if claimed.all(): break
        dists = np.where(~claimed, np.abs(cand_f - prev_f[slot]), np.inf)
        bi    = int(np.argmin(dists))
        if dists[bi] <= tol:
            out_f[slot] = cand_f[bi]
            out_a[slot] = cand_a[bi]
            claimed[bi] = True
        else:
            # Partial died — lock slot before reuse
            cooldowns[slot] = cooldown_frames

    # Assign unclaimed births to empty cooled-down slots only
    births = np.where(~claimed)[0]
    if len(births):
        births   = births[np.argsort(cand_a[births])[::-1]]
        empty    = np.where((out_a == 0) & (cooldowns == 0))[0]
        n_assign = min(len(births), len(empty))
        bi_valid = births[:n_assign]
        mask     = (cand_a[bi_valid] > 1e-6) & (cand_f[bi_valid] > 1e-3)
        sl       = empty[:n_assign][mask]
        bi_v     = bi_valid[mask]
        out_f[sl] = cand_f[bi_v]
        out_a[sl] = cand_a[bi_v]

    return out_f, out_a, cooldowns


# ---------------------------------------------
#  RSC4 Binary Writer
#  Header : 23 bytes  (identical layout to RSC3, magic = "RSC4")
#  Partial: 4 bytes   uint16 freq | uint16 amp
#           (phase removed — saves 2 bytes/partial/frame = 33% smaller files)
# ---------------------------------------------
def write_rsc(
    path: str,
    frame_freqs: np.ndarray,   # (n_frames, n_partials) float32
    frame_amps:  np.ndarray,   # (n_frames, n_partials) float32
    sample_rate: int, frame_size: int, total_samples: int,
) -> None:
    n_frames, n_partials = frame_freqs.shape
    HEADER = 23

    buf = bytearray(HEADER + n_frames * n_partials * 4)
    struct.pack_into("<4sBIIHII", buf, 0,
                     b"RSC4", 4,
                     sample_rate, frame_size, n_partials,
                     total_samples, n_frames)

    freq_scale = 65535.0 / (sample_rate / 2.0)
    f16 = np.clip(np.round(frame_freqs * freq_scale), 0, 65535).astype(np.uint16)
    a16 = np.clip(np.round(frame_amps  * 65535.0),    0, 65535).astype(np.uint16)

    f_bytes = f16.view(np.uint8).reshape(n_frames, n_partials, 2)
    a_bytes = a16.view(np.uint8).reshape(n_frames, n_partials, 2)

    buf[HEADER:] = np.concatenate([f_bytes, a_bytes], axis=2).tobytes()

    with open(path, "wb") as fh:
        fh.write(buf)

    kb = len(buf) / 1024
    print(f"  ✅ Wrote {n_frames} frames -> {path}  ({kb:.1f} KB, {kb/60:.1f} KB/s)")


# ---------------------------------------------
#  Main Encode Pipeline
# ---------------------------------------------
def encode(input_path: str, output_path: str, n_partials: int, target_sr: int) -> None:
    print(f"RSC Encoder  --  {input_path}")
    print(f"   Partials/frame : {n_partials}  |  Target SR: {target_sr} Hz")

    samples, native_sr = load_wav(input_path)
    print(f"   Native SR      : {native_sr} Hz  |  {len(samples)} samples  "
          f"({len(samples)/native_sr:.2f}s)")

    if native_sr != target_sr:
        print(f"   Resampling {native_sr} -> {target_sr} Hz ...")
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(target_sr, native_sr)
        samples = resample_poly(samples, target_sr // g, native_sr // g).astype(np.float32)
        peak = np.max(np.abs(samples))
        if peak > 1e-9: samples /= peak

    sample_rate   = target_sr
    total_samples = len(samples)
    frame_size    = int(round(sample_rate / TARGET_FPS))
    n_frames      = math.ceil(total_samples / frame_size)
    pad           = n_frames * frame_size - total_samples
    if pad > 0:
        samples = np.concatenate([samples, np.zeros(pad, dtype=np.float32)])

    print(f"   Frame size     : {frame_size} samp  ({1000*frame_size/sample_rate:.2f} ms)"
          f"  |  {n_frames} frames")

    state  = AnalysisState(sample_rate)
    n_cand = min(n_partials * 2, 768)
    print(f"   Analysis win   : {ANALYSIS_WIN} samp  ({state.bin_width:.1f} Hz/bin)"
          f"  |  n_cand={n_cand}  cooldown={SLOT_COOLDOWN} frames")

    all_f = np.zeros((n_frames, n_partials), dtype=np.float32)
    all_a = np.zeros((n_frames, n_partials), dtype=np.float32)

    prev_f    = np.zeros(n_partials, dtype=np.float32)
    prev_a    = np.zeros(n_partials, dtype=np.float32)
    cooldowns = np.zeros(n_partials, dtype=np.int32)

    for i in range(n_frames):
        center             = i * frame_size + frame_size // 2
        cf, ca             = _fft_candidates(samples, center, state, n_cand)
        of, oa, cooldowns  = _track_greedy(cf, ca, prev_f, prev_a,
                                            n_partials, cooldowns)
        all_f[i] = of
        all_a[i] = oa
        prev_f   = of
        prev_a   = oa

        if (i + 1) % 500 == 0 or (i + 1) == n_frames:
            print(f"   ... encoded frame {i+1}/{n_frames}", end="\r")

    print()
    write_rsc(output_path, all_f, all_a, sample_rate, frame_size, total_samples)
    kb = (23 + n_frames * n_partials * 4) / 1024
    print(f"   {kb:.1f} KB  ({kb/1024:.3f} MB)  |  Done!")


# ---------------------------------------------
#  CLI
# ---------------------------------------------
def main():
    p = argparse.ArgumentParser(description="RSC Encoder",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input",      "-i", required=True)
    p.add_argument("--output",     "-o", default=None)
    p.add_argument("--partials",   "-n", type=int, default=DEFAULT_PARTIALS)
    p.add_argument("--samplerate", "-r", type=int, default=DEFAULT_SAMPLERATE,
                   choices=[22050, 44100])
    args = p.parse_args()
    out  = args.output or (args.input.removesuffix(".wav") + RSC_EXTENSION)
    encode(args.input, out, args.partials, args.samplerate)

if __name__ == "__main__":
    main()