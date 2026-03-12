"""
rsc_encoder.py -- Roblox Sine Codec (RSC) Encoder  [optimized]

Usage:
    python rsc_encoder.py --input audio.wav --output audio.rsc
    python rsc_encoder.py --input audio.wav --output audio.rsc --partials 384 --samplerate 44100
"""

import argparse
import math
import struct
import wave

import numpy as np
from scipy.signal import windows


# ---------------------------------------------
#  Constants
# ---------------------------------------------
TARGET_FPS         = 60
DEFAULT_PARTIALS   = 384
DEFAULT_SAMPLERATE = 44100
RSC_EXTENSION      = ".rsc"
ANALYSIS_WIN       = 4096     # ~10.8 Hz/bin at 44100
UINT32_MAX         = 0xFFFFFFFF
HEADER             = 23


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
        if len(raw) % 3 != 0:
            raise ValueError(f"24-bit PCM data length {len(raw)} is not divisible by 3")
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

        self.window    = windows.dpss(analysis_win, NW, sym=False).astype(np.float64)
        self.win_scale = 2.0 / float(np.sum(self.window))

        self.freqs     = np.fft.rfftfreq(analysis_win, d=1.0 / sample_rate).astype(np.float32)
        self.bin_width = float(sample_rate) / analysis_win

        self.nyquist   = sample_rate / 2.0

        self.pad_buf   = np.zeros(analysis_win, dtype=np.float64)

        self.erb = (21.4 * np.log10(4.37e-3 * self.freqs + 1)).astype(np.float32)

        self.freq_mask = (self.freqs >= 20.0) & (self.freqs <= self.nyquist - self.bin_width)


# ---------------------------------------------
#  FFT  —  return ALL bins in the valid freq range, no selection
# ---------------------------------------------
def _fft_all_bins(
    audio: np.ndarray,
    center: int,
    state: AnalysisState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    half = state.win // 2
    s, e = center - half, center + half
    n    = len(audio)

    if s < 0 or e > n:
        chunk = state.pad_buf.copy()
        ss, se = max(0, s), min(n, e)
        chunk[ss - s : ss - s + (se - ss)] = audio[ss:se]
    else:
        chunk = audio[s:e].astype(np.float64)

    spec = np.fft.rfft(chunk * state.window)
    mags = (np.abs(spec) * state.win_scale).astype(np.float32)
    phs  = np.angle(spec).astype(np.float32)

    mask = state.freq_mask
    return state.freqs[mask], mags[mask], phs[mask], state.erb[mask]


# ---------------------------------------------
#  ERB Grid Selector
# ---------------------------------------------
def _erb_grid_select(
    freqs: np.ndarray,
    mags:  np.ndarray,
    phs:   np.ndarray,
    erb:   np.ndarray,
    n_partials: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    erb_min   = float(erb[0])
    erb_max   = float(erb[-1])
    erb_edges = np.linspace(erb_min, erb_max, n_partials + 1)

    scores = mags ** 2 * erb

    cell_idx = np.clip(
        np.searchsorted(erb_edges, erb, side='right') - 1,
        0, n_partials - 1
    )

    best_score = np.full(n_partials, -1.0, dtype=np.float32)
    np.maximum.at(best_score, cell_idx, scores)

    winner_bin = np.full(n_partials, -1, dtype=np.int64)
    is_winner  = scores == best_score[cell_idx]
    winner_idx = np.where(is_winner)[0]
    winner_bin[cell_idx[winner_idx]] = winner_idx
    winner_bin[:] = -1
    for i in range(len(freqs)):
        ci = cell_idx[i]
        if best_score[ci] > -1.0 and scores[i] == best_score[ci] and winner_bin[ci] == -1:
            winner_bin[ci] = i

    out_f = np.zeros(n_partials, dtype=np.float32)
    out_a = np.zeros(n_partials, dtype=np.float32)
    out_p = np.zeros(n_partials, dtype=np.float32)

    valid = (winner_bin >= 0)
    bi    = winner_bin[valid]
    out_f[valid] = freqs[bi]
    out_a[valid] = np.clip(mags[bi], 0.0, 1.0)
    out_p[valid] = phs[bi]

    return out_f, out_a, out_p


# ---------------------------------------------
#  Greedy Peak Tracker
# ---------------------------------------------
def _track_greedy(
    cand_f: np.ndarray, cand_a: np.ndarray, cand_p: np.ndarray,
    prev_f: np.ndarray, prev_a: np.ndarray,
    n_partials: int,
    cooldowns: np.ndarray,
    tol: float = 50.0,
    cooldown_frames: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    out_f = np.zeros(n_partials, dtype=np.float32)
    out_a = np.zeros(n_partials, dtype=np.float32)
    out_p = np.zeros(n_partials, dtype=np.float32)

    cooldowns[:] = np.maximum(0, cooldowns - 1)

    if len(cand_f) == 0:
        return out_f, out_a, out_p, cooldowns

    claimed = np.zeros(len(cand_f), dtype=bool)

    active = np.where(prev_a > 1e-6)[0]
    active = active[np.argsort(prev_a[active])[::-1]]

    for slot in active:
        if claimed.all(): break
        dists = np.where(~claimed, np.abs(cand_f - prev_f[slot]), np.inf)
        bi    = int(np.argmin(dists))
        if dists[bi] <= tol:
            out_f[slot] = cand_f[bi]
            out_a[slot] = cand_a[bi]
            out_p[slot] = cand_p[bi]
            claimed[bi] = True
        else:
            cooldowns[slot] = cooldown_frames

    births = np.where(~claimed)[0]
    if len(births):
        births = births[np.argsort(cand_a[births])[::-1]]
        empty  = np.where((out_a == 0) & (cooldowns == 0))[0]

        n_assign = min(len(births), len(empty))
        bi_valid = births[:n_assign]
        em_valid = empty[:n_assign]
        mask     = (cand_a[bi_valid] > 1e-6) & (cand_f[bi_valid] > 1e-3)
        sl       = em_valid[mask]
        bi_v     = bi_valid[mask]
        out_f[sl] = cand_f[bi_v]
        out_a[sl] = cand_a[bi_v]
        out_p[sl] = cand_p[bi_v]

    return out_f, out_a, out_p, cooldowns


# ---------------------------------------------
#  RSC3 Binary Writer
# ---------------------------------------------
def write_rsc(
    path: str,
    frame_freqs:  np.ndarray,
    frame_amps:   np.ndarray,
    frame_phases: np.ndarray,
    sample_rate: int, frame_size: int, total_samples: int,
) -> None:
    n_frames, n_partials = frame_freqs.shape

    if total_samples > UINT32_MAX:
        raise ValueError(f"total_samples {total_samples} exceeds uint32 max — file too long")

    buf = bytearray(HEADER + n_frames * n_partials * 6)
    struct.pack_into("<4sBIIHII", buf, 0,
                     b"RSC3", 3,
                     sample_rate, frame_size, n_partials,
                     total_samples, n_frames)

    freq_scale = 65535.0 / (sample_rate / 2.0)
    f16 = np.clip(np.round(frame_freqs  * freq_scale),           0, 65535).astype(np.uint16)
    a16 = np.clip(np.round(frame_amps   * 65535.0),              0, 65535).astype(np.uint16)
    p16 = np.clip(np.round(frame_phases / math.pi * 32767.0), -32768, 32767).astype(np.int16)

    f_bytes = f16.view(np.uint8).reshape(n_frames, n_partials, 2)
    a_bytes = a16.view(np.uint8).reshape(n_frames, n_partials, 2)
    p_bytes = p16.view(np.uint8).reshape(n_frames, n_partials, 2)

    # FIX 1: axis=2 → shape (n_frames, n_partials, 3, 2) → reshape → (n_frames, n_partials, 6)
    # per-partial layout: [f0, f1, a0, a1, p0, p1] ✅
    interleaved = np.stack([f_bytes, a_bytes, p_bytes], axis=2).reshape(n_frames, n_partials, 6)
    buf[HEADER:] = interleaved.tobytes()

    with open(path, "wb") as fh:
        fh.write(buf)


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
        from scipy.signal import resample_poly
        g       = math.gcd(target_sr, native_sr)
        samples = resample_poly(samples, target_sr // g, native_sr // g).astype(np.float32)
        peak    = np.max(np.abs(samples))
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
    n_bins = int(np.sum(state.freq_mask))
    print(f"   Analysis win   : {ANALYSIS_WIN} samp  ({state.bin_width:.1f} Hz/bin)"
          f"  |  {n_bins} bins -> {n_partials} partials via ERB grid")

    all_f = np.zeros((n_frames, n_partials), dtype=np.float32)
    all_a = np.zeros((n_frames, n_partials), dtype=np.float32)
    all_p = np.zeros((n_frames, n_partials), dtype=np.float32)

    prev_f    = np.zeros(n_partials, dtype=np.float32)
    prev_a    = np.zeros(n_partials, dtype=np.float32)
    cooldowns = np.zeros(n_partials, dtype=np.int32)

    for i in range(n_frames):
        center                = i * frame_size + frame_size // 2
        cf, ca, cp, ce        = _fft_all_bins(samples, center, state)
        cf, ca, cp            = _erb_grid_select(cf, ca, cp, ce, n_partials)
        of, oa, op, cooldowns = _track_greedy(cf, ca, cp, prev_f, prev_a,
                                               n_partials, cooldowns)
        all_f[i] = of;  all_a[i] = oa;  all_p[i] = op
        prev_f = of;    prev_a = oa

        if (i + 1) % 500 == 0 or (i + 1) == n_frames:
            print(f"   ... encoded frame {i+1}/{n_frames}", end="\r")

    print()
    write_rsc(output_path, all_f, all_a, all_p, sample_rate, frame_size, total_samples)

    kb = (HEADER + n_frames * n_partials * 6) / 1024
    print(f"  ✅ Wrote {n_frames} frames -> {output_path}  ({kb:.1f} KB, {kb/60:.1f} KB/s)")
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