"""
rsc_encoder.py — Roblox Sine Codec (RSC) Encoder  [optimized]

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
try:
    from numba import njit as _njit
    _NUMBA = True
except ImportError:
    def _njit(**kw):
        def decorator(fn): return fn
        return decorator
    _NUMBA = False


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
TARGET_FPS         = 120
DEFAULT_PARTIALS   = 384
DEFAULT_SAMPLERATE = 44100
RSC_EXTENSION      = ".rsc"
ANALYSIS_WIN       = 4096     # ~10.8 Hz/bin at 44100


# ─────────────────────────────────────────────
#  WAV Loading
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
#  Analysis State  (precomputed ONCE per run)
# ─────────────────────────────────────────────
class AnalysisState:
    """
    Precomputes all FFT constants so _fft_candidates() is pure computation.
    Previously np.hanning(4096) and rfftfreq(4096) were called every frame —
    at 3600 frames/min that's 7200 wasted allocations.
    """
    def __init__(self, sample_rate: int, analysis_win: int = ANALYSIS_WIN):
        self.win       = analysis_win
        self.sr        = sample_rate
        self.window    = np.blackman(analysis_win).astype(np.float32)
        self.win_scale = 2.0 / float(np.sum(self.window))
        self.freqs     = np.fft.rfftfreq(analysis_win, d=1.0 / sample_rate).astype(np.float32)
        self.bin_width = float(sample_rate) / analysis_win
        self.min_dist  = max(1, int(20.0 / self.bin_width))
        self.nyquist   = sample_rate / 2.0
        self.pad_buf   = np.zeros(analysis_win, dtype=np.float32)
        # Pre-allocated interpolation buffers — reused every frame, no heap alloc
        # Sized to max possible candidates (n_partials * 6, set at encode time)
        self._ibuf_f   = None   # set by encoder after n_cand is known
        self._ibuf_a   = None
        self._ibuf_p   = None


# ─────────────────────────────────────────────
#  FFT Candidate Extraction
# ─────────────────────────────────────────────
def _fft_candidates(
    audio: np.ndarray,
    center: int,
    state: AnalysisState,
    n_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (freqs, amps, phases) as float32 numpy arrays — no list boxing.
    Uses a pre-allocated pad buffer so boundary frames don't heap-allocate.
    """
    half = state.win // 2
    s, e = center - half, center + half
    n    = len(audio)

    if s < 0 or e > n:
        chunk = state.pad_buf.copy()
        ss, se = max(0, s), min(n, e)
        chunk[ss - s : ss - s + (se - ss)] = audio[ss:se]
    else:
        chunk = audio[s:e]

    # float64 for FFT numerical accuracy, then back to float32
    spec  = np.fft.rfft(chunk.astype(np.float64) * state.window)
    mags  = np.abs(spec).astype(np.float32) * state.win_scale
    phs   = np.angle(spec).astype(np.float32)

    # Vectorized peak detection: O(n) numpy comparison beats find_peaks for speed
    # A peak is any bin greater than both its neighbours AND above threshold
    above  = mags > 1e-7
    is_peak = (mags[1:-1] > mags[:-2]) & (mags[1:-1] > mags[2:]) & above[1:-1]
    peak_idx = np.where(is_peak)[0] + 1   # +1 to correct for sliced offset

    # Enforce minimum distance between peaks (same as find_peaks distance= param)
    if state.min_dist > 1 and len(peak_idx) > 1:
        # Walk through and suppress any peak within min_dist of a louder neighbour
        keep = np.ones(len(peak_idx), dtype=bool)
        for j in range(len(peak_idx)):
            if not keep[j]: continue
            for k in range(j + 1, len(peak_idx)):
                if peak_idx[k] - peak_idx[j] >= state.min_dist: break
                if mags[peak_idx[k]] > mags[peak_idx[j]]:
                    keep[j] = False
                else:
                    keep[k] = False
        peak_idx = peak_idx[keep]

    if len(peak_idx) == 0:
        peak_idx = np.argpartition(mags, -min(n_candidates, len(mags)))[-n_candidates:]

    # Top n_candidates by magnitude — argpartition is O(n) vs argsort O(n log n)
    pk = mags[peak_idx]
    if len(pk) > n_candidates:
        top = peak_idx[np.argpartition(pk, -n_candidates)[-n_candidates:]]
    else:
        top = peak_idx

    # Quadratic peak interpolation — write into pre-allocated buffers (no heap alloc)
    bw     = state.bin_width
    n_bins = len(mags)
    ntop   = len(top)
    buf_f  = state._ibuf_f[:ntop]
    buf_a  = state._ibuf_a[:ntop]
    buf_p  = state._ibuf_p[:ntop]

    for j, k in enumerate(top):
        if 1 <= k < n_bins - 1:
            alpha = float(mags[k - 1])
            beta  = float(mags[k])
            gamma = float(mags[k + 1])
            denom = alpha - 2.0 * beta + gamma
            dp    = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-12 else 0.0
            buf_f[j] = float(state.freqs[k]) + dp * bw
            buf_a[j] = beta - 0.25 * (alpha - gamma) * dp
        else:
            buf_f[j] = float(state.freqs[k])
            buf_a[j] = float(mags[k])
        buf_p[j] = float(phs[k])

    np.clip(buf_a, 0.0, 1.0, out=buf_a)
    mask = (buf_f[:ntop] >= 8.0) & (buf_f[:ntop] <= state.nyquist * 0.98)
    return buf_f[:ntop][mask].copy(), buf_a[:ntop][mask].copy(), buf_p[:ntop][mask].copy()


# ─────────────────────────────────────────────
#  Greedy Peak Tracker
# ─────────────────────────────────────────────
@_njit(nopython=True, cache=True)
def _track_greedy_kernel(
    cand_f: np.ndarray, cand_a: np.ndarray, cand_p: np.ndarray,
    prev_f: np.ndarray, prev_a: np.ndarray,
    n_partials: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-compiled greedy nearest-frequency matching kernel.
    """
    out_f = np.zeros(n_partials, dtype=np.float32)
    out_a = np.zeros(n_partials, dtype=np.float32)
    out_p = np.zeros(n_partials, dtype=np.float32)
    n_cand = len(cand_f)
    if n_cand == 0:
        return out_f, out_a, out_p

    claimed = np.zeros(n_cand, dtype=np.bool_)

    # Build active list sorted by amplitude descending
    active_idx = np.where(prev_a > 1e-9)[0]
    for i in range(1, len(active_idx)):
        key = active_idx[i]
        j   = i - 1
        while j >= 0 and prev_a[active_idx[j]] < prev_a[key]:
            active_idx[j + 1] = active_idx[j]
            j -= 1
        active_idx[j + 1] = key

    for ii in range(len(active_idx)):
        slot   = active_idx[ii]
        base_rel = 0.04 * prev_f[slot]
        amp_factor = 1.4 - 0.8 * prev_a[slot]
        tol=max(8.0, base_rel * amp_factor)
        best_i = -1
        best_d = tol + 1.0
        for ci in range(n_cand):
            if claimed[ci]: continue
            d = abs(cand_f[ci] - prev_f[slot])
            if d < best_d:
                best_d = d
                best_i = ci
        if best_i >= 0 and best_d <= tol:
            out_f[slot] = cand_f[best_i]
            out_a[slot] = cand_a[best_i]
            out_p[slot] = cand_p[best_i]
            claimed[best_i] = True

    # Births → empty slots, loudest first
    # Collect unclaimed, sort by amp desc
    n_births = 0
    birth_idx = np.empty(n_cand, dtype=np.int64)
    for ci in range(n_cand):
        if not claimed[ci] and cand_a[ci] > 1e-10:
            birth_idx[n_births] = ci
            n_births += 1
    birth_idx = birth_idx[:n_births]
    # Sort births by amplitude descending
    for i in range(1, n_births):
        key = birth_idx[i]
        j   = i - 1
        while j >= 0 and cand_a[birth_idx[j]] < cand_a[key]:
            birth_idx[j + 1] = birth_idx[j]
            j -= 1
        birth_idx[j + 1] = key

    # Voice stealing: sort slots by current amplitude ascending,
    # then let loud new births overwrite the quietest current partials.
    # Small hysteresis (0.005) prevents flip-flopping on near-equal partials.
    slot_order = np.argsort(out_a)   # quietest slot first
    bi = 0
    for si in range(n_partials):
        if bi >= n_births: break
        slot = slot_order[si]
        if cand_a[birth_idx[bi]] > out_a[slot] + 0.005:
            out_f[slot] = cand_f[birth_idx[bi]]
            out_a[slot] = cand_a[birth_idx[bi]]
            out_p[slot] = cand_p[birth_idx[bi]]
            bi += 1

    return out_f, out_a, out_p


def _track_greedy(
    cand_f: np.ndarray, cand_a: np.ndarray, cand_p: np.ndarray,
    prev_f: np.ndarray, prev_a: np.ndarray,
    n_partials: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _track_greedy_kernel(cand_f, cand_a, cand_p, prev_f, prev_a, n_partials)


# ─────────────────────────────────────────────
#  RSC Binary Writer  (fully vectorized)
# ─────────────────────────────────────────────
def write_rsc(
    path: str,
    frame_freqs:  np.ndarray,   # (n_frames, n_partials) float32
    frame_amps:   np.ndarray,   # (n_frames, n_partials) float32
    frame_phases: np.ndarray,   # (n_frames, n_partials) float32
    sample_rate: int, frame_size: int, total_samples: int,
) -> None:
    n_frames, n_partials = frame_freqs.shape
    HEADER = 23

    buf = bytearray(HEADER + n_frames * n_partials * 4)
    struct.pack_into("<4sBIIHII", buf, 0,
                     b"RSC2", 2,
                     sample_rate, frame_size, n_partials,
                     total_samples, n_frames)

    freq_scale = 65535.0 / (sample_rate / 2.0)
    f16 = np.clip(np.round(frame_freqs  * freq_scale),        0, 65535).astype(np.uint16)
    a8  = np.clip(np.round(np.sqrt(frame_amps) * 255.0),       0,   255).astype(np.uint8)
    p8  = np.clip(np.round(frame_phases / math.pi * 127.0), -128,   127).astype(np.int8)

    f_bytes = f16.view(np.uint8).reshape(n_frames, n_partials, 2)
    a_bytes = a8.reshape(n_frames, n_partials, 1)
    p_bytes = p8.view(np.uint8).reshape(n_frames, n_partials, 1)

    buf[HEADER:] = np.concatenate([f_bytes, a_bytes, p_bytes], axis=2).tobytes()

    with open(path, "wb") as fh:
        fh.write(buf)

    kb = len(buf) / 1024
    print(f"  ✅ Wrote {n_frames} frames → {path}  ({kb:.1f} KB, {kb/60:.1f} KB/s)")


# ─────────────────────────────────────────────
#  Main Encode Pipeline
# ─────────────────────────────────────────────
def encode(input_path: str, output_path: str, n_partials: int, target_sr: int) -> None:
    print(f"🎵 RSC Encoder  —  {input_path}")
    print(f"   Partials/frame : {n_partials}  |  Target SR: {target_sr} Hz")

    samples, native_sr = load_wav(input_path)
    print(f"   Native SR      : {native_sr} Hz  |  {len(samples)} samples  "
          f"({len(samples)/native_sr:.2f}s)")

    if native_sr != target_sr:
        print(f"   Resampling {native_sr} → {target_sr} Hz …")
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
    pad            = n_frames * frame_size - total_samples
    if pad > 0:
        samples = np.concatenate([samples, np.zeros(pad, dtype=np.float32)])

    print(f"   Frame size     : {frame_size} samp  ({1000*frame_size/sample_rate:.2f} ms)"
          f"  |  {n_frames} frames")

    state  = AnalysisState(sample_rate)
    n_cand = n_partials * 6
    # Initialise pre-alloc interp buffers now that n_cand is known
    state._ibuf_f = np.empty(n_cand, dtype=np.float32)
    state._ibuf_a = np.empty(n_cand, dtype=np.float32)
    state._ibuf_p = np.empty(n_cand, dtype=np.float32)
    print(f"   Analysis win   : {ANALYSIS_WIN} samp  ({state.bin_width:.1f} Hz/bin)")

    all_f = np.zeros((n_frames, n_partials), dtype=np.float32)
    all_a = np.zeros((n_frames, n_partials), dtype=np.float32)
    all_p = np.zeros((n_frames, n_partials), dtype=np.float32)

    prev_f = np.zeros(n_partials, dtype=np.float32)
    prev_a = np.zeros(n_partials, dtype=np.float32)

    if _NUMBA:
        print("   🔥 Numba JIT warm-up (first frame only) …")
    for i in range(n_frames):
        center = i * frame_size + frame_size // 2
        cf, ca, cp = _fft_candidates(samples, center, state, n_cand)
        of, oa, op = _track_greedy(cf, ca, cp, prev_f, prev_a, n_partials)
        all_f[i] = of;  all_a[i] = oa;  all_p[i] = op
        prev_f = of;    prev_a = oa

        if (i + 1) % 500 == 0 or (i + 1) == n_frames:
            print(f"   … encoded frame {i+1}/{n_frames}", end="\r")

    print()
    write_rsc(output_path, all_f, all_a, all_p, sample_rate, frame_size, total_samples)
    kb = (22 + n_frames * n_partials * 4) / 1024
    print(f"   📦 {kb:.1f} KB  ({kb/1024:.3f} MB)  |  🎉 Done!")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
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