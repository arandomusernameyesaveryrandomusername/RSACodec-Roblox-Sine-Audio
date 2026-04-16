#!/usr/bin/env python3
"""
RSC2 Audio Codec - Encoder (optimized, u16 amp+phase edition)

Binary format per frame (NEW):
  peak       f32be   per-frame max FFT magnitude (for decoder reconstruction)
  nPartials  u16be
  [ bin(u16be)  amp(u16be)  phase(u16be) ] × nPartials

amp   0-65535  →  0.0–1.0  normalised to per-frame peak
phase 0-65535  →  -π..π   (0 = -π, 65535 = +π)

6 bytes per partial (down from 10 with f32 amp+phase).
"""
import numpy as np
import struct
import wave
import argparse
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from numpy.fft import rfft
from tqdm import tqdm

# ── Try to JIT compile the scoring kernel with Numba ─────────────────────────
try:
    from numba import njit as _njit

    @_njit(cache=True, fastmath=True)
    def _score_numba(mags, score_n, score_hole):
        n = len(mags)
        score = np.empty(n, dtype=np.float32)
        for b in range(n):
            b_lo = max(0, b - score_n)
            b_hi = min(n - 1, b + score_n)
            lsum = np.float32(0.0)
            count = 0
            for k in range(b_lo, b_hi + 1):
                if k < (b - score_hole) or k > (b + score_hole):
                    lsum += mags[k]
                    count += 1
            lmean = lsum / (count + np.float32(1e-12))
            log_peak  = np.log(mags[b]  + np.float32(1e-12))
            log_floor = np.log(lmean    + np.float32(1e-12))
            t3 = log_peak - log_floor
            if t3 < 0.0:
                t3 = np.float32(0.0)
            if 0 < b < n - 1:
                t5 = log_peak - (np.log(mags[b-1] + np.float32(1e-12))
                                 + np.log(mags[b+1] + np.float32(1e-12))) * np.float32(0.5)
                if t5 < 0.0:
                    t5 = np.float32(0.0)
            else:
                t5 = np.float32(0.0)
            score[b] = mags[b] + t3 + t5
        return score

    _dummy = np.ones(513, dtype=np.float32)
    _score_numba(_dummy, 8, 3)
    _USE_NUMBA = True
    print("⚡ Numba JIT enabled for scoring kernel")

except Exception:
    _USE_NUMBA = False


def _score_numpy(mags: np.ndarray, score_n: int, score_hole: int) -> np.ndarray:
    """Fully vectorized scoring — no Python loops."""
    n    = len(mags)
    b_idx = np.arange(n)

    pad  = score_n
    mpad = np.pad(mags.astype(np.float64), pad, mode='constant')
    cs   = np.concatenate(([0.0], np.cumsum(mpad)))

    b_lo = np.maximum(0, b_idx - score_n)
    b_hi = np.minimum(n - 1, b_idx + score_n)

    win_sum = cs[b_hi + pad + 1] - cs[b_lo + pad]
    win_cnt = b_hi - b_lo + 1

    h_lo     = np.maximum(b_lo, b_idx - score_hole)
    h_hi     = np.minimum(b_hi, b_idx + score_hole)
    has_hole = h_lo <= h_hi
    hole_sum = np.where(has_hole, cs[h_hi + pad + 1] - cs[h_lo + pad], 0.0)
    hole_cnt = np.where(has_hole, h_hi - h_lo + 1, 0)

    lmean = (win_sum - hole_sum) / np.maximum(win_cnt - hole_cnt, 1).astype(np.float64)

    log_peak  = np.log(mags.astype(np.float64) + 1e-12)
    log_floor = np.log(lmean + 1e-12)
    t3 = np.maximum(0.0, log_peak - log_floor)

    log_m = log_peak
    t5    = np.zeros(n, dtype=np.float64)
    if n > 2:
        inner    = slice(1, n - 1)
        t5[inner] = np.maximum(0.0, log_m[inner] - 0.5 * (log_m[:-2] + log_m[2:]))

    return ((t3 + t5) * mags).astype(np.float32)


def compute_scores(mags: np.ndarray, score_n: int = 8, score_hole: int = 3) -> np.ndarray:
    if _USE_NUMBA:
        return _score_numba(mags, score_n, score_hole)
    return _score_numpy(mags, score_n, score_hole)





# ── Constants ─────────────────────────────────────────────────────────────────
RSC2_MAGIC   = b"RSC2"
RSC2_VERSION = 1
FFT_SIZE     = 1024
HOP_SIZE     = 1024
MAX_PARTIALS = 192
SCORE_N      = 6
SCORE_HOLE   = 2

HEADER_FMT  = "<BBIHHHIIf"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# Per-frame header: peak magnitude (f32) + partial count (u16)
FRAME_HDR_FMT  = "<fH"
FRAME_HDR_SIZE = struct.calcsize(FRAME_HDR_FMT)   # 6

# Per-partial record: bin(u16) amp(u16) phase(u16) = 6 bytes
PARTIAL_FMT  = "<HHH"
PARTIAL_SIZE = struct.calcsize(PARTIAL_FMT)        # 6

_TWO_PI     = 2.0 * np.pi
_INV_TWO_PI = 1.0 / _TWO_PI
_U16_MAX    = 65535.0


def _encode_channel(args):
    """Encode a single channel. Designed to run in a thread pool.

    Returns:
        all_frames  – list of (idx, amp_u16, ph_u16, peak_f32)
        n_frames    – int
    """
    sig, win, n_bins, fft_size, hop_size, max_partials = args

    n_samples = len(sig)
    n_frames  = max(0, (n_samples - fft_size) // hop_size + 1)

    # Zero-copy strided view → windowed frames
    frame_shape   = (n_frames, fft_size)
    frame_strides = (sig.strides[0] * hop_size, sig.strides[0])
    frames_raw    = np.lib.stride_tricks.as_strided(sig, shape=frame_shape,
                                                    strides=frame_strides)
    # OPTIMIZATION 1: Skip window multiply if rectangular (all 1.0)
    if np.allclose(win, 1.0):
        frames_win = frames_raw
    else:
        frames_win = frames_raw * win[np.newaxis, :]          # (n_frames, fft_size)

    # Batch FFT
    spec = rfft(frames_win, n=fft_size, axis=1)           # (n_frames, n_bins)
    mags = np.abs(spec).astype(np.float32)                # (n_frames, n_bins)
    phs  = np.angle(spec).astype(np.float32)              # (n_frames, n_bins)  -π..π

    k          = min(max_partials, n_bins)
    all_frames = []
    
    # OPTIMIZATION 2: Pre-allocate reusable buffers to reduce per-frame allocation overhead
    amp_u16_buf  = np.empty(k, dtype=np.uint16)
    ph_u16_buf   = np.empty(k, dtype=np.uint16)

    for fi in range(n_frames):
        sc  = compute_scores(mags[fi], SCORE_N, SCORE_HOLE)
        # OPTIMIZATION 3: Use argsort[-k:] for better cache efficiency vs argpartition+argsort combo
        idx = np.argsort(sc)[-k:][::-1]                  # top-k descending

        # ── Per-frame amplitude normalisation → u16 ──────────────────────────
        sel_mags = mags[fi][idx]
        peak     = float(sel_mags.max()) if len(sel_mags) else 0.0
        if peak < 1e-12:
            peak = 1.0   # silent frame: store ones, decoder will produce silence

        # Use pre-allocated buffer with in-place operation (no extra allocation)
        n_p = len(idx)
        np.divide(sel_mags, peak, out=sel_mags)
        np.multiply(sel_mags, _U16_MAX, out=sel_mags)
        np.clip(sel_mags, 0.0, _U16_MAX, out=sel_mags)
        amp_u16_buf[:n_p] = sel_mags.round().astype(np.uint16)

        # ── Phase -π..π → 0..65535 ───────────────────────────────────────────
        sel_phs = phs[fi][idx]
        np.add(sel_phs, np.pi, out=sel_phs)
        np.multiply(sel_phs, _INV_TWO_PI * _U16_MAX, out=sel_phs)
        np.clip(sel_phs, 0.0, _U16_MAX, out=sel_phs)
        ph_u16_buf[:n_p] = sel_phs.round().astype(np.uint16)

        all_frames.append((idx, amp_u16_buf[:n_p].copy(), ph_u16_buf[:n_p].copy(), peak))

    return all_frames, n_frames


def encode(pcm: np.ndarray, sample_rate: int, n_channels: int) -> bytes:
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    n_samples, channels = pcm.shape

    win     = np.ones(FFT_SIZE, dtype=np.float32)
    win_sum = float(win.sum())                            # = FFT_SIZE for rect window
    n_bins  = FFT_SIZE // 2 + 1

    channel_args = [
        (pcm[:, ch].astype(np.float32), win, n_bins,
         FFT_SIZE, HOP_SIZE, MAX_PARTIALS)
        for ch in range(channels)
    ]

    pbar = tqdm(total=channels, desc="Encoding channels", unit="ch", mininterval=1/60, smoothing=0.1)
    
    # OPTIMIZATION 4: Cap ThreadPoolExecutor workers at 4 to avoid thread overhead
    optimal_workers = min(channels, 4)
    with ThreadPoolExecutor(max_workers=optimal_workers) as ex:
        results = []
        for future in ex.map(_encode_channel, channel_args):
            results.append(future)
            pbar.update(1)
    
    pbar.close()
    n_frames = results[0][1]

    # Pre-allocate output buffer (exact size)
    buf_size = (
        4                                               # magic
        + HEADER_SIZE                                   # global header
        + channels * n_frames * FRAME_HDR_SIZE          # peak + nPartials per frame
        + channels * n_frames * MAX_PARTIALS * PARTIAL_SIZE  # partial records
    )
    buf = bytearray(buf_size)
    off = 0

    # Magic
    buf[off:off + 4] = RSC2_MAGIC
    off += 4

    # Global header
    struct.pack_into(HEADER_FMT, buf, off,
                     RSC2_VERSION, channels, sample_rate,
                     FFT_SIZE, HOP_SIZE, MAX_PARTIALS,
                     n_frames, n_samples, win_sum)
    off += HEADER_SIZE

    # Frame data — channel-major order
    pbar_write = tqdm(total=channels * n_frames, desc="Writing frames", unit="frames", mininterval=1/60, smoothing=0.1)
    for ch_frames, _ in results:
        for idx, amp_u16, ph_u16, peak in ch_frames:
            n_p = len(idx)
            # Frame header: peak magnitude + partial count
            struct.pack_into(FRAME_HDR_FMT, buf, off, peak, n_p)
            off += FRAME_HDR_SIZE
            
            # OPTIMIZATION 5: Vectorized binary writing using numpy instead of per-partial struct.pack loops
            # This eliminates per-call FFI overhead and copies all partials in one memcpy
            partial_buffer = np.empty((n_p, 3), dtype='<u2')
            partial_buffer[:, 0] = idx[:n_p]
            partial_buffer[:, 1] = amp_u16[:n_p]
            partial_buffer[:, 2] = ph_u16[:n_p]
            
            bytes_data = partial_buffer.tobytes()
            buf[off:off + len(bytes_data)] = bytes_data
            off += len(bytes_data)
            pbar_write.update(1)
    
    pbar_write.close()
    return bytes(buf[:off])


def load_wav(path: str):
    with wave.open(path, "rb") as wf:
        sr, ch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw == 2:
        s = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        s = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width {sw}")
    return s.reshape(-1, ch), sr, ch


def main():
    ap = argparse.ArgumentParser(description="RSC2 encoder (u16 amp+phase)")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--no-numba", action="store_true",
                    help="Force pure-numpy scoring even if Numba is available")
    a = ap.parse_args()

    if a.no_numba:
        global _USE_NUMBA
        _USE_NUMBA = False
        print("🔧 Numba disabled (--no-numba)")

    print(f"🎵 Loading {a.input}…")
    pcm, sr, ch = load_wav(a.input)
    print(f"   {sr} Hz | {ch} ch | {len(pcm)} samples")
    print(f"⚙️  Encoding (FFT={FFT_SIZE}, hop={HOP_SIZE}, max_partials={MAX_PARTIALS})…")

    t0   = time.perf_counter()
    data = encode(pcm, sr, ch)
    dt   = time.perf_counter() - t0

    with open(a.output, "wb") as f:
        f.write(data)

    ratio = (len(pcm) / sr) / dt if dt > 0 else float('inf')
    print(f"✅ Written {len(data):,} bytes → {a.output}")
    print(f"   Encoded in {dt * 1000:.1f} ms  ({ratio:.1f}× real-time)")
    print(f"   Partial record: {PARTIAL_SIZE} bytes  (u16 bin + u16 amp + u16 phase)")


if __name__ == "__main__":
    main()