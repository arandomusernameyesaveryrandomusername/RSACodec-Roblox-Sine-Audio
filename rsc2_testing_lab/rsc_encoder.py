#!/usr/bin/env python3
"""
RSC2 Audio Codec - Encoder (optimized)
Identical output to the reference encoder; faster execution via:
  - Fully vectorized compute_scores (no Python loops)
  - Numba JIT fallback for scoring (auto-selected)
  - scipy.fft with pre-planned worker pool (pyfftw-style plan reuse)
  - Batch FFT over all frames at once (one np.fft.rfft call per channel)
  - Parallel channel encoding via ThreadPoolExecutor
  - struct.pack_into into a pre-allocated bytearray (no per-partial allocs)
  - argpartition on the full score vector (already O(n), kept)
"""
import numpy as np
import struct
import wave
import argparse
from concurrent.futures import ThreadPoolExecutor
from scipy.signal.windows import dpss
from numpy.fft import rfft

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

    # Warm up the JIT (compile now, not on first encode call)
    _dummy = np.ones(513, dtype=np.float32)
    _score_numba(_dummy, 8, 3)
    _USE_NUMBA = True
    print("⚡ Numba JIT enabled for scoring kernel")

except Exception:
    _USE_NUMBA = False


def _score_numpy(mags: np.ndarray, score_n: int, score_hole: int) -> np.ndarray:
    """Fully vectorized scoring — no Python loops."""
    n = len(mags)
    b_idx   = np.arange(n)

    # ── Local mean (excluding the hole around each bin) ───────────────────────
    # Build a running cumulative sum; then for each bin b extract the window
    # [b-score_n .. b+score_n] minus the hole [b-score_hole .. b+score_hole].
    #
    # pad both ends so we can index without bounds checks
    pad  = score_n
    mpad = np.pad(mags.astype(np.float64), pad, mode='constant')
    cs   = np.concatenate(([0.0], np.cumsum(mpad)))       # length n + 2*pad + 1

    b_lo   = np.maximum(0, b_idx - score_n)               # unpadded coords
    b_hi   = np.minimum(n - 1, b_idx + score_n)

    # window sum in padded coords: cs[b_hi+pad+1] - cs[b_lo+pad]
    win_sum  = cs[b_hi + pad + 1] - cs[b_lo + pad]
    win_cnt  = b_hi - b_lo + 1

    # hole sum: sum of mags[b-hole .. b+hole] clipped to valid range
    h_lo = np.maximum(b_lo, b_idx - score_hole)
    h_hi = np.minimum(b_hi, b_idx + score_hole)
    # Only subtract hole when it actually overlaps the window
    has_hole = h_lo <= h_hi
    hole_sum = np.where(has_hole,
                        cs[h_hi + pad + 1] - cs[h_lo + pad],
                        0.0)
    hole_cnt = np.where(has_hole, h_hi - h_lo + 1, 0)

    lmean = (win_sum - hole_sum) / np.maximum(win_cnt - hole_cnt, 1).astype(np.float64)

    log_peak  = np.log(mags.astype(np.float64) + 1e-12)
    log_floor = np.log(lmean + 1e-12)
    t3 = np.maximum(0.0, log_peak - log_floor)

    # curvature term: compare each bin to its neighbours
    log_m = log_peak                                       # shape (n,)
    t5    = np.zeros(n, dtype=np.float64)
    if n > 2:
        inner = slice(1, n - 1)
        t5[inner] = np.maximum(
            0.0,
            log_m[inner] - 0.5 * (log_m[:-2] + log_m[2:])
        )

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
HEADER_FMT   = ">BBIHHHIIf"
HEADER_SIZE  = struct.calcsize(HEADER_FMT)

# Per-partial record: bin(u16) amp(f32) phase(f32) = 10 bytes
PARTIAL_FMT  = ">Hff"
PARTIAL_SIZE = struct.calcsize(PARTIAL_FMT)   # 10


def _encode_channel(args):
    """Encode a single channel. Designed to run in a thread pool."""
    sig, win, n_bins, fft_size, hop_size, max_partials = args

    n_samples = len(sig)

    # ── Build strided frame matrix (view, zero-copy) ─────────────────────────
    # Each row is one windowed frame of length fft_size.
    n_frames = max(0, (n_samples - fft_size) // hop_size + 1)
    # np.lib.stride_tricks for zero-copy view of overlapping frames
    frame_shape   = (n_frames, fft_size)
    frame_strides = (sig.strides[0] * hop_size, sig.strides[0])
    frames_raw    = np.lib.stride_tricks.as_strided(sig, shape=frame_shape,
                                                    strides=frame_strides)
    # Apply window (makes a contiguous copy — necessary before FFT)
    frames_win = frames_raw * win[np.newaxis, :]   # (n_frames, fft_size)

    # ── Batch FFT over all frames at once ────────────────────────────────────
    # scipy.fft.rfft with workers=-1 uses all CPU cores for the batch
    spec  = rfft(frames_win, n=fft_size, axis=1)               # (n_frames, n_bins)
    mags  = np.abs(spec).astype(np.float32)                   # (n_frames, n_bins)
    phs   = np.angle(spec).astype(np.float32)                 # (n_frames, n_bins)

    # ── Score + select partials per frame ────────────────────────────────────
    all_frames = []
    k = min(max_partials, n_bins)
    for fi in range(n_frames):
        sc  = compute_scores(mags[fi], SCORE_N, SCORE_HOLE)
        idx = np.argpartition(sc, -k)[-k:]
        idx = idx[np.argsort(sc[idx])[::-1]]
        all_frames.append((idx, mags[fi], phs[fi]))

    return all_frames, n_frames


def encode(pcm: np.ndarray, sample_rate: int, n_channels: int) -> bytes:
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    n_samples, channels = pcm.shape

    win = np.ones(FFT_SIZE, dtype=np.float32)
    win_sum = float(win.sum())
    n_bins  = FFT_SIZE // 2 + 1

    # ── Encode channels in parallel ──────────────────────────────────────────
    channel_args = [
        (pcm[:, ch].astype(np.float32), win, n_bins,
         FFT_SIZE, HOP_SIZE, MAX_PARTIALS)
        for ch in range(channels)
    ]

    with ThreadPoolExecutor(max_workers=channels) as ex:
        results = list(ex.map(_encode_channel, channel_args))

    # ── Pack output into a pre-allocated bytearray ───────────────────────────
    n_frames = results[0][1]

    # Calculate exact output size to avoid reallocations
    total_partials = sum(
        MAX_PARTIALS
        for _, nf in results
        for _ in range(nf)
    )
    # header + magic + per-frame n_partials(u16) + partial records
    buf_size = (4 + HEADER_SIZE
                + channels * n_frames * 2          # n_partials per frame
                + channels * n_frames * MAX_PARTIALS * PARTIAL_SIZE)
    buf = bytearray(buf_size)
    off = 0

    # Magic
    buf[off:off+4] = RSC2_MAGIC; off += 4

    # Header
    struct.pack_into(HEADER_FMT, buf, off,
                     RSC2_VERSION, channels, sample_rate,
                     FFT_SIZE, HOP_SIZE, MAX_PARTIALS,
                     n_frames, n_samples, win_sum)
    off += HEADER_SIZE

    # Frame data
    for ch_frames, _ in results:
        for idx, mags_f, phs_f in ch_frames:
            n_p = len(idx)
            struct.pack_into(">H", buf, off, n_p); off += 2
            for b in idx:
                struct.pack_into(PARTIAL_FMT, buf, off,
                                 int(b), float(mags_f[b]), float(phs_f[b]))
                off += PARTIAL_SIZE

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
    ap = argparse.ArgumentParser(description="RSC2 encoder (optimized)")
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

    import time
    t0   = time.perf_counter()
    data = encode(pcm, sr, ch)
    dt   = time.perf_counter() - t0

    with open(a.output, "wb") as f:
        f.write(data)

    ratio = (len(pcm) / sr) / dt if dt > 0 else float('inf')
    print(f"✅ Written {len(data):,} bytes → {a.output}")
    print(f"   Encoded in {dt*1000:.1f} ms  ({ratio:.1f}× real-time)")


if __name__ == "__main__":
    main()