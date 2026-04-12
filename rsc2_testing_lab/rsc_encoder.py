#!/usr/bin/env python3
"""
RSC2 Audio Codec - Encoder  (u16 amp+phase edition)
=====================================================
Identical scoring / partial-selection logic as before.
Amp and phase are now stored as uint16 instead of float32:
  amp   u16  : linear  0-65535  where 65535 = peak magnitude
  phase u16  : 0-65535 mapping 0 → 0, 65535 → 2π  (kept for completeness;
               the decoder ignores phase for additive sine synthesis)

Per-partial record is now 6 bytes (was 10):  bin(u16) amp(u16) phase(u16)

Header is unchanged (still big-endian):
  Magic      4 bytes  "RSC2"
  version    u8
  channels   u8
  sampleRate u32be
  fftSize    u16be
  hopSize    u16be
  maxPartials u16be
  nFrames    u32be
  nSamples   u32be
  winSum     f32be   (kept so old tooling can still read the header)
"""
import numpy as np
import struct
import wave
import argparse
from concurrent.futures import ThreadPoolExecutor
from numpy.fft import rfft

# ── Try Numba JIT ─────────────────────────────────────────────────────────────
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
    n      = len(mags)
    b_idx  = np.arange(n)
    pad    = score_n
    mpad   = np.pad(mags.astype(np.float64), pad, mode='constant')
    cs     = np.concatenate(([0.0], np.cumsum(mpad)))
    b_lo   = np.maximum(0,     b_idx - score_n)
    b_hi   = np.minimum(n - 1, b_idx + score_n)
    win_sum = cs[b_hi + pad + 1] - cs[b_lo + pad]
    win_cnt = b_hi - b_lo + 1
    h_lo   = np.maximum(b_lo, b_idx - score_hole)
    h_hi   = np.minimum(b_hi, b_idx + score_hole)
    has_hole = h_lo <= h_hi
    hole_sum = np.where(has_hole, cs[h_hi + pad + 1] - cs[h_lo + pad], 0.0)
    hole_cnt = np.where(has_hole, h_hi - h_lo + 1, 0)
    lmean    = (win_sum - hole_sum) / np.maximum(win_cnt - hole_cnt, 1).astype(np.float64)
    log_peak  = np.log(mags.astype(np.float64) + 1e-12)
    log_floor = np.log(lmean + 1e-12)
    t3 = np.maximum(0.0, log_peak - log_floor)
    t5 = np.zeros(n, dtype=np.float64)
    if n > 2:
        inner = slice(1, n - 1)
        t5[inner] = np.maximum(0.0, log_peak[inner] - 0.5 * (log_peak[:-2] + log_peak[2:]))
    return ((t3 + t5) * mags).astype(np.float32)


def compute_scores(mags, score_n=8, score_hole=3):
    if _USE_NUMBA:
        return _score_numba(mags, score_n, score_hole)
    return _score_numpy(mags, score_n, score_hole)


# ── Constants ─────────────────────────────────────────────────────────────────
RSC2_MAGIC    = b"RSC2"
RSC2_VERSION  = 1
FFT_SIZE      = 1024
HOP_SIZE      = 1024
MAX_PARTIALS  = 192
SCORE_N       = 6
SCORE_HOLE    = 2

# Header: version(B) channels(B) sampleRate(I) fftSize(H) hopSize(H)
#         maxPartials(H) nFrames(I) nSamples(I) winSum(f)
HEADER_FMT  = ">BBIHHHIIf"
HEADER_SIZE = struct.calcsize(HEADER_FMT)   # 23 bytes

# Per-partial: bin(u16) amp(u16) phase(u16) = 6 bytes  (was 10 with two f32s)
PARTIAL_FMT  = ">HHH"
PARTIAL_SIZE = struct.calcsize(PARTIAL_FMT)  # 6

AMP_MAX   = 65535.0   # u16 full scale for amplitude
PHASE_MAX = 65535.0   # u16 full scale for phase (0 → 0, 65535 → 2π)
TWO_PI    = 2.0 * np.pi


def _encode_channel(args):
    sig, win, n_bins, fft_size, hop_size, max_partials = args
    n_samples = len(sig)
    n_frames  = max(0, (n_samples - fft_size) // hop_size + 1)

    frame_shape   = (n_frames, fft_size)
    frame_strides = (sig.strides[0] * hop_size, sig.strides[0])
    frames_raw    = np.lib.stride_tricks.as_strided(sig, shape=frame_shape, strides=frame_strides)
    frames_win    = frames_raw * win[np.newaxis, :]

    spec = rfft(frames_win, n=fft_size, axis=1)              # (n_frames, n_bins)
    mags = np.abs(spec).astype(np.float32)                   # (n_frames, n_bins)
    phs  = np.angle(spec).astype(np.float32)                 # (n_frames, n_bins) in [-π, π]

    # Normalise magnitudes per-frame so amp u16 is relative (0=silent, 65535=peak)
    frame_max = mags.max(axis=1, keepdims=True)
    frame_max = np.where(frame_max < 1e-9, 1.0, frame_max)  # avoid /0
    mags_norm = mags / frame_max                             # 0-1 float

    all_frames = []
    k = min(max_partials, n_bins)
    for fi in range(n_frames):
        sc  = compute_scores(mags[fi], SCORE_N, SCORE_HOLE)
        idx = np.argpartition(sc, -k)[-k:]
        idx = idx[np.argsort(sc[idx])[::-1]]

        partials = []
        for b in idx:
            amp_u16   = int(round(float(mags_norm[fi, b]) * AMP_MAX))
            amp_u16   = max(0, min(65535, amp_u16))
            # phase: map [-π, π] → [0, 65535]
            phase_u16 = int(round((float(phs[fi, b]) + np.pi) / TWO_PI * PHASE_MAX))
            phase_u16 = max(0, min(65535, phase_u16))
            partials.append((int(b), amp_u16, phase_u16))

        all_frames.append(partials)

    return all_frames, n_frames


def encode(pcm: np.ndarray, sample_rate: int, n_channels: int) -> bytes:
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    n_samples, channels = pcm.shape

    win     = np.ones(FFT_SIZE, dtype=np.float32)
    win_sum = float(win.sum())
    n_bins  = FFT_SIZE // 2 + 1

    channel_args = [
        (pcm[:, ch].astype(np.float32), win, n_bins,
         FFT_SIZE, HOP_SIZE, MAX_PARTIALS)
        for ch in range(channels)
    ]

    with ThreadPoolExecutor(max_workers=channels) as ex:
        results = list(ex.map(_encode_channel, channel_args))

    n_frames = results[0][1]

    # Pre-calculate exact buffer size:
    #   4 (magic) + HEADER_SIZE
    #   + per channel * per frame: 2 (n_partials u16) + MAX_PARTIALS * PARTIAL_SIZE
    buf_size = (4 + HEADER_SIZE
                + channels * n_frames * 2
                + channels * n_frames * MAX_PARTIALS * PARTIAL_SIZE)
    buf = bytearray(buf_size)
    off = 0

    buf[off:off+4] = RSC2_MAGIC; off += 4

    struct.pack_into(HEADER_FMT, buf, off,
                     RSC2_VERSION, channels, sample_rate,
                     FFT_SIZE, HOP_SIZE, MAX_PARTIALS,
                     n_frames, n_samples, win_sum)
    off += HEADER_SIZE

    for ch_frames, _ in results:
        for partials in ch_frames:
            n_p = len(partials)
            struct.pack_into(">H", buf, off, n_p); off += 2
            for b, amp_u16, phase_u16 in partials:
                struct.pack_into(PARTIAL_FMT, buf, off, b, amp_u16, phase_u16)
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
    global MAX_PARTIALS, FFT_SIZE, HOP_SIZE
    ap = argparse.ArgumentParser(description="RSC2 encoder (u16 amp+phase)")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--no-numba", action="store_true")
    ap.add_argument("--max-partials", type=int, default=MAX_PARTIALS,
                    help=f"Partials per frame per channel (default {MAX_PARTIALS})")
    ap.add_argument("--fft-size", type=int, default=FFT_SIZE,
                    help=f"FFT window size (default {FFT_SIZE})")
    ap.add_argument("--hop-size", type=int, default=HOP_SIZE,
                    help=f"Hop size in samples (default {HOP_SIZE})")
    a = ap.parse_args()

    if a.no_numba:
        global _USE_NUMBA
        _USE_NUMBA = False

    # Allow overriding globals via CLI
    
    MAX_PARTIALS = a.max_partials
    FFT_SIZE     = a.fft_size
    HOP_SIZE     = a.hop_size

    print(f"🎵 Loading {a.input}…")
    pcm, sr, ch = load_wav(a.input)
    print(f"   {sr} Hz | {ch} ch | {len(pcm)} samples")
    print(f"⚙️  Encoding (FFT={FFT_SIZE}, hop={HOP_SIZE}, partials={MAX_PARTIALS})…")
    print(f"   Partial record: bin(u16) + amp(u16) + phase(u16) = 6 bytes")

    import time
    t0   = time.perf_counter()
    data = encode(pcm, sr, ch)
    dt   = time.perf_counter() - t0

    with open(a.output, "wb") as f:
        f.write(data)

    ratio = (len(pcm) / sr) / dt if dt > 0 else float('inf')
    orig  = len(pcm) * ch * 2   # approximate PCM16 size for comparison
    print(f"✅ Written {len(data):,} bytes → {a.output}  "
          f"({len(data)/orig*100:.1f}% of PCM16)")
    print(f"   Encoded in {dt*1000:.1f} ms  ({ratio:.1f}× real-time)")


if __name__ == "__main__":
    main()