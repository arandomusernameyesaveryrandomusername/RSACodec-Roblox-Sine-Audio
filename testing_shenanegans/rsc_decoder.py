"""
rsc_encoder.py — RSC6 Partial Encoder with DPSS + Scored Bin Selection

Pipeline:
  1. Load audio (WAV/FLAC/etc. via soundfile)
  2. Frame audio with overlap
  3. Per-frame: apply DPSS (NW=3) window, compute FFT magnitudes
  4. Score each bin using the provided local-crest + curvature formula
  5. Stochastically sample `n_partials` bins weighted by score
  6. Encode to RSC6 bitstream (Rice coding, born/alive masks, etc.)

Usage:
    python rsc_encoder.py --input audio.wav --output audio.rsc --partials 64
    python rsc_encoder.py --input audio.wav --output audio.rsc --partials 128 --frame-size 1024 --seed 42
"""

from __future__ import annotations

import argparse
import math
import struct
import wave
import io
from typing import Optional

import numpy as np
from scipy.signal import windows

# ─────────────────────────────────────────────────────────────────────────────
#  DPSS window
# ─────────────────────────────────────────────────────────────────────────────

def make_dpss_window(N: int, NW: float = 3.0) -> np.ndarray:
    """Return the first (most concentrated) DPSS taper, length N."""
    tapers, _ = windows.dpss(N, NW, Kmax=1, return_ratios=True)
    w = np.asarray(tapers[0], dtype=np.float32)
    # Normalise so the window has unit energy per sample
    w /= np.sqrt(np.sum(w ** 2))
    return w


# ─────────────────────────────────────────────────────────────────────────────
#  Bin scoring  (exact formula from the spec)
# ─────────────────────────────────────────────────────────────────────────────

def score_bins(mags: np.ndarray) -> np.ndarray:
    """
    Score every FFT bin in `mags` (1-D float32, length n_bins).

    Returns score[b] = mags[b] * (local_crest + curvature)  for each b.
    Uses:
      • local_crest  — log(peak) - log(local_mean)  with a ±3-bin hole
      • curvature    — log(peak) - 0.5*(log(prev)+log(next)), clamped ≥ 0
    """
    n_bins     = len(mags)
    score      = np.zeros(n_bins, dtype=np.float32)
    frame_max  = mags.max() + np.float32(1e-12)
    N          = 8        # half-window radius for local background
    hole_radius = 3

    for b in range(n_bins):
        b_lo = max(0, b - N)
        b_hi = min(n_bins - 1, b + N)

        local_sum = np.float32(0.0)
        for k in range(b_lo, b_hi + 1):
            if k < (b - hole_radius) or k > (b + hole_radius):
                local_sum += mags[k]

        actual_hole_size = (
            min(b_hi, b + hole_radius) - max(b_lo, b - hole_radius) + 1
        )
        denom      = np.float32(b_hi - b_lo + 1 - actual_hole_size) + np.float32(1e-12)
        local_mean = local_sum / denom

        log_peak  = math.log(float(mags[b])  + 1e-12)
        log_floor = math.log(float(local_mean) + 1e-12)
        local_crest = max(0.0, log_peak - log_floor)

        if 0 < b < n_bins - 1:
            log_prev  = math.log(float(mags[b - 1]) + 1e-12)
            log_next  = math.log(float(mags[b + 1]) + 1e-12)
            curvature = log_peak - (log_prev + log_next) * 0.5
            t5 = max(curvature, 0.0)
        else:
            t5 = 0.0

        score[b] = mags[b] * np.float32(local_crest + t5)

    return score


# ─────────────────────────────────────────────────────────────────────────────
#  Stochastic partial selection
# ─────────────────────────────────────────────────────────────────────────────

def select_partials(
    freqs_hz: np.ndarray,     # bin-centre frequencies (Hz), length n_bins
    mags:     np.ndarray,     # FFT magnitudes, length n_bins
    score:    np.ndarray,     # per-bin score, length n_bins
    n:        int,            # how many partials to keep
    rng:      np.random.Generator,
    sample_rate: int,
    nyquist_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly sample `n` bins weighted by `score`.
    Returns (selected_freqs_hz, selected_mags), each length n.
    """
    total = float(score.sum())
    if total < 1e-30:
        # Degenerate silent frame — return zeros
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

    probs   = score / total
    indices = rng.choice(len(score), size=min(n, len(score)),
                         replace=False, p=probs)
    # Sort by frequency for deterministic ordering
    indices = np.sort(indices)

    sel_f = freqs_hz[indices].astype(np.float32)
    sel_a = mags[indices].astype(np.float32)

    # Pad to n if fewer unique bins than requested
    if len(sel_f) < n:
        pad = n - len(sel_f)
        sel_f = np.concatenate([sel_f, np.zeros(pad, dtype=np.float32)])
        sel_a = np.concatenate([sel_a, np.zeros(pad, dtype=np.float32)])

    return sel_f, sel_a


# ─────────────────────────────────────────────────────────────────────────────
#  RSC6 bitstream writer
# ─────────────────────────────────────────────────────────────────────────────

# ── Rice zigzag helpers ───────────────────────────────────────────────────────

def zigzag_encode(v: int) -> int:
    return ((-v - 1) << 1) | 1 if v < 0 else v << 1


def rice_encode_word(u: int, k: int) -> list[int]:
    """Return a list of bits (MSB first) for Rice(k) encoding of `u`."""
    q = u >> k
    r = u & ((1 << k) - 1)
    bits = [0] * q + [1]        # unary quotient, terminated by 1
    for i in range(k - 1, -1, -1):
        bits.append((r >> i) & 1)
    return bits


class BitWriter:
    """Accumulate bits MSB-first into bytes."""

    def __init__(self):
        self._buf      = bytearray()
        self._current  = 0
        self._bits_left = 8   # bits remaining in current byte

    def write_bit(self, b: int):
        self._bits_left -= 1
        self._current |= (b & 1) << self._bits_left
        if self._bits_left == 0:
            self._buf.append(self._current)
            self._current  = 0
            self._bits_left = 8

    def write_bits(self, bits):
        for b in bits:
            self.write_bit(b)

    def flush(self) -> bytes:
        if self._bits_left < 8:
            self._buf.append(self._current)
            self._current   = 0
            self._bits_left = 8
        return bytes(self._buf)


# ── Quantisation ─────────────────────────────────────────────────────────────

def freq_to_u16(f_hz: float, sample_rate: int) -> int:
    nyquist = sample_rate / 2.0
    v = int(round(f_hz / nyquist * 65535.0))
    return max(0, min(65535, v))


def amp_to_u16(a: float, a_max: float) -> int:
    """Logarithmic amplitude quantisation matching the decoder's inverse."""
    if a_max < 1e-12 or a < 1e-12:
        return 0
    x = a / a_max                     # linear [0,1]
    mu = 65535.0
    # log-compressed: inverse of decoder's  (10^(x*log10(mu+1)) - 1) / mu
    # Encoder: x_comp = log10(mu*x + 1) / log10(mu + 1)
    x_comp = math.log10(mu * x + 1.0) / math.log10(mu + 1.0)
    v = int(round(x_comp * 65535.0))
    return max(0, min(65535, v))


# ── Main RSC6 writer ──────────────────────────────────────────────────────────

def encode_rsc6(
    all_freqs_u16: np.ndarray,   # int32 (n_frames, n_partials)
    all_amps_u16:  np.ndarray,   # int32 (n_frames, n_partials)
    sample_rate:   int,
    frame_size:    int,
    total_samples: int,
    k_freq:        int = 3,
    k_amp:         int = 3,
) -> bytes:
    n_frames, n_partials = all_freqs_u16.shape
    mask_bytes = math.ceil(n_partials / 8)   # bits per mask row → bytes

    # ── Determine alive / born masks ─────────────────────────────────────────
    alive = np.zeros((n_frames, n_partials), dtype=bool)
    born  = np.zeros((n_frames, n_partials), dtype=bool)

    for i in range(n_frames):
        for s in range(n_partials):
            a  = all_amps_u16[i, s] > 0
            pa = all_amps_u16[i - 1, s] > 0 if i > 0 else False
            alive[i, s] = a
            born[i, s]  = a and not pa

    # ── Bitmask section ───────────────────────────────────────────────────────
    bitmask_buf = bytearray()
    for i in range(n_frames):
        # Alive mask
        row = np.zeros(mask_bytes * 8, dtype=np.uint8)
        row[:n_partials] = alive[i].astype(np.uint8)
        packed = np.packbits(row.reshape(mask_bytes, 8)[:, ::-1].ravel())
        bitmask_buf.extend(packed[:mask_bytes])
        # Born mask
        row[:n_partials] = born[i].astype(np.uint8)
        packed = np.packbits(row.reshape(mask_bytes, 8)[:, ::-1].ravel())
        bitmask_buf.extend(packed[:mask_bytes])

    # ── Born-partial table ────────────────────────────────────────────────────
    born_buf = bytearray()
    for i in range(n_frames):
        for s in range(n_partials):
            if alive[i, s] and born[i, s]:
                born_buf.extend(struct.pack("<HH",
                    int(all_freqs_u16[i, s]),
                    int(all_amps_u16[i, s])))

    # ── Rice-coded frequency deltas ───────────────────────────────────────────
    freq_writer = BitWriter()
    prev_fq = np.zeros(n_partials, dtype=np.int32)

    for i in range(n_frames):
        for s in range(n_partials):
            if not alive[i, s] or born[i, s]:
                if not alive[i, s]:
                    prev_fq[s] = 0
                continue
            curr = int(all_freqs_u16[i, s])
            delta = curr - int(prev_fq[s])
            freq_writer.write_bits(rice_encode_word(zigzag_encode(delta), k_freq))
            prev_fq[s] = curr

    freq_bytes = freq_writer.flush()

    # ── Rice-coded amplitude deltas ───────────────────────────────────────────
    amp_writer = BitWriter()
    prev_amu = np.zeros(n_partials, dtype=np.int32)

    for i in range(n_frames):
        for s in range(n_partials):
            if not alive[i, s] or born[i, s]:
                if not alive[i, s]:
                    prev_amu[s] = 0
                continue
            curr = int(all_amps_u16[i, s])
            delta = curr - int(prev_amu[s])
            amp_writer.write_bits(rice_encode_word(zigzag_encode(delta), k_amp))
            prev_amu[s] = curr

    amp_bytes = amp_writer.flush()

    # ── RSC6 header ───────────────────────────────────────────────────────────
    # "<4sBIIHIIHBBII"
    #   magic(4) version(1) sample_rate(4) frame_size(4) n_partials(2)
    #   total_samples(4) total_frames(4) mask_sz(2) k_freq(1) k_amp(1)
    #   born_data_sz(4) rice_freq_sz(4)
    header = struct.pack(
        "<4sBIIHIIHBBII",
        b"RSC6",
        6,                          # version
        sample_rate,
        frame_size,
        n_partials,
        total_samples,
        n_frames,
        mask_bytes,
        k_freq,
        k_amp,
        len(born_buf),
        len(freq_bytes),
    )

    return header + bytes(bitmask_buf) + bytes(born_buf) + freq_bytes + amp_bytes


# ─────────────────────────────────────────────────────────────────────────────
#  Audio loader  (WAV via stdlib, others via soundfile if present)
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Return (samples float32 mono, sample_rate)."""
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        return data, sr
    except ImportError:
        pass

    # Fallback: stdlib wave (PCM only)
    with wave.open(path, "rb") as wf:
        sr        = wf.getframerate()
        n_ch      = wf.getnchannels()
        sw        = wf.getsampwidth()
        n_frames  = wf.getnframes()
        raw       = wf.readframes(n_frames)

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dt    = dtype_map.get(sw, np.int16)
    data  = np.frombuffer(raw, dtype=dt).astype(np.float32)
    data /= float(np.iinfo(dt).max)
    if n_ch > 1:
        data = data.reshape(-1, n_ch).mean(axis=1)
    return data, sr


# ─────────────────────────────────────────────────────────────────────────────
#  Main encode pipeline
# ─────────────────────────────────────────────────────────────────────────────

def encode(
    input_path:  str,
    output_path: str,
    n_partials:  int,
    frame_size:  int,
    seed:        Optional[int],
    k_freq:      int,
    k_amp:       int,
    hop_ratio:   float,
) -> None:
    import time
    print(f"🎙️  RSC6 Encoder  —  {input_path}")
    print(f"   partials={n_partials}  frame={frame_size}  k_freq={k_freq}  k_amp={k_amp}  seed={seed}")

    t0 = time.perf_counter()

    # ── Load audio ────────────────────────────────────────────────────────────
    samples, sr = load_audio(input_path)
    total_samples = len(samples)
    print(f"   {total_samples:,} samples  |  {sr} Hz  |  {total_samples/sr:.2f}s")

    # ── DPSS window + FFT setup ───────────────────────────────────────────────
    hop        = max(1, int(frame_size * hop_ratio))
    win        = make_dpss_window(frame_size, NW=3.0)
    n_bins     = frame_size // 2 + 1
    freqs_hz   = np.linspace(0.0, sr / 2.0, n_bins, dtype=np.float32)

    # Pad so we cover total_samples
    n_frames = max(1, math.ceil((total_samples - frame_size) / hop) + 1)
    pad_len  = (n_frames - 1) * hop + frame_size - total_samples
    if pad_len > 0:
        samples = np.concatenate([samples, np.zeros(pad_len, dtype=np.float32)])

    rng = np.random.default_rng(seed)

    all_freqs = np.zeros((n_frames, n_partials), dtype=np.int32)
    all_amps  = np.zeros((n_frames, n_partials), dtype=np.int32)

    # Global amplitude reference (max across all frames)
    global_max = np.float32(0.0)

    print(f"   🔍 Analysing {n_frames:,} frames with DPSS (NW=3)…")

    # Two-pass: first collect all magnitudes to find global max
    frame_mags = []
    for i in range(n_frames):
        start  = i * hop
        chunk  = samples[start : start + frame_size].astype(np.float32)
        if len(chunk) < frame_size:
            chunk = np.pad(chunk, (0, frame_size - len(chunk)))
        windowed  = chunk * win
        spectrum  = np.fft.rfft(windowed, n=frame_size)
        mags      = np.abs(spectrum).astype(np.float32)
        frame_mags.append(mags)
        pk = mags.max()
        if pk > global_max:
            global_max = pk

    print(f"   🎲 Scoring bins & selecting {n_partials} partials per frame…")

    for i, mags in enumerate(frame_mags):
        if i % 500 == 0:
            print(f"      frame {i:,}/{n_frames:,}", end="\r", flush=True)

        score = score_bins(mags)

        # Stochastic selection weighted by score
        sel_f, sel_a = select_partials(
            freqs_hz, mags, score, n_partials, rng, sr, n_bins
        )

        # Quantise
        for s in range(n_partials):
            all_freqs[i, s] = freq_to_u16(float(sel_f[s]), sr)
            all_amps[i, s]  = amp_to_u16(float(sel_a[s]), float(global_max))

    print(f"\n   ✅ Analysis complete")

    # ── Encode to RSC6 ────────────────────────────────────────────────────────
    print("   📦 Encoding RSC6 bitstream…")
    bitstream = encode_rsc6(
        all_freqs, all_amps,
        sample_rate=sr,
        frame_size=frame_size,
        total_samples=total_samples,
        k_freq=k_freq,
        k_amp=k_amp,
    )

    with open(output_path, "wb") as f:
        f.write(bitstream)

    elapsed = time.perf_counter() - t0
    dur = total_samples / sr
    sz_kb = len(bitstream) / 1024
    print(f"   🎉 Wrote {sz_kb:.1f} KB → {output_path}")
    print(f"   ⏱️  Done in {elapsed:.2f}s  (RTF {elapsed/dur:.3f}×)")
    print(f"   📊 Compression: {sz_kb*1024/(total_samples*2)*100:.1f}% of 16-bit PCM size")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="RSC6 Encoder — DPSS FFT + scored bin selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",      "-i", required=True,       help="Input audio file (WAV, FLAC, etc.)")
    p.add_argument("--output",     "-o", default=None,        help="Output .rsc file (default: <input>.rsc)")
    p.add_argument("--partials",   "-p", type=int, default=64,
                   help="Number of sinusoidal partials per frame (bins to keep)")
    p.add_argument("--frame-size", "-f", type=int, default=1024,
                   help="FFT/frame size in samples")
    p.add_argument("--hop",              type=float, default=0.5,
                   help="Hop ratio (fraction of frame_size), e.g. 0.5 = 50%% overlap")
    p.add_argument("--k-freq",           type=int, default=3,  help="Rice k for frequency deltas")
    p.add_argument("--k-amp",            type=int, default=3,  help="Rice k for amplitude deltas")
    p.add_argument("--seed",             type=int, default=None,
                   help="RNG seed for reproducible partial selection (default: random)")
    args = p.parse_args()

    if args.partials < 1:
        p.error("--partials must be ≥ 1")
    if args.frame_size < 64 or (args.frame_size & (args.frame_size - 1)):
        p.error("--frame-size must be a power of 2 and ≥ 64")

    out = args.output or (args.input.rsplit(".", 1)[0] + ".rsc")
    encode(
        input_path=args.input,
        output_path=out,
        n_partials=args.partials,
        frame_size=args.frame_size,
        seed=args.seed,
        k_freq=args.k_freq,
        k_amp=args.k_amp,
        hop_ratio=args.hop,
    )


if __name__ == "__main__":
    main()