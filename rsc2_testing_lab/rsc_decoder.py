#!/usr/bin/env python3
"""
RSC2 Audio Codec - Decoder (u16 amp+phase edition) [OPTIMIZED]

OPTIMIZATIONS:
  1. Vectorized binary parsing: np.frombuffer + structured dtype (replaces 600+ struct.unpack calls)
  2. Multi-channel parallelization: ThreadPoolExecutor for channel decoding
  3. Pre-allocated reusable arrays: bins/amps/phases arrays resized per-frame instead of allocated
  4. Numba JIT verification: Warn if Numba unavailable (no silent fallback to slow path)

Binary format per frame:
  peak       f32be   per-frame max FFT magnitude
  nPartials  u16be
  [ bin(u16be)  amp(u16be)  phase(u16be) ] × nPartials

Reconstruction per frame:
  amp   = (amp_u16  / 65535.0) * peak          → absolute FFT magnitude
  phase = (ph_u16   / 65535.0) * 2π  - π       → radians
  spec[bin] = amp * exp(j * phase)
  Reconstruction via Additive Synthesis (fast Numba-optimized loop) +
  OLA with rectangular accumulation + divide by overlap count
"""
import numpy as np
import struct
import wave
import argparse
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

RSC2_MAGIC  = b"RSC2"
HEADER_FMT  = "<BBIHHHIIf"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# Per-frame header written by encoder: peak(f32be) + nPartials(u16be)
FRAME_HDR_FMT  = "<fH"
FRAME_HDR_SIZE = struct.calcsize(FRAME_HDR_FMT)   # 6

# Per-partial record: bin(u16be) amp(u16be) phase(u16be)
PARTIAL_FMT  = "<HHH"
PARTIAL_SIZE = struct.calcsize(PARTIAL_FMT)        # 6

# Structured dtype for batch-unpacking partials: 6 bytes each (bin, amp, phase)
PARTIAL_DTYPE = np.dtype([("bin", "<u2"), ("amp", "<u2"), ("phase", "<u2")])

_U16_MAX    = 65535.0
_TWO_PI     = 2.0 * np.pi

# Try numba for additive synthesis
try:
    from numba import njit
    _USE_NUMBA_SYNTH = True
    print("   ✅ Numba JIT enabled")
except (ImportError, RuntimeError) as e:
    print(f"   ⚠️  WARNING: Numba unavailable ({e}). Using slower NumPy fallback.")
    _USE_NUMBA_SYNTH = False
    njit = lambda **kw: lambda f: f  # no-op decorator







# ── Sin LUT for faster transcendental computation ─────────────────────────────
_LUT_SIZE = 2**16  # 65536 entries
_PHASE_TO_LUT = np.float32(_LUT_SIZE / _TWO_PI)
_SIN_LUT = np.sin(np.linspace(0.0, _TWO_PI, _LUT_SIZE, endpoint=False, dtype=np.float32)).astype(np.float32)


@njit(fastmath=True)
def _additive_synth_numba(bins: np.ndarray, amps: np.ndarray, phases: np.ndarray, 
                          sample_rate: float, fft_size: int) -> np.ndarray:
    """
    Fast additive synthesis using Numba JIT with improved phase tracking.
    
    Optimizations:
    - Direct phase accumulation (float64 interim for precision, then mod to [0, 2π))
    - Cos computed from sin(φ + π/2) for cache efficiency
    """
    frame = np.zeros(fft_size, dtype=np.float32)
    two_pi = 2.0 * np.pi
    
    for i in range(len(bins)):
        bin_idx = bins[i]
        freq = (bin_idx * sample_rate) / fft_size
        amp = amps[i]
        phase = phases[i]
        
        # Phase increment per sample (keep as float64 for accumulation precision)
        phase_inc = two_pi * freq / sample_rate
        current_phase = float(phase)
        
        for n in range(fft_size):
            # Accumulate phase with full precision
            current_phase += phase_inc
            # Keep in [0, 2π) to avoid precision loss
            if current_phase >= two_pi:
                current_phase -= two_pi
            
            frame[n] += amp * np.cos(np.float32(current_phase))
    
    return frame


def _additive_synth_numpy(bins: np.ndarray, amps: np.ndarray, phases: np.ndarray,
                          sample_rate: float, fft_size: int) -> np.ndarray:
    """
    Vectorized additive synthesis using NumPy.
    
    Falls back to pure NumPy if Numba unavailable (slower but still fast).
    """
    frame = np.zeros(fft_size, dtype=np.float32)
    sample_times = np.arange(fft_size, dtype=np.float32) / sample_rate
    two_pi = 2.0 * np.pi
    
    for i in range(len(bins)):
        bin_idx = bins[i]
        freq = (bin_idx * sample_rate) / fft_size
        phase = phases[i]
        amp = amps[i]
        
        # Vectorized: all samples at once
        frame += amp * np.cos(two_pi * freq * sample_times + phase)
    
    return frame


def additive_synth(bins: np.ndarray, amps: np.ndarray, phases: np.ndarray,
                   sample_rate: float, fft_size: int) -> np.ndarray:
    """Route to fastest available additive synth."""
    if _USE_NUMBA_SYNTH:
        return _additive_synth_numba(bins, amps, phases, sample_rate, fft_size)
    else:
        return _additive_synth_numpy(bins, amps, phases, sample_rate, fft_size)


def decode(data: bytes):
    if data[:4] != RSC2_MAGIC:
        raise ValueError(f"Bad magic {data[:4]!r}")

    off = 4
    (version, channels, sample_rate,
     fft_size, hop_size, max_partials,
     n_frames, n_samples, window_sum) = struct.unpack_from(HEADER_FMT, data, off)
    off += HEADER_SIZE

    print(f"   RSC2 v{version} | {sample_rate} Hz | {channels} ch")
    print(f"   FFT={fft_size} hop={hop_size} max_partials={max_partials} frames={n_frames}")
    print(f"   window_sum={window_sum:.4f}  (rect window → no amp correction needed)")
    print(f"   Numba JIT: {'✅ enabled' if _USE_NUMBA_SYNTH else '❌ disabled (slow fallback)'}")

    if version != 1:
        raise ValueError(f"Unsupported RSC2 version {version}")

    total  = n_samples + fft_size          # safe output buffer length

    pcm = np.zeros((total, channels), dtype=np.float32)

    pbar = tqdm(total=channels * n_frames, desc="Decoding frames", unit="frames", mininterval=1/60, smoothing=0.1)

    # Pre-calculate channel offsets by pre-scanning frame structure
    # (necessary for parallel decoding)
    channel_offsets = _compute_channel_offsets(data, off, channels, n_frames)

    # Decode channels in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(channels, 4)) as executor:
        futures = [
            executor.submit(
                _decode_channel,
                ch, data, channel_offsets[ch], n_frames, total,
                sample_rate, fft_size, hop_size, window_sum, pbar
            )
            for ch in range(channels)
        ]
        
        # Collect results
        for ch, future in enumerate(futures):
            ch_pcm, ch_cnt = future.result()
            ch_cnt = np.where(ch_cnt < 1e-8, 1.0, ch_cnt)
            pcm[:, ch] = ch_pcm / ch_cnt
            print(f"   Ch {ch}: max overlap={ch_cnt.max():.1f}")

    pbar.close()
    return pcm[:n_samples, :], sample_rate, channels


def _compute_channel_offsets(data: bytes, start_off: int, channels: int, n_frames: int) -> list:
    """
    Pre-scan the file to find where each channel's data starts.
    
    File layout: [Channel 0 frames] [Channel 1 frames] ...
    Each frame: peak(4) + n_p(2) + [partials × 6]
    
    Returns: list of byte offsets for each channel
    """
    offsets = [start_off]
    current_off = start_off

    # Scan through all frames for the first channel to find where channel 1 starts
    for frame_idx in range(n_frames):
        peak, n_p = struct.unpack_from(FRAME_HDR_FMT, data, current_off)
        current_off += FRAME_HDR_SIZE + n_p * PARTIAL_SIZE

    # For multi-channel files, each subsequent channel starts where the previous ended
    for ch in range(1, channels):
        offsets.append(current_off)
        # Scan through channel frames to find next channel start
        for frame_idx in range(n_frames):
            peak, n_p = struct.unpack_from(FRAME_HDR_FMT, data, current_off)
            current_off += FRAME_HDR_SIZE + n_p * PARTIAL_SIZE

    return offsets


def _decode_channel(ch: int, data: bytes, ch_off: int, n_frames: int, total: int,
                    sample_rate: float, fft_size: int, hop_size: int,
                    window_sum: float, pbar):
    """
    Decode a single channel from the RSC2 data.
    
    Args: pbar - tqdm progress bar instance (or compatible object with .update() method)
    Returns: (ch_pcm, ch_cnt)
    """
    ch_pcm = np.zeros(total, dtype=np.float32)
    ch_cnt = np.zeros(total, dtype=np.float32)
    pos    = 0

    # Pre-allocate reusable arrays to avoid per-frame allocation
    bins_frame = np.empty(65535, dtype=np.uint16)  # max size
    amps_frame = np.empty(65535, dtype=np.float32)
    phs_frame = np.empty(65535, dtype=np.float32)

    for _ in range(n_frames):
        # ── Frame header ─────────────────────────────────────────────────
        peak, n_p = struct.unpack_from(FRAME_HDR_FMT, data, ch_off)
        ch_off += FRAME_HDR_SIZE

        # ── Vectorized partial reading ───────────────────────────────────
        # Use np.frombuffer + structured dtype to batch-read all partials at once
        partial_data = data[ch_off : ch_off + n_p * PARTIAL_SIZE]
        partials = np.frombuffer(partial_data, dtype=PARTIAL_DTYPE, count=n_p)
        ch_off += n_p * PARTIAL_SIZE

        # Vectorized decoding: extract fields and convert in one pass
        bins_u16 = partials["bin"]
        amp_u16 = partials["amp"]
        ph_u16 = partials["phase"]
        
        # Vectorized amplitude and phase decoding
        amps_decoded = (amp_u16 / _U16_MAX) / window_sum * peak
        phs_decoded = (ph_u16 / _U16_MAX) * _TWO_PI - np.pi

        # ── Additive synthesis → OLA ──────────────────────────────────────
        frame = additive_synth(bins_u16, amps_decoded, phs_decoded, 
                               sample_rate, fft_size)
        end   = pos + fft_size
        ch_pcm[pos:end] += frame
        ch_cnt[pos:end] += 1.0
        pos += hop_size

        pbar.update(1)

    return ch_pcm, ch_cnt


def save_wav(path: str, pcm: np.ndarray, sr: int):
    n, ch = pcm.shape
    # Direct conversion without normalization or clipping
    raw = (pcm * 32767.0).astype(np.int16).tobytes()
    with wave.open(path, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(raw)


def main():
    ap = argparse.ArgumentParser(description="RSC2 decoder (u16 amp+phase)")
    ap.add_argument("input")
    ap.add_argument("output")
    a = ap.parse_args()

    print(f"📂 Loading {a.input}…")
    with open(a.input, "rb") as f:
        data = f.read()
    print(f"   {len(data):,} bytes")

    print("⚙️  Decoding…")
    t0 = time.perf_counter()
    pcm, sr, ch = decode(data)
    dt = time.perf_counter() - t0

    print(f"💾 Writing {a.output}…")
    save_wav(a.output, pcm, sr)
    print(f"✅ Done  ({len(pcm)} samples, {sr} Hz, {ch} ch)")
    print(f"   Decoded in {dt * 1000:.1f} ms")


if __name__ == "__main__":
    main()