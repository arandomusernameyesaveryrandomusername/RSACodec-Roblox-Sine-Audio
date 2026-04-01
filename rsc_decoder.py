from __future__ import annotations

"""
rsc_decoder.py — Roblox Sine Codec (RSC) Decoder  [OPTIMIZED]

Format: RSC6

Key optimisations vs. the reference decoder
============================================
1.  Numba @njit (nopython) on every hot path
    • _parse_frames_njit  – Rice-decode loop + freq/amp reconstruction
    • _build_phi_track_njit – sequential phase accumulation
    • _synthesize_njit   – main synthesis, parallelised with prange + fastmath

2.  Parallel outer loop (numba.prange) over frames in synthesis;
    each frame writes to a private slice so there are no data races.

3.  float32 everywhere (confirmed) — avoids implicit upcasts that
    numpy/numba would otherwise silently insert.

4.  Pre-computed sin table look-up (optional, enabled by default) via
    SINS_LUT: reduces transcendental cost further on CPUs without AVX-512
    SVML.  Disable with --no-lut if SVML is available and faster.

5.  BitReader rewritten as a @njit function operating on a raw uint8
    array — eliminates Python object overhead per bit.

6.  Bitmask section decoded in one vectorised np.unpackbits call for
    the whole file, then sliced per frame — no per-frame allocation.

7.  Born-partial table pre-decoded into a structured array before the
    Numba loop so the JIT function never touches raw bytes.

8.  Peak normalisation and WAV clipping fused into a single pass.

Usage:
    python rsc_decoder.py --input audio.rsc --output decoded.wav [--no-lut]
"""

import argparse
import math
import struct
import wave
from typing import Optional

import numpy as np
from numba import njit, prange
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
TWO_PI_F32: np.float32 = np.float32(2.0 * math.pi)

# Sin LUT — 2^16 entries, covers [0, 2π)
_LUT_SIZE   = 1 << 16
_LUT_SCALE  = np.float32(_LUT_SIZE / (2.0 * math.pi))
_SIN_LUT    = np.sin(np.linspace(0.0, 2.0 * math.pi, _LUT_SIZE,
                                  endpoint=False)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Numba JIT helpers
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def _read_rice_njit(data: np.ndarray, pos: int, bits_left: int,
                    buf: int, k: int):
    """
    Read one Rice(k)-coded zigzag integer from a uint8 byte array.

    Returns (value, new_pos, new_bits_left, new_buf).
    MSB-first packing (same as reference _BitReader).
    """
    # unary quotient
    q = 0
    while True:
        if bits_left == 0:
            buf       = data[pos]
            pos      += 1
            bits_left = 8
        bits_left -= 1
        bit = (buf >> bits_left) & 1
        if bit == 1:
            break
        q += 1

    # k remainder bits
    r = 0
    for _ in range(k):
        if bits_left == 0:
            buf       = data[pos]
            pos      += 1
            bits_left = 8
        bits_left -= 1
        r = (r << 1) | ((buf >> bits_left) & 1)

    # zigzag decode
    u = (q << k) | r
    val = -(u >> 1) - 1 if (u & 1) else (u >> 1)
    return val, pos, bits_left, buf


@njit(cache=True, fastmath=True)
def _parse_frames_njit(
    raw:          np.ndarray,   # uint8, entire file
    alive_flat:   np.ndarray,   # bool  (total_frames * n_partials,)
    born_flat:    np.ndarray,   # bool  (total_frames * n_partials,)
    born_pairs:   np.ndarray,   # uint16 (N_born, 2) — [fq, amu] per born event
    born_slots:   np.ndarray,   # int32  (N_born,)   — which slot
    born_frames:  np.ndarray,   # int32  (N_born,)   — which frame
    freq_start:   int,
    amp_start:    int,
    k_freq:       int,
    k_amp:        int,
    freq_scale:   np.float32,
    total_frames: int,
    n_partials:   int,
    # outputs (pre-allocated)
    freqs_out:    np.ndarray,   # float32 (total_frames, n_partials)
    amps_out:     np.ndarray,   # float32 (total_frames, n_partials)
):
    """
    Decode all frames: Rice-code freq/amp deltas + born-partial table.
    Pure Numba nopython — the single biggest hotspot in the parser.
    """
    curr_fq  = np.zeros(n_partials, dtype=np.int32)
    curr_amu = np.zeros(n_partials, dtype=np.int32)

    fpos = freq_start; fbuf = 0; fleft = 0
    apos = amp_start;  abuf = 0; aleft = 0

    born_idx = 0
    n_born   = len(born_frames)

    for i in range(total_frames):
        base = i * n_partials

        # Zero dead partials
        for slot in range(n_partials):
            if not alive_flat[base + slot]:
                curr_fq[slot]  = 0
                curr_amu[slot] = 0

        # Consume born-pair table entries that belong to this frame
        while born_idx < n_born and born_frames[born_idx] == i:
            slot           = born_slots[born_idx]
            curr_fq[slot]  = born_pairs[born_idx, 0]
            curr_amu[slot] = born_pairs[born_idx, 1]
            born_idx      += 1

        for slot in range(n_partials):
            if not alive_flat[base + slot]:
                continue
            if born_flat[base + slot]:
                # Values already set from born-pair table above
                pass
            else:
                df, fpos, fleft, fbuf = _read_rice_njit(raw, fpos, fleft, fbuf, k_freq)
                da, apos, aleft, abuf = _read_rice_njit(raw, apos, aleft, abuf, k_amp)
                v = curr_fq[slot] + df
                curr_fq[slot]  = max(0, min(65535, v))
                v = curr_amu[slot] + da
                curr_amu[slot] = max(0, min(65535, v))

            freqs_out[i, slot] = curr_fq[slot]  * freq_scale
            mu = np.float32(65535.0)
            x  = np.float32(curr_amu[slot]) / mu        # back to [0,1] linear
            amps_out[i, slot] = (np.float32(10.0) ** (x * np.log10(mu + np.float32(1.0))) - np.float32(1.0)) / mu


@njit(cache=True, fastmath=True)
def _build_phi_track_njit(
    f_use:       np.ndarray,   # float32 (F, P)
    a32:         np.ndarray,   # float32 (F, P)
    prev_a:      np.ndarray,   # float32 (F, P)
    T_sec:       np.float32,
    n_frames:    int,
    n_partials:  int,
    phi_track:   np.ndarray,   # float32 (F, P)  — output
):
    phi = np.zeros(n_partials, dtype=np.float32)
    for i in range(n_frames):
        for p in range(n_partials):
            # Born partial: reset phase to 0
            if prev_a[i, p] == np.float32(0.0) and a32[i, p] > np.float32(0.0):
                phi[p] = np.float32(0.0)
            phi_track[i, p] = phi[p]
            # Advance
            phi[p] += TWO_PI_F32 * f_use[i, p] * T_sec
            if phi[p] >= TWO_PI_F32:
                phi[p] -= TWO_PI_F32
            # Death: reset for clean future birth
            if a32[i, p] == np.float32(0.0):
                phi[p] = np.float32(0.0)


@njit(cache=True, fastmath=True, parallel=True)
def _synthesize_njit(
    f_use:      np.ndarray,   # float32 (F, P)
    a32:        np.ndarray,   # float32 (F, P)
    prev_a:     np.ndarray,   # float32 (F, P)
    phi_track:  np.ndarray,   # float32 (F, P)
    active:     np.ndarray,   # bool    (F, P)
    t_sec:      np.ndarray,   # float32 (T,)
    t_norm:     np.ndarray,   # float32 (T,)
    output:     np.ndarray,   # float32 (F*T,)
    T:          int,
    sin_lut:    np.ndarray,   # float32 (_LUT_SIZE,) or empty
    lut_scale:  np.float32,
    use_lut:    bool,
):
    n_frames, n_partials = f_use.shape
    # prange → each frame on a separate thread (no output overlap)
    for i in prange(n_frames):
        base = i * T
        for p in range(n_partials):
            if not active[i, p]:
                continue
            pa = prev_a[i, p]
            ca = a32[i, p]
            fu = f_use[i, p]
            ph = phi_track[i, p]
            da = ca - pa
            for t in range(T):
                phase   = ph + TWO_PI_F32 * fu * t_sec[t]
                # Keep phase in [0, 2π) for LUT
                if use_lut:
                    idx  = int(phase * lut_scale) & (len(sin_lut) - 1)
                    s    = sin_lut[idx]
                else:
                    s    = np.sin(phase)
                # cubic bivaer interpolation
                tn = t_norm[t]
                tn2 = tn * tn
                fade = 3*tn2 - 2*tn2*tn
                amp = pa + da * fade
                output[base + t] += s * amp

# ─────────────────────────────────────────────────────────────────────────────
#  RSC6 parser  (Python/NumPy level — one-time cost)
# ─────────────────────────────────────────────────────────────────────────────

def parse_rsc(path: str) -> tuple[dict, np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        raw = f.read()

    if len(raw) < 35:
        raise ValueError("File too short to be RSC6.")

    (magic, version, sample_rate, frame_size, n_partials,
     total_samples, total_frames, mask_sz, k_freq, k_amp,
     born_data_sz, rice_freq_sz) = struct.unpack_from("<4sBIIHIIHBBII", raw, 0)

    if magic != b"RSC6":
        raise ValueError(f"Unknown magic: {magic!r}  (expected RSC6)")

    metadata = dict(
        magic=magic.decode(), version=version,
        sample_rate=sample_rate, frame_size=frame_size,
        n_partials=n_partials, total_samples=total_samples,
        total_frames=total_frames,
    )

    # ── Section offsets ───────────────────────────────────────────────────────
    bitmask_start    = 35
    bitmask_sz       = total_frames * 2 * mask_sz
    born_start       = bitmask_start + bitmask_sz
    rice_freq_start  = born_start    + born_data_sz
    rice_amp_start   = rice_freq_start + rice_freq_sz

    nyquist    = sample_rate / 2.0
    freq_scale = np.float32(nyquist / 65535.0)

    raw_arr = np.frombuffer(raw, dtype=np.uint8)

    # ── Bitmasks: decode entire section in one shot ───────────────────────────
    # Shape: (total_frames, 2, mask_sz) bytes → unpack → slice to n_partials
    bitmask_bytes = raw_arr[bitmask_start : bitmask_start + bitmask_sz]
    bitmask_bytes = bitmask_bytes.reshape(total_frames, 2, mask_sz)

    alive_flat_packed = bitmask_bytes[:, 0, :]          # (F, mask_sz)
    born_flat_packed  = bitmask_bytes[:, 1, :]

    # unpackbits returns (F, mask_sz*8); slice columns to n_partials, flatten
    alive_flat = np.unpackbits(alive_flat_packed, axis=1,
                               bitorder="little")[:, :n_partials].ravel()
    born_flat  = np.unpackbits(born_flat_packed,  axis=1,
                               bitorder="little")[:, :n_partials].ravel()

    # ── Pre-decode the born-partial table ─────────────────────────────────────
    # Each entry is 4 bytes (uint16 fq, uint16 amu).  We need to know which
    # frame and slot each entry belongs to, and provide them as sorted arrays
    # so the Numba loop can consume them in O(1) per frame.
    n_born_total = born_data_sz // 4
    born_table   = np.frombuffer(raw, dtype=np.uint16,
                                 count=n_born_total * 2, offset=born_start)
    born_table   = born_table.reshape(n_born_total, 2)   # [[fq, amu], ...]

    # Determine frame/slot for each born event (sequential scan, O(F*P))
    born_frames_list = []
    born_slots_list  = []
    born_idx = 0
    for i in range(total_frames):
        base = i * n_partials
        for slot in range(n_partials):
            if alive_flat[base + slot] and born_flat[base + slot]:
                born_frames_list.append(i)
                born_slots_list.append(slot)
                born_idx += 1
    born_frames_arr = np.array(born_frames_list, dtype=np.int32)
    born_slots_arr  = np.array(born_slots_list,  dtype=np.int32)

    # ── Output arrays ─────────────────────────────────────────────────────────
    freqs = np.zeros((total_frames, n_partials), dtype=np.float32)
    amps  = np.zeros((total_frames, n_partials), dtype=np.float32)

    print("   🔍 Parsing frames (JIT)…")
    _parse_frames_njit(
        raw_arr, alive_flat.astype(np.bool_), born_flat.astype(np.bool_),
        born_table.astype(np.uint16), born_slots_arr, born_frames_arr,
        rice_freq_start, rice_amp_start,
        int(k_freq), int(k_amp), freq_scale,
        total_frames, n_partials,
        freqs, amps,
    )

    return metadata, freqs, amps


# ─────────────────────────────────────────────────────────────────────────────
#  McAulay-Quatieri synthesis  (dispatches to JIT core)
# ─────────────────────────────────────────────────────────────────────────────

def synthesize(
    freqs:       np.ndarray,
    amps:        np.ndarray,
    frame_size:  int,
    sample_rate: int,
    use_lut:     bool = True,
) -> np.ndarray:
    n_frames, n_partials = freqs.shape
    T      = frame_size
    T_sec  = np.float32(T / sample_rate)

    f32    = freqs   # already float32 from parser
    a32    = amps    # already float32 from parser
    prev_a = np.empty_like(a32)
    prev_a[0]  = 0.0
    prev_a[1:] = a32[:-1]

    prev_f = np.empty_like(f32)
    prev_f[0]  = 0.0
    prev_f[1:] = f32[:-1]

    f_use   = np.where(f32 > 0, f32, prev_f)
    active  = (a32 > 0) | (prev_a > 0)

    # Phase track (sequential — must be serial)
    phi_track = np.zeros((n_frames, n_partials), dtype=np.float32)
    print("   🌀 Building phase track (JIT)…")
    _build_phi_track_njit(f_use, a32, prev_a, T_sec,
                          n_frames, n_partials, phi_track)

    t_sec  = np.arange(T, dtype=np.float32) / np.float32(sample_rate)
    t_norm = np.arange(T, dtype=np.float32) / np.float32(T)
    output = np.zeros(T * n_frames, dtype=np.float32)

    sin_lut   = _SIN_LUT   if use_lut else np.empty(0, dtype=np.float32)
    lut_scale = _LUT_SCALE if use_lut else np.float32(0.0)

    print("   🎵 Synthesis (parallel JIT)…")
    _synthesize_njit(
        f_use, a32, prev_a, phi_track, active,
        t_sec, t_norm, output, T,
        sin_lut, lut_scale, use_lut,
    )
    return output


# ─────────────────────────────────────────────────────────────────────────────
#  WAV writer
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def _normalize_clip_njit(samples: np.ndarray, pcm: np.ndarray):
    """Fused peak-normalize + clip + int16 cast in one pass."""
    peak = np.float32(0.0)
    for v in samples:
        a = v if v >= 0.0 else -v
        if a > peak:
            peak = a
    scale = np.float32(32767.0) / peak if peak > np.float32(1e-9) else np.float32(32767.0)
    for i in range(len(samples)):
        v = samples[i] * scale
        if   v >  32767.0: v =  32767.0
        elif v < -32767.0: v = -32767.0
        pcm[i] = np.int16(v)


def write_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.empty(len(samples), dtype=np.int16)
    _normalize_clip_njit(samples, pcm)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    dur = len(samples) / sample_rate
    print(f"  ✅ Wrote {len(samples):,} samples ({dur:.2f}s) → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Top-level
# ─────────────────────────────────────────────────────────────────────────────

def decode(input_path: str, output_path: str,
           override_sr: Optional[int], trim: bool,
           use_lut: bool) -> None:
    import time
    print(f"🔊 RSC Decoder (optimised)  —  {input_path}")

    t0 = time.perf_counter()
    meta, freqs, amps = parse_rsc(input_path)
    sr         = override_sr or meta["sample_rate"]
    frame_size = meta["frame_size"]
    dur        = meta["total_samples"] / sr
    print(f"   {meta['total_frames']:,} frames  |  {meta['n_partials']} partials  "
          f"|  {sr} Hz  |  {dur:.2f}s")

    output = synthesize(freqs, amps, frame_size, sr, use_lut=use_lut)

    if trim and meta.get("total_samples"):
        output = output[: meta["total_samples"]]

    write_wav(output_path, output, sr)
    elapsed = time.perf_counter() - t0
    print(f"   🎉 Done in {elapsed:.2f}s  (RTF {elapsed/dur:.3f}×)")


def main():
    p = argparse.ArgumentParser(
        description="RSC6 Decoder (optimised — Numba JIT + parallel synthesis)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",      "-i", required=True)
    p.add_argument("--output",     "-o", default=None)
    p.add_argument("--samplerate", "-r", type=int, default=None, choices=[22050, 44100])
    p.add_argument("--no-trim",         action="store_true")
    p.add_argument("--no-lut",          action="store_true",
                   help="Use math.sin instead of sin LUT (faster if SVML present)")
    args = p.parse_args()
    out  = args.output or (args.input.removesuffix(".rsc") + "_decoded.wav")
    decode(args.input, out, args.samplerate,
           trim=not args.no_trim, use_lut=not args.no_lut)


if __name__ == "__main__":
    main()