from __future__ import annotations

"""
rsc_decoder.py — Roblox Sine Codec (RSC) Decoder

Format: RSC6  (see encoder for full spec)

Synthesis uses McAulay-Quatieri (MQ) interpolated sinusoidal synthesis:
  • Amplitude is linearly ramped from the previous frame's value to the current
    value within each frame — eliminates rectangular-envelope Gibbs ringing.
  • Phase is accumulated continuously across frames — no inter-frame phase
    discontinuity regardless of frequency.
  • Born partials fade in from amplitude 0 with phase starting at 0 — clean onset.
  • Dead partials fade out to amplitude 0 — clean release with no hard cut.

All synthesis is done in float32 throughout for ~10× speedup vs float64.

Usage:
    python rsc_decoder.py --input audio.rsc --output decoded.wav
"""

import argparse
import math
import struct
import wave

import numpy as np
from tqdm import tqdm

TWO_PI    = 2.0 * math.pi
MU        = 255.0
_LOG1P_MU = math.log1p(MU)


# ─────────────────────────────────────────────────────────────
#  Mu-law expand  (inverse of encoder's _mulaw_encode)
# ─────────────────────────────────────────────────────────────
def _mulaw_decode(u: np.ndarray) -> np.ndarray:
    """uint8 [0,255] → float32 [0,1]  (inverse mu-law expansion)"""
    u_norm = u.astype(np.float32) / np.float32(MU)
    return (np.exp(u_norm * np.float32(_LOG1P_MU)) - 1.0) / np.float32(MU)


# ─────────────────────────────────────────────────────────────
#  Rice decode
# ─────────────────────────────────────────────────────────────
def _zigzag_dec(u: int) -> int:
    return (u >> 1) if (u & 1) == 0 else -((u >> 1) + 1)


class _BitReader:
    """MSB-first bit reader over a bytes-like object."""
    __slots__ = ("_data", "_pos", "_buf", "_bits_left")

    def __init__(self, data: bytes, start: int):
        self._data      = data
        self._pos       = start
        self._buf       = 0
        self._bits_left = 0

    def read_bit(self) -> int:
        if self._bits_left == 0:
            self._buf       = self._data[self._pos]
            self._pos      += 1
            self._bits_left = 8
        self._bits_left -= 1
        return (self._buf >> self._bits_left) & 1

    def read_rice(self, k: int) -> int:
        q = 0
        while self.read_bit() == 0:
            q += 1
        r = 0
        for _ in range(k):
            r = (r << 1) | self.read_bit()
        return _zigzag_dec((q << k) | r)


# ─────────────────────────────────────────────────────────────
#  RSC6 parser
# ─────────────────────────────────────────────────────────────
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

    # ── Section offsets ───────────────────────────────────────────────────
    bitmask_start   = 35
    bitmask_sz      = total_frames * 2 * mask_sz
    born_start      = bitmask_start + bitmask_sz
    rice_freq_start = born_start    + born_data_sz
    rice_amp_start  = rice_freq_start + rice_freq_sz

    nyquist    = sample_rate / 2.0
    freq_scale = nyquist / 65535.0

    # ── Pass 1: read all bitmasks ─────────────────────────────────────────
    alive_masks = []
    born_masks  = []
    pos = bitmask_start
    for _ in range(total_frames):
        alive_raw = np.frombuffer(raw, dtype=np.uint8, count=mask_sz, offset=pos)
        born_raw  = np.frombuffer(raw, dtype=np.uint8, count=mask_sz, offset=pos + mask_sz)
        alive_masks.append(np.unpackbits(alive_raw, bitorder="little")[:n_partials].astype(bool))
        born_masks.append( np.unpackbits(born_raw,  bitorder="little")[:n_partials].astype(bool))
        pos += 2 * mask_sz

    # ── Pass 2: Rice-decode freq and amp delta streams ────────────────────
    freq_reader = _BitReader(raw, rice_freq_start)
    amp_reader  = _BitReader(raw, rice_amp_start)

    # ── Pass 3: reconstruct freq/amp arrays ──────────────────────────────
    freqs   = np.zeros((total_frames, n_partials), dtype=np.float32)
    amps_mu = np.zeros((total_frames, n_partials), dtype=np.uint8)

    curr_fq  = np.zeros(n_partials, dtype=np.int32)
    curr_amu = np.zeros(n_partials, dtype=np.int32)
    born_pos = born_start

    for i in tqdm(range(total_frames), desc="   Parsing  ",
                  unit="frame", dynamic_ncols=True,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} frames  [{elapsed}<{remaining}  {rate_fmt}]"):
        alive = alive_masks[i]
        born  = born_masks[i]
        dead  = ~alive

        curr_fq[dead]  = 0
        curr_amu[dead] = 0

        for slot in np.where(alive)[0]:
            if born[slot]:
                fq, amu    = struct.unpack_from("<HB", raw, born_pos)
                born_pos  += 3
                curr_fq[slot]  = fq
                curr_amu[slot] = amu
            else:
                curr_fq[slot]  = max(0, min(65535, curr_fq[slot]  + freq_reader.read_rice(k_freq)))
                curr_amu[slot] = max(0, min(255,   curr_amu[slot] + amp_reader.read_rice(k_amp)))

            freqs[i, slot]   = curr_fq[slot] * freq_scale
            amps_mu[i, slot] = curr_amu[slot]

    amps = _mulaw_decode(amps_mu)
    return metadata, freqs, amps


# ─────────────────────────────────────────────────────────────
#  McAulay-Quatieri Interpolated Synthesis  (click-free)
# ─────────────────────────────────────────────────────────────
def synthesize(
    freqs:       np.ndarray,   # (n_frames, n_partials) float32, Hz
    amps:        np.ndarray,   # (n_frames, n_partials) float32, linear [0,1]
    frame_size:  int,
    sample_rate: int,
) -> np.ndarray:
    """
    Phase-continuous, amplitude-interpolated sinusoidal synthesis.

    Within each frame, for each active partial:
      amp(t)   = prev_amp + (curr_amp - prev_amp) * t/T      — linear ramp
      phase(t) = phi_0 + 2π · f · t                          — constant freq
      phi_0    is carried across frame boundaries for continuity

    Birth (prev_amp == 0): amplitude ramps from 0 → clean fade-in.
    Death (curr_amp == 0): amplitude ramps to 0 → clean fade-out,
                           then phi is zeroed so any future rebirth starts clean.

    The frequency-kink at frame boundaries (constant-freq approximation) is
    inaudible in practice because:
      a) The amplitude is near-zero at birth/death (the discontinuous moments).
      b) Continuous partials have smooth frequency tracks from the encoder.

    All arithmetic is float32 for speed (~10x faster than float64 for sin).
    """
    n_frames, n_partials = freqs.shape
    T       = frame_size
    T_sec   = np.float32(T / sample_rate)
    TWO_PI  = np.float32(2.0 * math.pi)
    output  = np.zeros(T * n_frames, dtype=np.float32)

    # Per-frame time ramps — shape (T,), float32
    t_sec  = np.arange(T, dtype=np.float32) / np.float32(sample_rate)
    t_norm = np.arange(T, dtype=np.float32) / np.float32(T)   # 0 .. (T-1)/T

    f32 = freqs.astype(np.float32)    # (F, P)
    a32 = amps.astype(np.float32)     # (F, P)

    # Shift-by-one to get previous-frame values (prev[0] = zeros = silence)
    prev_f = np.vstack([np.zeros((1, n_partials), np.float32), f32[:-1]])  # (F, P)
    prev_a = np.vstack([np.zeros((1, n_partials), np.float32), a32[:-1]])  # (F, P)

    # During a fade-out (curr_f == 0), hold the previous frequency so the
    # partial doesn't chirp down to 0 Hz — it just fades out at its last pitch.
    f_use = np.where(f32 > 0, f32, prev_f)   # (F, P)

    # Active mask: frames where either prev or curr amplitude is nonzero
    active = (a32 > 0) | (prev_a > 0)        # (F, P) bool

    # Accumulated per-partial phase at the start of each frame.
    # Computed sequentially (O(F*P) additions) — fast, ~3ms for 8224*384.
    phi = np.zeros(n_partials, dtype=np.float32)
    phi_track = np.zeros((n_frames, n_partials), dtype=np.float32)

    for i in range(n_frames):
        # Born partials start with phase 0 (phi is already 0 from a previous
        # death reset, but we explicitly zero it here for clarity).
        born = (prev_a[i] == 0) & (a32[i] > 0)
        phi  = np.where(born, np.float32(0.0), phi)

        phi_track[i] = phi

        # Advance phase by 2π·f·T for all partials (even inactive ones —
        # the phase of an inactive partial is irrelevant since we reset it
        # to 0 on death, but advancing prevents stale values if needed).
        phi = (phi + TWO_PI * f_use[i] * T_sec) % TWO_PI

        # Zero phase after death so any future birth starts at phase 0.
        phi = np.where(a32[i] == 0, np.float32(0.0), phi)

    # ── Main synthesis loop: one iteration per frame, vectorized over partials
    for i in tqdm(range(n_frames), desc="   Synthesis",
                  unit="frame", dynamic_ncols=True,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} frames  [{elapsed}<{remaining}  {rate_fmt}]"):
        act = active[i]               # (P,) bool mask
        if not act.any():
            continue

        pf  = prev_f[i, act]          # (K,) previous frame frequency
        pa  = prev_a[i, act]          # (K,) previous frame amplitude
        ca  = a32[i, act]             # (K,) current frame amplitude
        fu  = f_use[i, act]           # (K,) frequency to use (held on death)
        ph  = phi_track[i, act]       # (K,) phase at frame start

        # Phase trajectory within this frame: φ(t) = φ₀ + 2π·f·t
        # Shape: (K, T) via broadcasting — no Python loop, pure numpy
        phase   = ph[:, None] + TWO_PI * fu[:, None] * t_sec[None, :]   # (K, T)

        # Linear amplitude envelope from prev to curr — float32 sin is ~10x
        # faster than float64 on modern hardware with vector extensions
        amp_env = pa[:, None] + (ca - pa)[:, None] * t_norm[None, :]    # (K, T)

        # Sum all K partials into this frame's output samples
        output[i * T : (i + 1) * T] = (amp_env * np.sin(phase)).sum(axis=0)

    return output


# ─────────────────────────────────────────────────────────────
#  WAV Writer
# ─────────────────────────────────────────────────────────────
def write_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    print(f"  ✅ Wrote {len(samples)} samples ({len(samples)/sample_rate:.2f}s) → {path}")


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────
def decode(input_path: str, output_path: str,
           override_sr: int | None, trim: bool) -> None:
    print(f"🔊 RSC Decoder  —  {input_path}")
    meta, freqs, amps = parse_rsc(input_path)
    sr         = override_sr or meta["sample_rate"]
    frame_size = meta["frame_size"]
    dur        = meta["total_samples"] / sr
    print(f"   {meta['total_frames']} frames  |  {meta['n_partials']} partials  "
          f"|  {sr} Hz  |  {dur:.2f}s")

    output = synthesize(freqs, amps, frame_size, sr)
    if trim and meta.get("total_samples"):
        output = output[:meta["total_samples"]]
    peak = np.max(np.abs(output))
    if peak > 1e-9:
        output /= peak
    write_wav(output_path, output, sr)
    print("   🎉 Done!")


def main():
    p = argparse.ArgumentParser(description="RSC6 Decoder",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input",      "-i", required=True)
    p.add_argument("--output",     "-o", default=None)
    p.add_argument("--samplerate", "-r", type=int, default=None, choices=[22050, 44100])
    p.add_argument("--no-trim",         action="store_true")
    args = p.parse_args()
    out  = args.output or (args.input.removesuffix(".rsc") + "_decoded.wav")
    decode(args.input, out, args.samplerate, not args.no_trim)

if __name__ == "__main__":
    main()