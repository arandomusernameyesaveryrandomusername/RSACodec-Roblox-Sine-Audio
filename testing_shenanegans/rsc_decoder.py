from __future__ import annotations

"""
rsc_decoder.py — Roblox Sine Codec (RSC) Decoder  —  RSC7

Format: RSC7  (backward-compatible magic check; RSC6 files still work,
               they just won't have a noise layer)

Synthesis:
  • McAulay-Quatieri (MQ) interpolated sinusoidal synthesis (unchanged).
  • NEW — Noise layer (SMS residual model):
      For each frame, N_NOISE_BANDS per-band RMS amplitudes are decoded
      from Section 5 of the file.  White noise is generated, shaped by a
      linear-interpolated spectral envelope matching those band energies,
      then added to the sine output.  This recovers transient texture,
      breathiness, fricatives, and other noise-like components that
      sinusoidal modelling leaves behind.

RSC7 Header (37 bytes):
  "RSC7" | u8 ver | u32 sr | u32 frame_sz | u16 n_partials |
  u32 total_samples | u32 total_frames | u16 mask_sz |
  u8 k_freq | u8 k_amp | u32 born_data_sz | u32 rice_freq_sz |
  u8 n_noise_bands | u32 noise_data_sz

RSC6 files (35-byte header, magic "RSC6") are still decoded — they just
produce sinusoidal-only output with no noise layer.

Usage:
    python rsc_decoder.py --input audio.rsc --output decoded.wav
    python rsc_decoder.py --input audio.rsc --output decoded.wav --no-noise
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

# Number of log-spaced noise bands (must match encoder)
N_NOISE_BANDS = 32


# ─────────────────────────────────────────────────────────────
#  Mu-law expand
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
#  Noise-band edge pre-computation  (mirrors encoder exactly)
# ─────────────────────────────────────────────────────────────
def _noise_band_edges(sample_rate: int, n_bands: int,
                      f_lo: float = 20.0) -> np.ndarray:
    f_hi = sample_rate / 2.0
    return np.exp(np.linspace(np.log(f_lo), np.log(f_hi), n_bands + 1)).astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  RSC6/RSC7 parser
# ─────────────────────────────────────────────────────────────
def parse_rsc(path: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Returns:
        metadata   : dict
        freqs      : (nF, nP) float32  Hz
        amps       : (nF, nP) float32  linear [0,1]
        noise_bands: (nF, n_bands) float32 linear [0,1], or None for RSC6
    """
    with open(path, "rb") as f:
        raw = f.read()

    if len(raw) < 35:
        raise ValueError("File too short.")

    magic = raw[:4]
    if magic == b"RSC6":
        version = 6
        (_, version, sample_rate, frame_size, n_partials,
         total_samples, total_frames, mask_sz, k_freq, k_amp,
         born_data_sz, rice_freq_sz) = struct.unpack_from("<4sBIIHIIHBBII", raw, 0)
        n_noise_bands = 0
        noise_data_sz = 0
        header_sz     = 35
    elif magic == b"RSC7":
        if len(raw) < 37:
            raise ValueError("RSC7 file too short for header.")
        (_, version, sample_rate, frame_size, n_partials,
         total_samples, total_frames, mask_sz, k_freq, k_amp,
         born_data_sz, rice_freq_sz,
         n_noise_bands, noise_data_sz) = struct.unpack_from("<4sBIIHIIHBBIIBI", raw, 0)
        header_sz = 37
    else:
        raise ValueError(f"Unknown magic: {magic!r}  (expected RSC6 or RSC7)")

    metadata = dict(
        magic=magic.decode(), version=version,
        sample_rate=sample_rate, frame_size=frame_size,
        n_partials=n_partials, total_samples=total_samples,
        total_frames=total_frames,
        n_noise_bands=n_noise_bands,
    )

    # ── Section offsets ───────────────────────────────────────────────────
    bitmask_start   = header_sz
    bitmask_sz      = total_frames * 2 * mask_sz
    born_start      = bitmask_start + bitmask_sz
    rice_freq_start = born_start    + born_data_sz
    rice_amp_start  = rice_freq_start + rice_freq_sz
    noise_start     = rice_amp_start  + (len(raw) - rice_amp_start - noise_data_sz) \
                      if n_noise_bands > 0 else None

    # Simpler: noise section is the last noise_data_sz bytes
    if n_noise_bands > 0:
        noise_start = len(raw) - noise_data_sz

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
    amps_mu= np.zeros((total_frames, n_partials), dtype=np.uint16)

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

    # ── Pass 4: noise bands (RSC7 only) ──────────────────────────────────
    noise_bands_linear = None
    if n_noise_bands > 0 and noise_start is not None:
        nb_raw = np.frombuffer(raw, dtype=np.uint8,
                               count=total_frames * n_noise_bands,
                               offset=noise_start).reshape(total_frames, n_noise_bands)
        noise_bands_linear = _mulaw_decode(nb_raw)   # (nF, nB) float32 [0,1]

    return metadata, freqs, amps, noise_bands_linear


# ─────────────────────────────────────────────────────────────
#  Noise synthesis
# ─────────────────────────────────────────────────────────────
def synthesize_noise(
    noise_bands:  np.ndarray,   # (nF, n_bands) float32 [0,1]
    frame_size:   int,
    sample_rate:  int,
    rng:          np.random.Generator,
) -> np.ndarray:
    """
    Synthesise the noise residual layer.

    For each frame:
      1. Generate white noise of length frame_size.
      2. FFT it, scale each bin by the linearly-interpolated spectral
         envelope derived from the per-band RMS amplitudes.
      3. IFFT back to time domain.
      4. Apply a Hann window to avoid frame-edge clicks, overlap-add.

    The overlap-add (OLA) with 50% overlap ensures smooth transitions
    between frames — identical to SMS-style noise synthesis.

    Returns float32 array of length nF * frame_size.
    """
    nF, n_bands    = noise_bands.shape
    T              = frame_size
    hop            = T // 2          # 50 % overlap
    total_len      = nF * T + hop    # extra hop for OLA tail
    output         = np.zeros(total_len, dtype=np.float32)
    win            = np.sqrt(np.hanning(T)).astype(np.float32)  # sqrt-Hann for OLA
    fft_freqs      = np.fft.rfftfreq(T, d=1.0 / sample_rate).astype(np.float32)
    n_bins         = len(fft_freqs)
    edges          = _noise_band_edges(sample_rate, n_bands)
    band_centers   = np.sqrt(edges[:-1] * edges[1:]).astype(np.float32)  # geometric center

    for i in range(nF):
        # Build spectral envelope by linear-interp of band amplitudes onto FFT bins
        # np.interp clamps at endpoints — correct behaviour
        env    = np.interp(fft_freqs, band_centers, noise_bands[i]).astype(np.float32)

        # White noise → frequency domain
        noise  = rng.standard_normal(T).astype(np.float32) * win
        spec   = np.fft.rfft(noise)

        # Shape by envelope
        spec  *= env.astype(np.complex64)

        # Back to time
        frame  = np.fft.irfft(spec, n=T).astype(np.float32)

        # Overlap-add at current frame start (frames placed every T samples,
        # with an extra OLA window one hop before)
        start  = i * T
        output[start        : start + T ] += frame * win
        # Second copy shifted by hop for smooth transitions
        if start + hop + T <= total_len:
            output[start + hop : start + hop + T] += frame * win * 0.5

    # Trim to nF*T
    return output[:nF * T]


# ─────────────────────────────────────────────────────────────
#  McAulay-Quatieri Interpolated Synthesis  (click-free)
# ─────────────────────────────────────────────────────────────
def synthesize(
    freqs:       np.ndarray,
    amps:        np.ndarray,
    frame_size:  int,
    sample_rate: int,
    noise_bands: np.ndarray | None = None,   # (nF, nB) float32, or None
    noise_gain:  float = 1.0,
) -> np.ndarray:
    """
    Phase-continuous, amplitude-interpolated sinusoidal synthesis
    + optional noise layer.
    """
    n_frames, n_partials = freqs.shape
    T       = frame_size
    T_sec   = np.float32(T / sample_rate)
    TWO_PI  = np.float32(2.0 * math.pi)
    output  = np.zeros(T * n_frames, dtype=np.float32)

    t_sec  = np.arange(T, dtype=np.float32) / np.float32(sample_rate)
    t_norm = np.arange(T, dtype=np.float32) / np.float32(T)

    f32 = freqs.astype(np.float32)
    a32 = amps.astype(np.float32)

    prev_f = np.vstack([np.zeros((1, n_partials), np.float32), f32[:-1]])
    prev_a = np.vstack([np.zeros((1, n_partials), np.float32), a32[:-1]])

    f_use  = np.where(f32 > 0, f32, prev_f)
    active = (a32 > 0) | (prev_a > 0)

    phi = np.zeros(n_partials, dtype=np.float32)
    phi_track = np.zeros((n_frames, n_partials), dtype=np.float32)

    for i in range(n_frames):
        born = (prev_a[i] == 0) & (a32[i] > 0)
        phi  = np.where(born, np.float32(0.0), phi)
        phi_track[i] = phi
        phi = (phi + TWO_PI * f_use[i] * T_sec) % TWO_PI
        phi = np.where(a32[i] == 0, np.float32(0.0), phi)

    for i in tqdm(range(n_frames), desc="   Synthesis",
                  unit="frame", dynamic_ncols=True,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} frames  [{elapsed}<{remaining}  {rate_fmt}]"):
        act = active[i]
        if not act.any():
            continue

        pa  = prev_a[i, act]
        ca  = a32[i, act]
        fu  = f_use[i, act]
        ph  = phi_track[i, act]

        phase   = ph[:, None] + TWO_PI * fu[:, None] * t_sec[None, :]
        amp_env = pa[:, None] + (ca - pa)[:, None] * t_norm[None, :]
        output[i * T : (i + 1) * T] = (amp_env * np.sin(phase)).sum(axis=0)

    # ── Add noise layer ───────────────────────────────────────────────────
    if noise_bands is not None and noise_gain > 0.0:
        rng   = np.random.default_rng(seed=0)   # deterministic for reproducibility
        noise = synthesize_noise(noise_bands, T, sample_rate, rng)
        output += noise.astype(np.float32) * np.float32(noise_gain)

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
           override_sr: int | None, trim: bool,
           noise_gain: float) -> None:
    print(f"🔊 RSC Decoder  —  {input_path}")
    meta, freqs, amps, noise_bands = parse_rsc(input_path)
    sr         = override_sr or meta["sample_rate"]
    frame_size = meta["frame_size"]
    dur        = meta["total_samples"] / sr
    has_noise  = noise_bands is not None
    print(f"   {meta['total_frames']} frames  |  {meta['n_partials']} partials  "
          f"|  {sr} Hz  |  {dur:.2f}s  |  "
          f"noise={'yes (' + str(meta['n_noise_bands']) + ' bands)' if has_noise else 'none (RSC6)'}")

    output = synthesize(freqs, amps, frame_size, sr,
                        noise_bands=noise_bands,
                        noise_gain=noise_gain)
    if trim and meta.get("total_samples"):
        output = output[:meta["total_samples"]]
    peak = np.max(np.abs(output))
    if peak > 1e-9:
        output /= peak
    write_wav(output_path, output, sr)
    print("   🎉 Done!")


def main():
    p = argparse.ArgumentParser(description="RSC6/RSC7 Decoder",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input",      "-i", required=True)
    p.add_argument("--output",     "-o", default=None)
    p.add_argument("--samplerate", "-r", type=int, default=None, choices=[22050, 44100])
    p.add_argument("--no-trim",         action="store_true")
    p.add_argument("--no-noise",        action="store_true",
                   help="Disable noise layer (sinusoidal-only output)")
    p.add_argument("--noise-gain",      type=float, default=1.0,
                   help="Scale factor for noise layer amplitude (default 1.0)")
    args = p.parse_args()
    out  = args.output or (args.input.removesuffix(".rsc") + "_decoded.wav")
    decode(args.input, out, args.samplerate, not args.no_trim,
           noise_gain=0.0 if args.no_noise else args.noise_gain)

if __name__ == "__main__":
    main()