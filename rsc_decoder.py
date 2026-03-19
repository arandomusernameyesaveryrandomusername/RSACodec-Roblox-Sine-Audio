from __future__ import annotations
"""
rsc_decoder.py — Roblox Sine Codec (RSC) Decoder

Formats: RSC6, RSC7  (RSC7 adds ERB residual band envelope)

Usage:
    python rsc_decoder.py --input audio.rsc --output decoded.wav
    python rsc_decoder.py --input audio.rsc --output decoded.wav --residual-mix 0.4
"""

import argparse
import math
import struct
import wave

import numpy as np

TWO_PI    = 2.0 * math.pi
MU        = 255.0
_LOG1P_MU = math.log1p(MU)

# ─────────────────────────────────────────────────────────────
#  Mu-law expand  (inverse of encoder's _mulaw_encode)
# ─────────────────────────────────────────────────────────────
def _mulaw_decode(u: np.ndarray) -> np.ndarray:
    """uint8 [0,255] → float64 [0,1]"""
    u_norm = u.astype(np.float64) / MU
    return (np.exp(u_norm * _LOG1P_MU) - 1.0) / MU

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
#  ERB band layout  (must match encoder's _erb_band_bins)
# ─────────────────────────────────────────────────────────────
def _erb_band_bins(
    frame_size: int,
    sample_rate: int,
    n_bands: int,
) -> list[tuple[int, int]]:
    freqs   = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    erb     = 21.4 * np.log10(4.37e-3 * np.maximum(freqs, 1e-3) + 1)
    erb_min = erb[1]
    erb_max = erb[-1]
    edges   = np.linspace(erb_min, erb_max, n_bands + 1)
    bands: list[tuple[int, int]] = []
    for b in range(n_bands):
        lo = int(np.searchsorted(erb, edges[b]))
        hi = int(np.searchsorted(erb, edges[b + 1]))
        hi = max(hi, lo + 1)
        bands.append((lo, min(hi, len(freqs))))
    return bands

# ─────────────────────────────────────────────────────────────
#  RSC parser  (handles RSC6 and RSC7)
# ─────────────────────────────────────────────────────────────
def parse_rsc(path: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Returns (metadata, freqs, amps, band_energies).
    band_energies is (n_frames, n_bands) float32 in [0,1], or None for RSC6.
    """
    with open(path, "rb") as f:
        raw = f.read()

    if len(raw) < 35:
        raise ValueError("File too short.")

    magic = raw[:4]
    if magic == b"RSC7":
        if len(raw) < 40:
            raise ValueError("File too short to be RSC7.")
        (_, version, sample_rate, frame_size, n_partials,
         total_samples, total_frames, mask_sz, k_freq, k_amp,
         born_data_sz, rice_freq_sz,
         n_bands, residual_sz) = struct.unpack_from("<4sBIIHIIHBBIIBI", raw, 0)
        header_sz = 40
        is_rsc7   = True
    elif magic == b"RSC6":
        (_, version, sample_rate, frame_size, n_partials,
         total_samples, total_frames, mask_sz, k_freq, k_amp,
         born_data_sz, rice_freq_sz) = struct.unpack_from("<4sBIIHIIHBBII", raw, 0)
        header_sz = 35
        n_bands   = 0
        residual_sz = 0
        is_rsc7   = False
    else:
        raise ValueError(f"Unknown magic: {magic!r}  (expected RSC6 or RSC7)")

    metadata = dict(
        magic=magic.decode(), version=version,
        sample_rate=sample_rate, frame_size=frame_size,
        n_partials=n_partials, total_samples=total_samples,
        total_frames=total_frames,
        n_bands=n_bands,
    )

    # ── Section offsets ───────────────────────────────────────────────
    bitmask_start   = header_sz
    bitmask_sz      = total_frames * 2 * mask_sz
    born_start      = bitmask_start + bitmask_sz
    rice_freq_start = born_start    + born_data_sz
    rice_amp_start  = rice_freq_start + rice_freq_sz
    residual_start  = rice_amp_start  + (len(raw) - rice_amp_start - residual_sz) \
                      if is_rsc7 else None
    # simpler: residual is always the last residual_sz bytes
    if is_rsc7:
        residual_start = len(raw) - residual_sz

    nyquist    = sample_rate / 2.0
    freq_scale = nyquist / 65535.0

    # ── Pass 1: bitmasks ──────────────────────────────────────────────
    alive_masks = []
    born_masks  = []
    pos = bitmask_start
    for _ in range(total_frames):
        alive_raw = np.frombuffer(raw, dtype=np.uint8, count=mask_sz, offset=pos)
        born_raw  = np.frombuffer(raw, dtype=np.uint8, count=mask_sz, offset=pos + mask_sz)
        alive_masks.append(np.unpackbits(alive_raw, bitorder="little")[:n_partials].astype(bool))
        born_masks.append( np.unpackbits(born_raw,  bitorder="little")[:n_partials].astype(bool))
        pos += 2 * mask_sz

    # ── Pass 2: Rice readers ──────────────────────────────────────────
    freq_reader = _BitReader(raw, rice_freq_start)
    amp_reader  = _BitReader(raw, rice_amp_start)

    # ── Pass 3: reconstruct freq/amp ─────────────────────────────────
    freqs   = np.zeros((total_frames, n_partials), dtype=np.float32)
    amps_mu = np.zeros((total_frames, n_partials), dtype=np.uint8)

    curr_fq  = np.zeros(n_partials, dtype=np.int32)
    curr_amu = np.zeros(n_partials, dtype=np.int32)
    born_pos = born_start

    for i in range(total_frames):
        alive = alive_masks[i]
        born  = born_masks[i]
        dead  = ~alive
        curr_fq[dead]  = 0
        curr_amu[dead] = 0

        for slot in np.where(alive)[0]:
            if born[slot]:
                fq, amu        = struct.unpack_from("<HB", raw, born_pos)
                born_pos      += 3
                curr_fq[slot]  = fq
                curr_amu[slot] = amu
            else:
                curr_fq[slot]  = max(0, min(65535, curr_fq[slot]  + freq_reader.read_rice(k_freq)))
                curr_amu[slot] = max(0, min(255,   curr_amu[slot] + amp_reader.read_rice(k_amp)))

            freqs[i, slot]   = curr_fq[slot] * freq_scale
            amps_mu[i, slot] = curr_amu[slot]

        if (i + 1) % 500 == 0 or (i + 1) == total_frames:
            print(f"   ... parsed frame {i+1}/{total_frames}", end="\r")
    print()

    amps = _mulaw_decode(amps_mu).astype(np.float32)

    # ── Pass 4: residual band envelope (RSC7 only) ────────────────────
    band_energies: np.ndarray | None = None
    if is_rsc7 and residual_sz > 0:
        raw_bands = np.frombuffer(raw, dtype=np.uint8,
                                  count=total_frames * n_bands,
                                  offset=residual_start)
        # reshape and normalise back to [0, 1]
        band_energies = (raw_bands.reshape(total_frames, n_bands).astype(np.float32) / 255.0)

    return metadata, freqs, amps, band_energies

# ─────────────────────────────────────────────────────────────
#  Residual noise synthesis
# ─────────────────────────────────────────────────────────────
def _synthesize_noise(
    band_energies: np.ndarray,      # (n_frames, n_bands) float32 [0,1]
    bands: list[tuple[int, int]],   # ERB bin ranges
    frame_size: int,
    sample_rate: int,
    mix: float,
) -> np.ndarray:
    """
    Generate shaped white noise from the residual band envelope and
    return a float32 array the same length as the sine output.

    Each frame: white noise → rfft → scale each ERB band by its stored
    energy → irfft → overlap-add with a Hann window for smooth frames.
    """
    n_frames   = band_energies.shape[0]
    n_fft_bins = frame_size // 2 + 1
    out_len    = n_frames * frame_size
    output     = np.zeros(out_len, dtype=np.float64)
    win        = np.hanning(frame_size)

    rng = np.random.default_rng(seed=0)   # deterministic seed — same noise every decode

    for i in range(n_frames):
        noise      = rng.standard_normal(frame_size) * win
        noise_spec = np.fft.rfft(noise)

        for b, (lo, hi) in enumerate(bands):
            noise_spec[lo:hi] *= band_energies[i, b]

        frame = np.fft.irfft(noise_spec, n=frame_size)
        output[i * frame_size : (i + 1) * frame_size] += frame

        if (i + 1) % 500 == 0 or (i + 1) == n_frames:
            print(f"   ... noise frame {i+1}/{n_frames}", end="\r")
    print()

    # normalise noise independently before mixing so mix ratio is meaningful
    peak = np.max(np.abs(output))
    if peak > 1e-9:
        output /= peak

    return (output * mix).astype(np.float32)

# ─────────────────────────────────────────────────────────────
#  Phase-Continuous Sine Synthesis
# ─────────────────────────────────────────────────────────────
def synthesize(
    freqs:        np.ndarray,
    amps:         np.ndarray,
    frame_size:   int,
    sample_rate:  int,
    band_energies: np.ndarray | None = None,
    residual_mix:  float = 0.35,
) -> np.ndarray:
    n_frames, n_partials = freqs.shape
    output = np.zeros(frame_size * n_frames, dtype=np.float64)
    t      = np.arange(frame_size, dtype=np.float64) / sample_rate
    phi    = np.zeros(n_partials, dtype=np.float64)

    for i in range(n_frames):
        f      = freqs[i].astype(np.float64)
        a      = amps[i].astype(np.float64)
        active = a > 1e-6
        if active.any():
            fa = f[active]; aa = a[active]; pa = phi[active]
            output[i * frame_size : (i + 1) * frame_size] = (
                aa[:, None] * np.sin(TWO_PI * fa[:, None] * t + pa[:, None])
            ).sum(axis=0)
        phi = (phi + TWO_PI * f * frame_size / sample_rate) % TWO_PI
        phi[~active] = 0.0

        if (i + 1) % 500 == 0 or (i + 1) == n_frames:
            print(f"   ... synthesized frame {i+1}/{n_frames}", end="\r")
    print()

    output = output.astype(np.float32)

    # ── Mix in shaped noise residual (RSC7 only) ──────────────────────
    if band_energies is not None and residual_mix > 0.0:
        print(f"   Synthesizing residual noise (mix={residual_mix:.2f}) ...")
        n_bands = band_energies.shape[1]
        bands   = _erb_band_bins(frame_size, sample_rate, n_bands)
        noise   = _synthesize_noise(band_energies, bands, frame_size, sample_rate, residual_mix)
        output  = output + noise[:len(output)]

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
           residual_mix: float) -> None:
    print(f"🔊 RSC Decoder  —  {input_path}")
    meta, freqs, amps, band_energies = parse_rsc(input_path)
    sr         = override_sr or meta["sample_rate"]
    frame_size = meta["frame_size"]
    dur        = meta["total_samples"] / sr
    fmt        = meta["magic"]
    bands_info = f"  |  {meta['n_bands']} residual bands" if meta["n_bands"] else ""
    print(f"   [{fmt}]  {meta['total_frames']} frames  |  {meta['n_partials']} partials"
          f"  |  {sr} Hz  |  {dur:.2f}s{bands_info}")

    output = synthesize(freqs, amps, frame_size, sr,
                        band_energies=band_energies,
                        residual_mix=residual_mix)
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
    p.add_argument("--input",        "-i", required=True)
    p.add_argument("--output",       "-o", default=None)
    p.add_argument("--samplerate",   "-r", type=int, default=None, choices=[22050, 44100])
    p.add_argument("--residual-mix", "-m", type=float, default=0.35,
                   help="Residual noise mix level [0.0–1.0] (RSC7 only)")
    p.add_argument("--no-trim",           action="store_true")
    args = p.parse_args()
    out  = args.output or (args.input.removesuffix(".rsc") + "_decoded.wav")
    decode(args.input, out, args.samplerate, not args.no_trim, args.residual_mix)

if __name__ == "__main__":
    main()