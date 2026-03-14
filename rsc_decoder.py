"""
rsc_decoder.py — Roblox Sine Codec (RSC) Decoder

Format: RSC6  (see encoder for full spec)

Usage:
    python rsc_decoder.py --input audio.rsc --output decoded.wav
"""

import argparse
import math
import struct
import wave

import numpy as np

TWO_PI    = 2.0 * math.pi


# ─────────────────────────────────────────────────────────────
#  Rice decode
# ─────────────────────────────────────────────────────────────
def _zigzag_dec(u: int) -> int:
    """Non-negative uint → signed int"""
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
        """Decode one Rice(k) symbol and zigzag-decode it to a signed int."""
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

    # ── Pass 1: read all bitmasks + count born/continuing per frame ───────
    alive_masks = []   # list of bool arrays
    born_masks  = []
    pos = bitmask_start
    for _ in range(total_frames):
        alive_raw = np.frombuffer(raw, dtype=np.uint8, count=mask_sz, offset=pos)
        born_raw  = np.frombuffer(raw, dtype=np.uint8, count=mask_sz, offset=pos + mask_sz)
        alive_masks.append(np.unpackbits(alive_raw, bitorder="little")[:n_partials].astype(bool))
        born_masks.append( np.unpackbits(born_raw,  bitorder="little")[:n_partials].astype(bool))
        pos += 2 * mask_sz

    # ── Pass 2: count total continuing partials for pre-allocation ────────
    n_continuing = sum(
        int(np.sum(alive_masks[i] & ~born_masks[i]))
        for i in range(total_frames)
    )

    # ── Pass 3: Rice-decode freq and amp delta streams ────────────────────
    freq_reader = _BitReader(raw, rice_freq_start)
    amp_reader  = _BitReader(raw, rice_amp_start)

    # ── Pass 4: reconstruct freq/amp arrays ──────────────────────────────
    freqs = np.zeros((total_frames, n_partials), dtype=np.float32)
    amps  = np.zeros((total_frames, n_partials), dtype=np.float32)

    curr_fq  = np.zeros(n_partials, dtype=np.int32)
    curr_amu = np.zeros(n_partials, dtype=np.int32)
    born_pos = born_start   # byte offset into raw for born data

    for i in range(total_frames):
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
                curr_fq[slot]  += freq_reader.read_rice(k_freq)
                curr_amu[slot] += amp_reader.read_rice(k_amp)

            freqs[i, slot] = curr_fq[slot]  * freq_scale
            amps[i, slot]  = curr_amu[slot] / 255.0

        if (i + 1) % 500 == 0 or (i + 1) == total_frames:
            print(f"   ... parsed frame {i+1}/{total_frames}", end="\r")

    print()
    return metadata, freqs, amps


# ─────────────────────────────────────────────────────────────
#  Phase-Continuous Synthesis
# ─────────────────────────────────────────────────────────────
def synthesize(
    freqs:       np.ndarray,
    amps:        np.ndarray,
    frame_size:  int,
    sample_rate: int,
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

        if (i + 1) % 500 == 0 or (i + 1) == n_frames:
            print(f"   ... synthesized frame {i+1}/{n_frames}", end="\r")

    print()
    return output.astype(np.float32)


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