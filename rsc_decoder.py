"""
rsc_decoder.py — Roblox Sine Codec (RSC) Decoder  [optimized]

Format: RSC4
  Header : 23 bytes
  Partial: 4 bytes   uint16 freq | uint16 amp
           (phase not stored — accumulated continuously per slot)

Usage:
    python rsc_decoder.py --input audio.rsc --output decoded.wav
"""

import argparse
import math
import struct
import wave

import numpy as np

TWO_PI = 2.0 * math.pi


# ─────────────────────────────────────────────
#  RSC Binary Reader  (fully vectorized)
# ─────────────────────────────────────────────
def parse_rsc(path: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Read the binary RSC4 file and return decoded frame arrays.

    RSC4 layout per partial: uint16 freq | uint16 amp  (4 bytes, no phase)
    Bulk-decoded via np.frombuffer structured dtype — one operation
    regardless of file size.

    Returns:
        metadata   dict
        freqs      float32 array  (n_frames, n_partials)  Hz
        amps       float32 array  (n_frames, n_partials)  0–1
    """
    with open(path, "rb") as f:
        raw = f.read()

    if len(raw) < 23:
        raise ValueError("File too short to be a valid .rsc file.")

    magic,         = struct.unpack_from("4s",  raw,  0)
    version,       = struct.unpack_from("<B",  raw,  4)
    sample_rate,   = struct.unpack_from("<I",  raw,  5)
    frame_size,    = struct.unpack_from("<I",  raw,  9)
    n_partials,    = struct.unpack_from("<H",  raw, 13)
    total_samples, = struct.unpack_from("<I",  raw, 15)
    total_frames,  = struct.unpack_from("<I",  raw, 19)

    if magic != b"RSC4":
        raise ValueError(f"Unknown magic: {magic!r}  (expected RSC4)")

    metadata = dict(magic=magic.decode(), version=version, sample_rate=sample_rate,
                    frame_size=frame_size, n_partials=n_partials,
                    total_samples=total_samples, total_frames=total_frames)

    # RSC4: 4 bytes per partial — uint16 freq | uint16 amp
    dtype = np.dtype([("freq", "<u2"), ("amp", "<u2")])
    body  = np.frombuffer(raw, dtype=dtype, offset=23)
    body  = body.reshape(total_frames, n_partials)

    nyquist    = sample_rate / 2.0
    freq_scale = nyquist / 65535.0

    freqs = body["freq"].astype(np.float32) * freq_scale
    amps  = body["amp"].astype(np.float32)  / 65535.0

    return metadata, freqs, amps


# ─────────────────────────────────────────────
#  Phase-Continuous Synthesis  (all-partial vectorized)
# ─────────────────────────────────────────────
def synthesize(
    freqs:       np.ndarray,   # (n_frames, n_partials) float32 Hz
    amps:        np.ndarray,   # (n_frames, n_partials) float32
    frame_size:  int,
    sample_rate: int,
) -> np.ndarray:
    """
    Slot-indexed phase-continuous additive synthesis, fully vectorized.

    Phase is never read from the file — it is accumulated per slot:
        phi_next = (phi + TWO_PI * f * frame_size / sample_rate) % TWO_PI

    This matches the Roblox decoder exactly: each looping sine Sound object
    maintains its own continuous phase; we only control pitch and volume.
    Slots that go silent keep advancing phase so they re-enter in-sync.

    One np.sin call on an (n_active × frame_size) matrix per frame —
    numpy dispatches this to a BLAS-level loop, ~100× faster than
    individual per-partial Python calls.
    """
    n_frames, n_partials = freqs.shape
    total_len = frame_size * n_frames
    output    = np.zeros(total_len, dtype=np.float64)

    t   = np.arange(frame_size, dtype=np.float64) / sample_rate
    phi = np.zeros(n_partials, dtype=np.float64)   # per-slot phase accumulator

    for i in range(n_frames):
        f = freqs[i].astype(np.float64)
        a = amps[i].astype(np.float64)

        active = a > 1e-6

        if active.any():
            fa = f[active]
            aa = a[active]
            pa = phi[active]
            output[i * frame_size : (i + 1) * frame_size] = (
                aa[:, None] * np.sin(TWO_PI * fa[:, None] * t + pa[:, None])
            ).sum(axis=0)

        # Advance phase for ALL slots (active or not) to stay continuous
        phi = (phi + TWO_PI * f * frame_size / sample_rate) % TWO_PI

        if (i + 1) % 500 == 0 or (i + 1) == n_frames:
            print(f"   ... synthesized frame {i+1}/{n_frames}", end="\r")

    print()
    return output.astype(np.float32)


# ─────────────────────────────────────────────
#  WAV Writer
# ─────────────────────────────────────────────
def write_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    print(f"  ✅ Wrote {len(samples)} samples ({len(samples)/sample_rate:.2f}s) → {path}")


# ─────────────────────────────────────────────
#  Main Decode Pipeline
# ─────────────────────────────────────────────
def decode(input_path: str, output_path: str,
           override_sr: int | None, trim: bool) -> None:
    print(f"🔊 RSC Decoder  —  {input_path}")

    meta, freqs, amps = parse_rsc(input_path)
    sr         = override_sr or meta["sample_rate"]
    frame_size = meta["frame_size"]
    print(f"   {meta['total_frames']} frames  |  {meta['n_partials']} partials  "
          f"|  {sr} Hz  |  {meta['total_samples']/sr:.2f}s")

    output = synthesize(freqs, amps, frame_size, sr)

    if trim and meta.get("total_samples"):
        output = output[:meta["total_samples"]]

    peak = np.max(np.abs(output))
    if peak > 1e-9: output /= peak

    write_wav(output_path, output, sr)
    print("   🎉 Decoding complete!")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="RSC Decoder",
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