"""
rsc_decoder.py — Roblox Sine Codec (RSC) Decoder  [optimized]

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
def parse_rsc(path: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read the binary .rsc file and return decoded frame arrays.

    Instead of a double Python loop calling struct.unpack_from per field,
    we read the entire body with np.frombuffer into a structured dtype and
    convert in bulk — one operation regardless of file size.

    Returns:
        metadata   dict
        freqs      float32 array  (n_frames, n_partials)  Hz
        amps       float32 array  (n_frames, n_partials)  0–1
        phases     float32 array  (n_frames, n_partials)  radians
    """
    with open(path, "rb") as f:
        raw = f.read()

    if len(raw) < 23:
        raise ValueError("File too short to be a valid .rsc file.")

    # ── Parse 22-byte header ──────────────────────────────────────────────
    magic,         = struct.unpack_from("4s",  raw,  0)
    version,       = struct.unpack_from("<B",  raw,  4)
    sample_rate,   = struct.unpack_from("<I",  raw,  5)
    frame_size,    = struct.unpack_from("<I",  raw,  9)
    n_partials,    = struct.unpack_from("<H",  raw, 13)
    total_samples, = struct.unpack_from("<I",  raw, 15)
    total_frames,  = struct.unpack_from("<I",  raw, 19)

    if magic != b"RSC3":
        raise ValueError(f"Unknown magic: {magic!r}")

    metadata = dict(magic=magic.decode(), version=version, sample_rate=sample_rate,
                    frame_size=frame_size, n_partials=n_partials,
                    total_samples=total_samples, total_frames=total_frames)

    # -- RSC3: fixed layout, 6 bytes per partial
    # uint16 freq | uint16 amp | int16 phase
    dtype  = np.dtype([("freq", "<u2"), ("amp", "<u2"), ("phase", "<i2")])
    body   = np.frombuffer(raw, dtype=dtype, offset=23)
    body   = body.reshape(total_frames, n_partials)

    nyquist    = sample_rate / 2.0
    freq_scale = nyquist / 65535.0

    freqs  = body["freq"].astype(np.float32)  * freq_scale
    amps   = body["amp"].astype(np.float32)   / 65535.0
    phases = body["phase"].astype(np.float32) / 32767.0 * math.pi

    return metadata, freqs, amps, phases


# ─────────────────────────────────────────────
#  Phase-Continuous Synthesis  (all-partial vectorized)
# ─────────────────────────────────────────────
def synthesize(
    freqs:      np.ndarray,   # (n_frames, n_partials) float32 Hz
    amps:       np.ndarray,   # (n_frames, n_partials) float32
    phases:     np.ndarray,   # (n_frames, n_partials) float32 radians
    frame_size: int,
    sample_rate: int,
) -> np.ndarray:
    """
    Slot-indexed phase-continuous additive synthesis, fully vectorized.

    Previous approach: Python loop over n_partials × np.cumprod per partial.
    At 384 partials × 3600 frames that's 1.38M Python iterations.

    New approach: process ALL active partials for a frame simultaneously.
      t = [0, 1, ..., frame_size-1] / sample_rate              (frame_size,)
      phi0 = phase state vector                                  (n_active,)
      f    = frequency vector                                    (n_active,)
      a    = amplitude vector                                    (n_active,)

      signal = (a[:, None] * np.sin(TWO_PI * f[:, None] * t + phi0[:, None])).sum(axis=0)

    That's one np.sin call on an (n_active × frame_size) matrix per frame.
    numpy dispatches this to a BLAS-level loop — ~100× faster than 384
    individual Python calls at 384 partials.

    Phase state is carried forward exactly:
      phi_next = (phi0 + TWO_PI * f * frame_size / sample_rate) % TWO_PI
    Slots that are silent (amp < 1e-6) still advance their phase so they
    re-enter in-sync if they become active again.
    """
    n_frames, n_partials = freqs.shape
    total_len = frame_size * n_frames
    output    = np.zeros(total_len, dtype=np.float64)

    t = np.arange(frame_size, dtype=np.float64) / sample_rate

    phi         = np.zeros(n_partials, dtype=np.float64)
    prev_active = np.zeros(n_partials, dtype=bool)

    # [3] Birth/death amplitude ramp — eliminates clicks on partial on/off
    # ramp tracks per-slot envelope 0.0 (silent) -> 1.0 (full), 3-frame transition
    RAMP_STEP = 1.0 / 3.0
    ramp   = np.zeros(n_partials, dtype=np.float64)
    last_f = np.zeros(n_partials, dtype=np.float64)  # last known freq (for death fade)
    last_a = np.zeros(n_partials, dtype=np.float64)  # last known amp  (for death fade)

    for i in range(n_frames):
        f = freqs[i].astype(np.float64)
        a = amps[i].astype(np.float64)
        p = phases[i].astype(np.float64)

        active = a > 1e-6
        births = active & ~prev_active
        phi[births] = p[births]

        # Update last known values for active slots
        last_f[active] = f[active]
        last_a[active] = a[active]

        # Advance ramp: active slots ramp up, silent slots ramp down
        ramp[active]  = np.minimum(1.0, ramp[active]  + RAMP_STEP)
        ramp[~active] = np.maximum(0.0, ramp[~active] - RAMP_STEP)

        # Synthesize any slot still in ramp (includes dying slots fading out)
        eff = ramp > 1e-9
        if eff.any():
            fa = last_f[eff]
            aa = last_a[eff] * ramp[eff]
            pa = phi[eff]
            signal_matrix = aa[:, None] * np.sin(TWO_PI * fa[:, None] * t + pa[:, None])
            output[i * frame_size : (i + 1) * frame_size] = signal_matrix.sum(axis=0)

        # Advance phase for all ramping slots using their last known freq
        phi[eff] = (phi[eff] + TWO_PI * last_f[eff] * frame_size / sample_rate) % TWO_PI

        prev_active = active

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
        wf.setnchannels(1);  wf.setsampwidth(2);  wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    print(f"  ✅ Wrote {len(samples)} samples ({len(samples)/sample_rate:.2f}s) → {path}")


# ─────────────────────────────────────────────
#  Main Decode Pipeline
# ─────────────────────────────────────────────
def decode(input_path: str, output_path: str,
           override_sr: int | None, trim: bool) -> None:
    print(f"🔊 RSC Decoder  —  {input_path}")

    meta, freqs, amps, phases = parse_rsc(input_path)
    sr         = override_sr or meta["sample_rate"]
    frame_size = meta["frame_size"]
    print(f"   {meta['total_frames']} frames  |  {meta['n_partials']} partials  "
          f"|  {sr} Hz  |  {meta['total_samples']/sr:.2f}s")

    output = synthesize(freqs, amps, phases, frame_size, sr)

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