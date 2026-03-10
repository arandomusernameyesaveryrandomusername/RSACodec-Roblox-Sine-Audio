"""
rsc_decoder.py — Roblox Sine Codec (RSC) Decoder
Reconstructs audio from a .rsc file via additive synthesis and writes a WAV.

Usage:
    python rsc_decoder.py --input audio.rsc --output decoded.wav
    python rsc_decoder.py --input audio.rsc --output decoded.wav --samplerate 44100
"""

import argparse
import gzip
import wave
import json
import math
import re
import struct
import numpy as np


# ─────────────────────────────────────────────
#  RSC File Reader
# ─────────────────────────────────────────────
def parse_rsc(path: str) -> tuple[dict, list[list[tuple[float, float, float]]]]:
    """
    Read an .rsc file and return (metadata_dict, frames).
    Each frame is a list of (frequency, amplitude, phase) tuples.
    """
    # Pre-compiled regex for a single partial: {f=...,a=...,p=...}
    partial_re = re.compile(
        r"\{f=([+-]?\d+(?:\.\d+)?),a=([+-]?\d+(?:\.\d+)?),p=([+-]?\d+(?:\.\d+)?)\}"
    )

    metadata: dict = {}
    frames: list[list[tuple[float, float, float]]] = []

    # Auto-detect gzip by inspecting the first two bytes (magic 0x1f 0x8b).
    # The encoder always writes gzip; this guard also accepts plain-text .rsc
    # files for backwards compatibility.
    with open(path, "rb") as fbin:
        magic = fbin.read(2)
    if magic == b"\x1f\x8b":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    for line_no, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue

        # Line 0 — Lua comment / header
        if line.startswith("--"):
            continue

        # Line 1 — JSON metadata
        if line.startswith("{") and '"magic"' in line:
            try:
                metadata = json.loads(line)
            except json.JSONDecodeError:
                print(f"  ⚠️  Could not parse metadata on line {line_no+1}")
            continue

        # Frame lines — match all partials in the line
        matches = partial_re.findall(line)
        if matches:
            partials = [
                (float(f), float(a), float(p))
                for f, a, p in matches
            ]
            frames.append(partials)

    if not metadata:
        raise ValueError("No metadata found in .rsc file — is the file valid?")
    if not frames:
        raise ValueError("No frame data found in .rsc file.")

    return metadata, frames


# ─────────────────────────────────────────────
#  Phase-Continuous Synthesis
# ─────────────────────────────────────────────
TWO_PI = 2.0 * math.pi

def synthesize_with_phase_accum(
    frames: list[list[tuple[float, float, float]]],
    frame_size: int,
    sample_rate: int,
) -> np.ndarray:
    """
    Slot-indexed, phasor-based, phase-continuous additive synthesis.

    WHY SLOT-INDEXED (not frequency-keyed):
    ─────────────────────────────────────────
    Keying phase state by frequency (a float) breaks whenever a tracked
    partial drifts even one bin (440.0 → 432.0).  The lookup misses,
    falls back to raw FFT phase, and produces a click.

    Keying by SLOT INDEX is correct because the encoder's tracker already
    guarantees: slot K in frame N+1 is the nearest-frequency continuation
    of slot K in frame N.  We never need to look up by frequency — we just
    say "slot K continues from where slot K left off", regardless of drift.

    BIRTH DETECTION:
    ─────────────────────────────────────────
    A slot that was silent last frame (amp_prev < 1e-6) is a new birth.
    We seed its phasor from the FFT-extracted phase so it starts in-phase
    with the actual signal.  Every other frame it glides continuously.
    The previous code only seeded on `first_frame` — births mid-stream
    inherited stale phasor state from the last occupant of that slot,
    causing a phase discontinuity and an audible click.

    PHASOR ROTATION (no per-sample sin calls):
    ─────────────────────────────────────────
    Instead of computing amp * sin(2π·f·k/SR + φ) for each sample k,
    we maintain a complex unit-circle phasor z and rotate it by
    e^(i·θ) each step (θ = 2π·f/SR).  np.cumprod does the whole frame
    in one vectorised pass.  sin/cos are called exactly twice per partial
    per frame regardless of frame_size.
    """
    n_frames   = len(frames)
    n_partials = max(len(f) for f in frames)
    total_len  = frame_size * n_frames
    output     = np.zeros(total_len, dtype=np.float64)

    # phasor_state[slot] = complex point on unit circle, phase at frame start
    # amp_state[slot]    = amplitude used last frame (0.0 = slot was silent)
    phasor_state = [complex(1.0, 0.0)] * n_partials
    amp_state    = [0.0]               * n_partials

    for i, partials in enumerate(frames):
        start        = i * frame_size
        frame_signal = np.zeros(frame_size, dtype=np.float64)

        for slot, (freq, amp, phase_fft) in enumerate(partials):
            if amp < 1e-6 or freq < 1e-3:
                amp_state[slot] = 0.0
                continue

            amp_prev = amp_state[slot]

            # Per-sample rotation factor — two trig calls per partial per frame
            theta = TWO_PI * freq / sample_rate
            rot   = complex(math.cos(theta), math.sin(theta))

            # BIRTH: slot was silent → seed phasor from FFT phase.
            # CONTINUATION: slot was active → glide from stored phasor.
            if amp_prev < 1e-6:
                phi0 = phase_fft
                z0   = complex(math.cos(phi0), math.sin(phi0))
            else:
                z0   = phasor_state[slot]

            # Vectorised phasor rotation over entire frame via cumprod:
            #   phasors[k] = z0 * rot^k  (unit circle walk)
            phasors      = np.empty(frame_size, dtype=np.complex128)
            phasors[0]   = z0
            phasors[1:]  = rot
            phasors      = np.cumprod(phasors)

            frame_signal += amp * phasors.imag

            # Advance one more step to get the phasor at the START of
            # the next frame, then re-normalise to prevent float drift
            z_end              = phasors[-1] * rot
            mag                = abs(z_end)
            phasor_state[slot] = z_end / mag if mag > 1e-12 else complex(1.0, 0.0)
            amp_state[slot]    = amp

        output[start : start + frame_size] = frame_signal

        if (i + 1) % 500 == 0 or (i + 1) == n_frames:
            print(f"   ... synthesized frame {i+1}/{n_frames}", end="\r")

    print()
    return output.astype(np.float32)


# ─────────────────────────────────────────────
#  WAV Writer
# ─────────────────────────────────────────────
def write_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
    """Write a float32 array as a 16-bit PCM WAV file."""
    # Clip and scale
    clipped = np.clip(samples, -1.0, 1.0)
    pcm     = (clipped * 32767.0).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    duration = len(samples) / sample_rate
    print(f"  ✅ Wrote {len(samples)} samples ({duration:.2f}s) → {path}")


# ─────────────────────────────────────────────
#  Main Decode Pipeline
# ─────────────────────────────────────────────
def decode(
    input_path: str,
    output_path: str,
    override_sample_rate: int | None,
    trim_to_original: bool,
) -> None:
    print(f"🔊 RSC Decoder  —  {input_path}")

    # 1. Parse .rsc
    metadata, frames = parse_rsc(input_path)
    print(f"   Magic          : {metadata.get('magic', '?')}")
    print(f"   Sample rate    : {metadata.get('sample_rate', '?')} Hz")
    print(f"   Frame size     : {metadata.get('frame_size', '?')} samples")
    print(f"   Partials/frame : {metadata.get('n_partials', '?')}")
    print(f"   Total frames   : {len(frames)}")

    sample_rate = override_sample_rate or metadata.get("sample_rate", 44100)
    frame_size  = metadata.get("frame_size", int(round(sample_rate / 60)))
    total_samples_orig = metadata.get("total_samples", None)

    print(f"   Using SR       : {sample_rate} Hz")

    # 2. Phase-continuous additive synthesis
    output = synthesize_with_phase_accum(
        frames=frames,
        frame_size=frame_size,
        sample_rate=sample_rate,
    )

    # 3. Trim to original length to remove padding
    if trim_to_original and total_samples_orig is not None:
        output = output[:total_samples_orig]
        print(f"   Trimmed to     : {total_samples_orig} samples")

    # 4. Peak-normalise output (optional but recommended)
    peak = np.max(np.abs(output))
    if peak > 1e-9:
        output = output / peak

    # 5. Write WAV
    write_wav(output_path, output, sample_rate)
    print("   🎉 Decoding complete!")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Roblox Sine Codec (RSC) — Decoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",      "-i", required=True,
                        help="Input .rsc file")
    parser.add_argument("--output",     "-o", default=None,
                        help="Output .wav file (defaults to input name with .wav)")
    parser.add_argument("--samplerate", "-r", type=int, default=None,
                        choices=[22050, 44100],
                        help="Override sample rate (reads from .rsc metadata by default)")
    parser.add_argument("--no-trim", action="store_true",
                        help="Do not trim output to original sample count")

    args = parser.parse_args()

    output = args.output
    if output is None:
        base = args.input
        if base.lower().endswith(".rsc"):
            base = base[:-4]
        output = base + "_decoded.wav"


    decode(
        input_path=args.input,
        output_path=output,
        override_sample_rate=args.samplerate,
        trim_to_original=not args.no_trim,
    )


if __name__ == "__main__":
    main()