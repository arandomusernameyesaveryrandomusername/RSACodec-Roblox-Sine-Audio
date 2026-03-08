#!/usr/bin/env python3
"""
GenerateSineReference.py
========================
Generates a clean 440 Hz (A4) pure sine wave WAV file for upload to Roblox
as the reference sound asset used by RSADecoder.lua.

The decoder tunes each Sound object's Pitch relative to this reference:
    Pitch = target_frequency / REFERENCE_FREQUENCY (440 Hz)

So if you want a 880 Hz sine, you set Pitch = 2.0 on this Sound.

WHY THIS IS NEEDED:
    Roblox's additive synthesis approach needs a base waveform to pitch-shift.
    A pure 440 Hz sine is ideal because:
    ✓  No harmonics to corrupt the output timbre
    ✓  Pitch math is clean (880 Hz = Pitch 2.0, 220 Hz = Pitch 0.5, etc.)
    ✓  Roblox pitch-shifts linearly which is correct for frequency ratios

USAGE:
    python GenerateSineReference.py
    python GenerateSineReference.py --freq 440 --duration 2 --output sine_ref.wav
    python GenerateSineReference.py --suite    # generates a full frequency suite

OUTPUT:
    sine_440hz_reference.wav  — upload this to Roblox as an Audio asset
    Then paste the rbxassetid into RSADecoder.lua as SINE_REFERENCE_ASSET_ID

ROBLOX UPLOAD STEPS:
    1. Run this script to generate sine_440hz_reference.wav
    2. Open Roblox Studio → Asset Manager → Audio → Import
    3. Select sine_440hz_reference.wav
    4. Copy the resulting asset ID (rbxassetid://XXXXXXXXXX)
    5. Paste into RSADecoder.lua: SINE_REFERENCE_ASSET_ID = "rbxassetid://XXXXXXXXXX"
"""

import struct
import math
import argparse
import os


# ─── WAV WRITER ───────────────────────────────────────────────────────────────

def write_wav(path, samples, sample_rate=44100, bits=16):
    """Write PCM samples (floats in [-1, 1]) to a WAV file."""
    num_samples = len(samples)
    num_channels = 1
    bytes_per_sample = bits // 8
    byte_rate = sample_rate * num_channels * bytes_per_sample
    block_align = num_channels * bytes_per_sample
    data_size = num_samples * bytes_per_sample
    chunk_size = 36 + data_size

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", chunk_size))
        f.write(b"WAVE")

        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))            # chunk size
        f.write(struct.pack("<H", 1))             # PCM format
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits))

        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        if bits == 16:
            for s in samples:
                val = int(max(-1.0, min(1.0, s)) * 32767)
                f.write(struct.pack("<h", val))
        elif bits == 24:
            for s in samples:
                val = int(max(-1.0, min(1.0, s)) * 8388607)
                b = struct.pack("<i", val)
                f.write(b[:3])

    size_kb = os.path.getsize(path) / 1024
    print(f"  ✓  Wrote {path}  ({num_samples} samples, {size_kb:.1f} KB)")


# ─── GENERATORS ───────────────────────────────────────────────────────────────

def envelope(i, n, attack=0.005, release=0.02, sample_rate=44100):
    """Simple linear attack / sustain / release envelope."""
    t = i / sample_rate
    dur = n / sample_rate
    attack_end = attack
    release_start = dur - release
    if t < attack_end:
        return t / attack_end
    elif t > release_start:
        remaining = dur - t
        return max(0.0, remaining / release)
    return 1.0


def generate_sine(frequency, duration, sample_rate=44100, amplitude=0.85,
                  apply_envelope=True):
    """
    Generate a pure sine wave at `frequency` Hz for `duration` seconds.
    Returns list of float samples in [-1, 1].
    """
    n = int(sample_rate * duration)
    samples = []
    for i in range(n):
        t = i / sample_rate
        s = amplitude * math.sin(2.0 * math.pi * frequency * t)
        if apply_envelope:
            s *= envelope(i, n, sample_rate=sample_rate)
        samples.append(s)
    return samples


def generate_chord(frequencies, duration, sample_rate=44100, amplitude=0.75):
    """
    Generate a chord by summing multiple sine waves.
    Useful for testing the additive synthesis pipeline end-to-end.
    """
    n = int(sample_rate * duration)
    weight = amplitude / len(frequencies)
    samples = [0.0] * n
    for freq in frequencies:
        for i in range(n):
            t = i / sample_rate
            samples[i] += weight * math.sin(2.0 * math.pi * freq * t)
            samples[i] *= envelope(i, n, sample_rate=sample_rate)
    return samples


def generate_sweep(f_start, f_end, duration, sample_rate=44100, amplitude=0.8):
    """
    Linear frequency sweep from f_start to f_end — useful for testing
    the decoder's pitch tracking across the full frequency range.
    """
    n = int(sample_rate * duration)
    samples = []
    phase = 0.0
    for i in range(n):
        t = i / sample_rate
        freq = f_start + (f_end - f_start) * (t / duration)
        phase += 2.0 * math.pi * freq / sample_rate
        s = amplitude * math.sin(phase)
        s *= envelope(i, n, sample_rate=sample_rate)
        samples.append(s)
    return samples


def generate_harmonic_stack(fundamental, num_harmonics, duration,
                             sample_rate=44100, amplitude=0.8):
    """
    Generate a tone with natural harmonic series (1/n amplitude rolloff).
    Useful for testing the encoder → RSA → decoder fidelity visually.
    """
    n = int(sample_rate * duration)
    samples = [0.0] * n
    total_weight = sum(1.0 / k for k in range(1, num_harmonics + 1))
    for k in range(1, num_harmonics + 1):
        freq = fundamental * k
        weight = (1.0 / k) / total_weight * amplitude
        for i in range(n):
            t = i / sample_rate
            samples[i] += weight * math.sin(2.0 * math.pi * freq * t)
    for i in range(n):
        samples[i] *= envelope(i, n, sample_rate=sample_rate)
    return samples


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate sine wave reference audio for RSADecoder.lua",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python GenerateSineReference.py
  python GenerateSineReference.py --freq 440 --duration 2
  python GenerateSineReference.py --suite
        """
    )
    parser.add_argument("--freq",     "-f", type=float, default=440.0,
                        help="Sine frequency in Hz (default: 440)")
    parser.add_argument("--duration", "-d", type=float, default=1.0,
                        help="Duration in seconds (default: 1.0)")
    parser.add_argument("--sample-rate", "-r", type=int, default=44100,
                        help="Sample rate in Hz (default: 44100)")
    parser.add_argument("--output",   "-o", type=str,
                        default="sine_440hz_reference.wav",
                        help="Output WAV path (default: sine_440hz_reference.wav)")
    parser.add_argument("--bits",     "-b", type=int, default=16,
                        choices=[16, 24],
                        help="Bit depth (default: 16)")
    parser.add_argument("--suite",    action="store_true",
                        help="Generate a full test suite of WAV files")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════╗")
    print("║   RSA Sine Reference Generator           ║")
    print("╚══════════════════════════════════════════╝")
    print()

    if args.suite:
        print("  Generating test suite...")
        print()

        # 1. Primary reference — what RSADecoder.lua needs
        samples = generate_sine(440.0, 1.0, args.sample_rate, apply_envelope=False)
        write_wav("sine_440hz_reference.wav", samples, args.sample_rate, 16)
        print("    ↑ Upload THIS ONE to Roblox as the decoder reference")
        print()

        # 2. Longer loopable version (cleaner loop points)
        samples = generate_sine(440.0, 2.0, args.sample_rate, apply_envelope=False)
        write_wav("sine_440hz_2sec_loopable.wav", samples, args.sample_rate, 16)

        # 3. Common octaves for cross-checking pitch ratios
        for freq, name in [(220, "A3"), (440, "A4"), (880, "A5"), (1760, "A6")]:
            samples = generate_sine(freq, 1.0, args.sample_rate)
            write_wav(f"sine_{freq}hz_{name}.wav", samples, args.sample_rate, 16)

        # 4. Harmonic stack — encode this with rsa_encoder.py to test fidelity
        samples = generate_harmonic_stack(220, 8, 3.0, args.sample_rate)
        write_wav("harmonic_stack_A3_8partials.wav", samples, args.sample_rate, 16)
        print("    ↑ Encode this with rsa_encoder.py to test the full pipeline")
        print()

        # 5. Frequency sweep — stress test for decoder pitch tracking
        samples = generate_sweep(80, 4000, 4.0, args.sample_rate)
        write_wav("sweep_80hz_4000hz.wav", samples, args.sample_rate, 16)

        # 6. A major chord (A4, C#5, E5) — tests harmonic blending
        samples = generate_chord([440, 554.37, 659.25], 2.0, args.sample_rate)
        write_wav("chord_Amajor.wav", samples, args.sample_rate, 16)

        print()
        print("  Test suite complete!")
        print()
        print("  PIPELINE TEST:")
        print("  1. python rsa_encoder.py harmonic_stack_A3_8partials.wav test.rsa")
        print("  2. Upload sine_440hz_reference.wav to Roblox as Audio asset")
        print("  3. Paste asset ID into RSADecoder.lua → SINE_REFERENCE_ASSET_ID")
        print("  4. Upload test.rsa binary data to a StringValue in ReplicatedStorage")
        print("  5. Add RSADecoder ModuleScript + RSAPlayerGUI LocalScript")
        print("  6. Hit Play in Roblox Studio — the harmonics should match!")

    else:
        print(f"  Generating pure {args.freq} Hz sine wave...")
        print(f"  Duration     : {args.duration}s")
        print(f"  Sample rate  : {args.sample_rate} Hz")
        print(f"  Bit depth    : {args.bits}-bit")
        print()

        samples = generate_sine(args.freq, args.duration, args.sample_rate,
                                 apply_envelope=(args.freq != 440.0 or args.duration != 1.0))
        write_wav(args.output, samples, args.sample_rate, args.bits)

        print()
        print("  NEXT STEPS:")
        print(f"  1. Open Roblox Studio → Asset Manager → Audio → Import")
        print(f"  2. Select:  {args.output}")
        print(f"  3. Copy the asset ID  (rbxassetid://XXXXXXXXXX)")
        print(f"  4. Open RSADecoder.lua and set:")
        print(f'       SINE_REFERENCE_ASSET_ID = "rbxassetid://XXXXXXXXXX"')
        print(f"  5. The decoder will pitch-shift this sine to reconstruct frequencies")
        print()

    print()


if __name__ == "__main__":
    main()