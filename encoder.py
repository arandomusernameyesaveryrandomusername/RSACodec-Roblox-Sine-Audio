"""
RSA Encoder - Roblox Sine Audio Format
=======================================
Encodes audio into additive sine wave components readable by Roblox Lua.

Format: .rsa (Roblox Sine Audio)
- Header: "RSA1" magic bytes
- Metadata: sample_rate, duration, num_frames, frame_size, max_harmonics
- Frames: Each frame contains N sine wave components (frequency, amplitude, phase)

The decoder in Roblox Lua reconstructs audio by summing sine waves per frame.

Usage:
    python rsa_encoder.py input.wav output.rsa
    python rsa_encoder.py input.wav output.rsa --harmonics 16 --frame-size 512
    python rsa_encoder.py --demo output.rsa   (generates a demo tone)
"""

import struct
import argparse
import math
import sys
import os


# ─── RSA FORMAT CONSTANTS ────────────────────────────────────────────────────

MAGIC           = b"RSA1"          # 4-byte magic
FORMAT_VERSION  = 1                # Format version
MAX_HARMONICS   = 32               # Hard cap per frame
DEFAULT_HARMONICS = 16
DEFAULT_FRAME_SIZE = 512           # Samples per analysis frame
DEFAULT_SAMPLE_RATE = 44100


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def log(msg): print(f"  {msg}")


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


# ─── FFT (Pure Python, no numpy required) ────────────────────────────────────

def fft(x):
    """Cooley-Tukey FFT – x must have power-of-2 length."""
    N = len(x)
    if N <= 1:
        return x
    if N % 2 != 0:
        raise ValueError("FFT length must be a power of 2")
    even = fft(x[0::2])
    odd  = fft(x[1::2])
    T = [complex(math.cos(-2 * math.pi * k / N),
                 math.sin(-2 * math.pi * k / N)) * odd[k % (N // 2)]
         for k in range(N // 2)]
    return [even[k] + T[k]       for k in range(N // 2)] + \
           [even[k] - T[k]       for k in range(N // 2)]


def next_power_of_2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def hann_window(n):
    """Hann window coefficients."""
    return [0.5 * (1 - math.cos(2 * math.pi * i / (n - 1))) for i in range(n)]


# ─── ANALYSIS ────────────────────────────────────────────────────────────────

def analyse_frame(samples, sample_rate, max_harmonics):
    """
    Analyse one frame of PCM samples via FFT.
    Returns list of (frequency_hz, amplitude_0_to_1, phase_radians)
    for the top `max_harmonics` partials by amplitude.
    """
    n = len(samples)
    padded_n = next_power_of_2(n)

    # Apply Hann window
    window = hann_window(n)
    windowed = [samples[i] * window[i] for i in range(n)]

    # Zero-pad to next power of 2
    windowed += [0.0] * (padded_n - n)

    # FFT
    spectrum = fft([complex(s, 0) for s in windowed])

    # Only use first half (positive frequencies)
    half = padded_n // 2
    freq_resolution = sample_rate / padded_n

    bins = []
    for k in range(1, half):          # skip DC (k=0)
        freq = k * freq_resolution
        if freq > 20000:              # above human hearing, skip
            break
        amp  = abs(spectrum[k]) / (padded_n / 2)   # normalise
        phase = math.atan2(spectrum[k].imag, spectrum[k].real)
        bins.append((amp, freq, phase))

    # Sort by amplitude descending, take top N
    bins.sort(reverse=True)
    top = bins[:max_harmonics]

    # Normalise amplitudes relative to strongest partial
    if top:
        max_amp = top[0][0] if top[0][0] > 0 else 1.0
        result = [(freq, clamp(amp / max_amp, 0.0, 1.0), phase)
                  for amp, freq, phase in top]
    else:
        result = []

    return result


# ─── WAV READER (no external libs) ───────────────────────────────────────────

def read_wav(path):
    """
    Read a WAV file. Returns (sample_rate, samples_list_of_float).
    Supports 8-bit, 16-bit, 24-bit, 32-bit PCM and 32-bit float.
    Mixes down to mono.
    """
    with open(path, "rb") as f:
        data = f.read()

    def u16(offset): return struct.unpack_from("<H", data, offset)[0]
    def u32(offset): return struct.unpack_from("<I", data, offset)[0]
    def i16(offset): return struct.unpack_from("<h", data, offset)[0]

    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Not a valid WAV file")

    # Walk chunks
    pos = 12
    fmt_chunk = None
    data_chunk = None

    while pos < len(data) - 8:
        chunk_id   = data[pos:pos+4]
        chunk_size = u32(pos + 4)
        chunk_data = data[pos+8 : pos+8+chunk_size]
        if chunk_id == b"fmt ":
            fmt_chunk = chunk_data
        elif chunk_id == b"data":
            data_chunk = chunk_data
        pos += 8 + chunk_size
        if chunk_size % 2: pos += 1   # padding byte

    if not fmt_chunk or not data_chunk:
        raise ValueError("WAV missing fmt or data chunk")

    audio_fmt   = struct.unpack_from("<H", fmt_chunk, 0)[0]
    num_channels= struct.unpack_from("<H", fmt_chunk, 2)[0]
    sample_rate = struct.unpack_from("<I", fmt_chunk, 4)[0]
    bits        = struct.unpack_from("<H", fmt_chunk, 14)[0]

    FLOAT_FMT = 3

    log(f"WAV: {sample_rate} Hz, {num_channels}ch, {bits}-bit, "
        f"{'float' if audio_fmt==FLOAT_FMT else 'PCM'}")

    # Decode samples
    samples_raw = []
    bytes_per_sample = bits // 8
    total_samples = len(data_chunk) // (bytes_per_sample * num_channels)

    for i in range(total_samples):
        frame_vals = []
        base = i * bytes_per_sample * num_channels
        for c in range(num_channels):
            off = base + c * bytes_per_sample
            raw = data_chunk[off : off + bytes_per_sample]
            if audio_fmt == FLOAT_FMT and bits == 32:
                val = struct.unpack_from("<f", raw)[0]
            elif bits == 8:
                val = (struct.unpack_from("B", raw)[0] - 128) / 128.0
            elif bits == 16:
                val = struct.unpack_from("<h", raw)[0] / 32768.0
            elif bits == 24:
                b0, b1, b2 = raw[0], raw[1], raw[2]
                v = b0 | (b1 << 8) | (b2 << 16)
                if v & 0x800000: v -= 0x1000000
                val = v / 8388608.0
            elif bits == 32:
                val = struct.unpack_from("<i", raw)[0] / 2147483648.0
            else:
                val = 0.0
            frame_vals.append(val)
        # Mix to mono
        samples_raw.append(sum(frame_vals) / num_channels)

    return sample_rate, samples_raw


# ─── DEMO TONE GENERATOR ─────────────────────────────────────────────────────

def generate_demo_tone(sample_rate=44100, duration=2.0):
    """
    Generate a synthesised test tone: A4 (440 Hz) with harmonics.
    Returns list of float samples.
    """
    log("Generating demo tone: A4 (440 Hz) with 6 harmonics, 2 seconds")
    n = int(sample_rate * duration)
    samples = []
    harmonics = [
        (440.0,  1.00),   # fundamental
        (880.0,  0.50),   # 2nd
        (1320.0, 0.33),   # 3rd
        (1760.0, 0.25),   # 4th
        (2200.0, 0.15),   # 5th
        (2640.0, 0.10),   # 6th
    ]
    for i in range(n):
        t = i / sample_rate
        # Simple envelope: attack 0.05s, decay to sustain, release 0.2s
        env = 1.0
        if t < 0.05:
            env = t / 0.05
        elif t > duration - 0.2:
            env = (duration - t) / 0.2
        s = sum(amp * math.sin(2 * math.pi * freq * t)
                for freq, amp in harmonics)
        # Normalise peak
        s *= (0.8 / sum(amp for _, amp in harmonics))
        samples.append(s * env)
    return sample_rate, samples


# ─── RSA WRITER ──────────────────────────────────────────────────────────────

def write_rsa(path, sample_rate, duration, frames_data, frame_size, max_harmonics):
    """
    Write an .rsa binary file.

    File layout:
    ┌─────────────────────────────────────────────────────┐
    │  HEADER  (20 bytes)                                 │
    │  magic[4]  "RSA1"                                   │
    │  version[1] u8                                      │
    │  sample_rate[4] u32                                 │
    │  duration_ms[4] u32  (milliseconds)                 │
    │  num_frames[4] u32                                  │
    │  frame_size[4] u32   (samples per frame)            │
    │  max_harmonics[1] u8                                │
    ├─────────────────────────────────────────────────────┤
    │  FRAMES  (num_frames × entries)                     │
    │  Each frame:                                        │
    │    harmonic_count[1] u8                             │
    │    For each harmonic:                               │
    │      freq[4]  f32  (Hz)                             │
    │      amp[2]   u16  (0-65535 mapped from 0.0-1.0)   │
    │      phase[2] i16  (mapped from -π to π)           │
    └─────────────────────────────────────────────────────┘
    """
    num_frames  = len(frames_data)
    duration_ms = int(duration * 1000)

    with open(path, "wb") as f:
        # ── Header ────────────────────────────────────────
        f.write(MAGIC)
        f.write(struct.pack("B", FORMAT_VERSION))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", duration_ms))
        f.write(struct.pack("<I", num_frames))
        f.write(struct.pack("<I", frame_size))
        f.write(struct.pack("B", max_harmonics))

        # ── Frames ────────────────────────────────────────
        for partials in frames_data:
            count = min(len(partials), max_harmonics)
            f.write(struct.pack("B", count))
            for i in range(count):
                freq, amp, phase = partials[i]
                amp_u16   = int(clamp(amp, 0.0, 1.0) * 65535)
                phase_i16 = int(clamp(phase / math.pi, -1.0, 1.0) * 32767)
                f.write(struct.pack("<f", freq))
                f.write(struct.pack("<H", amp_u16))
                f.write(struct.pack("<h", phase_i16))

    size_kb = os.path.getsize(path) / 1024
    log(f"Wrote {num_frames} frames → {path}  ({size_kb:.1f} KB)")


# ─── MAIN ENCODE PIPELINE ────────────────────────────────────────────────────

def encode(input_path, output_path, max_harmonics, frame_size, verbose):
    print()
    print("╔══════════════════════════════════════╗")
    print("║      RSA Encoder  v1.0               ║")
    print("║  Roblox Sine Audio Format            ║")
    print("╚══════════════════════════════════════╝")
    print()

    # Load audio
    if input_path == "--demo":
        log("Mode: DEMO (synthesised test tone)")
        sample_rate, samples = generate_demo_tone()
    else:
        log(f"Reading: {input_path}")
        sample_rate, samples = read_wav(input_path)

    duration   = len(samples) / sample_rate
    num_frames = math.ceil(len(samples) / frame_size)

    log(f"Duration : {duration:.3f}s  ({len(samples)} samples)")
    log(f"Frames   : {num_frames}  (frame size: {frame_size} samples)")
    log(f"Harmonics: up to {max_harmonics} per frame")
    print()

    # Analyse frames
    frames_data = []
    hop = frame_size

    for i in range(num_frames):
        start = i * hop
        end   = min(start + frame_size, len(samples))
        frame_samples = samples[start:end]

        # Pad last frame if needed
        if len(frame_samples) < frame_size:
            frame_samples += [0.0] * (frame_size - len(frame_samples))

        partials = analyse_frame(frame_samples, sample_rate, max_harmonics)
        frames_data.append(partials)

        if verbose or (i % max(1, num_frames // 20) == 0):
            pct = (i + 1) / num_frames * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"\r  Analysing [{bar}] {pct:5.1f}%  frame {i+1}/{num_frames}", end="", flush=True)

    print()
    print()

    # Write .rsa file
    write_rsa(output_path, sample_rate, duration, frames_data, frame_size, max_harmonics)

    # ── Summary ───────────────────────────────────────────
    total_partials = sum(len(f) for f in frames_data)
    avg_partials   = total_partials / max(1, num_frames)
    print()
    print("  ✓ Encoding complete!")
    print(f"  Average partials/frame : {avg_partials:.1f}")
    print(f"  Output                 : {output_path}")
    print()
    print("  Next step: use rsa_decoder.lua in Roblox Studio")
    print()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RSA Encoder — Convert WAV audio to Roblox Sine Audio format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rsa_encoder.py song.wav song.rsa
  python rsa_encoder.py song.wav song.rsa --harmonics 24 --frame-size 1024
  python rsa_encoder.py --demo demo_tone.rsa
        """
    )
    parser.add_argument("input",  help="Input WAV file path, or '--demo' for a test tone")
    parser.add_argument("output", help="Output .rsa file path")
    parser.add_argument("--harmonics",  "-n", type=int, default=DEFAULT_HARMONICS,
                        metavar="N",
                        help=f"Max sine harmonics per frame (default: {DEFAULT_HARMONICS}, max: {MAX_HARMONICS})")
    parser.add_argument("--frame-size", "-f", type=int, default=DEFAULT_FRAME_SIZE,
                        metavar="S",
                        help=f"FFT frame size in samples (default: {DEFAULT_FRAME_SIZE})")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print info for every frame")

    args = parser.parse_args()

    if args.harmonics > MAX_HARMONICS:
        print(f"Warning: harmonics capped at {MAX_HARMONICS}")
        args.harmonics = MAX_HARMONICS

    if args.frame_size < 64 or (args.frame_size & (args.frame_size - 1)) != 0:
        print("Error: --frame-size must be a power of 2 and ≥ 64")
        sys.exit(1)

    encode(
        input_path   = args.input,
        output_path  = args.output,
        max_harmonics= args.harmonics,
        frame_size   = args.frame_size,
        verbose      = args.verbose,
    )


if __name__ == "__main__":
    main()