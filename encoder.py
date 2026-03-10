"""
rsc_encoder.py — Roblox Sine Codec (RSC) Encoder
Decomposes a WAV file into additive sine components per frame and writes
a Lua-parsable .rsc file.

Usage:
    python rsc_encoder.py --input audio.wav --output audio.rsc
    python rsc_encoder.py --input audio.wav --output audio.rsc --partials 24 --samplerate 44100
"""

import argparse
import gzip
import wave
import struct
import math
import json
import numpy as np
from scipy.signal import find_peaks


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
TARGET_FPS        = 60
DEFAULT_PARTIALS  = 64
DEFAULT_SAMPLERATE = 44100
RSC_MAGIC         = "RSC1"          # File header magic string
RSC_EXTENSION     = ".rsc"


# ─────────────────────────────────────────────
#  WAV Loading
# ─────────────────────────────────────────────
def load_wav(path: str) -> tuple[np.ndarray, int]:
    """
    Load a WAV file and return (mono_float32_samples, sample_rate).
    Stereo is mixed down to mono. Samples are normalised to [-1, 1].
    """
    with wave.open(path, "rb") as wf:
        n_channels   = wf.getnchannels()
        sampwidth    = wf.getsampwidth()
        sample_rate  = wf.getframerate()
        n_frames     = wf.getnframes()
        raw          = wf.readframes(n_frames)

    # Determine dtype from sample width
    if sampwidth == 1:
        dtype  = np.uint8
        scale  = 128.0
        offset = -128.0 / 128.0   # unsigned 8-bit centres on 128
    elif sampwidth == 2:
        dtype  = np.int16
        scale  = 32768.0
        offset = 0.0
    elif sampwidth == 3:
        # 24-bit: manually unpack to int32
        raw_array = np.frombuffer(raw, dtype=np.uint8)
        n_samples = len(raw_array) // 3
        raw_int32 = np.zeros(n_samples, dtype=np.int32)
        raw_int32  = (raw_array[0::3].astype(np.int32) |
                      (raw_array[1::3].astype(np.int32) << 8) |
                      (raw_array[2::3].astype(np.int32) << 16))
        # Sign-extend from 24-bit
        raw_int32[raw_int32 >= 0x800000] -= 0x1000000
        samples = raw_int32.astype(np.float32) / 8388608.0
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)
        return samples, sample_rate
    elif sampwidth == 4:
        dtype  = np.int32
        scale  = 2147483648.0
        offset = 0.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if sampwidth == 1:
        samples = (samples - 128.0) / 128.0
    else:
        samples = samples / scale + offset

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples, sample_rate


# ─────────────────────────────────────────────
#  Normalisation
# ─────────────────────────────────────────────
def normalize(samples: np.ndarray) -> np.ndarray:
    """Peak-normalise audio to [-1, 1]."""
    peak = np.max(np.abs(samples))
    if peak < 1e-9:
        return samples
    return samples / peak


# ─────────────────────────────────────────────
#  Wide-Window FFT Analysis + Peak Tracking
# ─────────────────────────────────────────────

# Analysis window is intentionally much larger than the hop size.
# hop  ≈ 735 samples = 16.7 ms  → 60 FPS output rate
# win  = 4096 samples = 92.9 ms → ~10.8 Hz frequency resolution
# This gives stable, well-resolved peaks while still updating at 60 FPS.
ANALYSIS_WIN = 4096

def _fft_candidates(
    audio: np.ndarray,
    center: int,
    analysis_win: int,
    sample_rate: int,
    n_candidates: int,
) -> list[tuple[float, float, float]]:
    """
    Extract FFT peak candidates from a large window centred on `center`.
    Returns up to n_candidates (freq, amp, phase) tuples sorted by amplitude.
    """
    half = analysis_win // 2
    start = center - half
    end   = center + half

    # Slice with zero-padding if we're near the signal boundaries
    if start < 0 or end > len(audio):
        chunk = np.zeros(analysis_win, dtype=np.float64)
        src_s = max(0, start)
        src_e = min(len(audio), end)
        dst_s = src_s - start
        dst_e = dst_s + (src_e - src_s)
        chunk[dst_s:dst_e] = audio[src_s:src_e]
    else:
        chunk = audio[start:end].astype(np.float64)

    window     = np.hanning(analysis_win)
    windowed   = chunk * window
    spectrum   = np.fft.rfft(windowed)
    freqs      = np.fft.rfftfreq(analysis_win, d=1.0 / sample_rate)
    magnitudes = np.abs(spectrum) * (2.0 / np.sum(window))
    phases     = np.angle(spectrum)

    bin_width    = sample_rate / analysis_win       # ≈10.8 Hz
    min_distance = max(1, int(20.0 / bin_width))
    peak_indices, _ = find_peaks(magnitudes, distance=min_distance, height=1e-6)

    if len(peak_indices) == 0:
        peak_indices = np.argsort(magnitudes)[::-1][:n_candidates]
    else:
        order        = np.argsort(magnitudes[peak_indices])[::-1]
        peak_indices = peak_indices[order[:n_candidates]]

    candidates = []
    for idx in peak_indices:
        f = float(freqs[idx])
        if f < 20.0 or f > sample_rate / 2.0 - bin_width:
            continue
        a = min(1.0, max(0.0, float(magnitudes[idx])))
        p = float(phases[idx])
        candidates.append((f, a, p))

    return candidates


# ─────────────────────────────────────────────
#  Greedy Peak Tracker
# ─────────────────────────────────────────────

def _track_greedy(
    candidates: list[tuple[float, float, float]],
    slots: list[tuple[float, float, float] | None],
    n_partials: int,
    tolerance_hz: float = 50.0,
) -> list[tuple[float, float, float]]:
    """
    Greedy nearest-neighbour slot-stable peak tracking.

    SLOT STABILITY IS CRITICAL FOR THE DECODER:
    The decoder indexes its phasor oscillators by slot position.
    Slot 0 in frame N+1 must be the same physical partial as slot 0
    in frame N — otherwise the phase accumulator jumps to a wrong
    value and produces a click.

    Algorithm:
      For each occupied slot (sorted by amplitude, loudest first so
      strong partials get priority in contested matches):
        - Find the closest unmatched candidate within tolerance_hz.
        - If found  → fill same slot with that candidate (continuation).
        - If not found → slot goes silent (0,0,0) this frame.
      Leftover candidates are new births; assign them into empty/silent
      slots, loudest first.
      Any still-empty slots are padded with (0,0,0).
    """
    remaining   = list(enumerate(candidates))   # (original_idx, (f,a,p))
    result      = [(0.0, 0.0, 0.0)] * n_partials

    # Sort occupied slots by amplitude descending so loudest get first pick
    occupied = sorted(
        [(s, slots[s]) for s in range(n_partials) if slots[s] is not None],
        key=lambda x: x[1][1], reverse=True
    )

    for slot_idx, (pf, pa, pp) in occupied:
        if not remaining:
            break
        dists  = [abs(cand[1][0] - pf) for cand in remaining]
        best_i = int(np.argmin(dists))
        if dists[best_i] <= tolerance_hz:
            _, (cf, ca, cp) = remaining.pop(best_i)
            result[slot_idx] = (round(cf, 2), round(ca, 3), round(cp, 4))
        # else: slot stays (0,0,0) — partial died

    # Assign births into empty slots, loudest candidate first
    remaining.sort(key=lambda x: x[1][1], reverse=True)
    empty_slots = [s for s in range(n_partials) if result[s] == (0.0, 0.0, 0.0)]
    for slot_idx, (_, (cf, ca, cp)) in zip(empty_slots, remaining):
        if ca > 1e-6 and cf > 1e-3:
            result[slot_idx] = (round(cf, 2), round(ca, 3), round(cp, 4))

    return result


def analyse_frame(
    audio: np.ndarray,
    center: int,
    sample_rate: int,
    n_partials: int,
    prev_partials: list[tuple[float, float, float]],
    analysis_win: int = ANALYSIS_WIN,
) -> list[tuple[float, float, float]]:
    """
    Analyse one hop position and return slot-stable tracked partials.
    """
    candidates = _fft_candidates(
        audio, center, analysis_win, sample_rate, n_candidates=n_partials * 4
    )
    # Convert prev_partials to slot state (None for silent slots)
    slots = [p if p[1] > 1e-6 else None for p in prev_partials]
    return _track_greedy(candidates, slots, n_partials)


# ─────────────────────────────────────────────
#  RSC File Writer (Lua-parsable format)
# ─────────────────────────────────────────────
def write_rsc(
    path: str,
    frames: list[list[tuple[float, float, float]]],
    sample_rate: int,
    frame_size: int,
    n_partials: int,
    total_samples: int,
) -> None:
    """
    Write encoded frames to a .rsc file.

    Format:
        Line 0: header comment
        Line 1: metadata JSON
        Line 2+: one frame per line as a Lua-table string

        Each frame line:
            {{f=440.0,a=0.800,p=0.0000},{f=880.0,a=0.400,p=1.5708},...}
    """
    metadata = {
        "magic":         RSC_MAGIC,
        "version":       1,
        "sample_rate":   sample_rate,
        "frame_size":    frame_size,
        "n_partials":    n_partials,
        "total_samples": total_samples,
        "total_frames":  len(frames),
        "fps_target":    TARGET_FPS,
    }

    # Build the full text payload in memory, then gzip-compress it.
    # The decoder auto-detects gzip via the 0x1f 0x8b magic bytes so
    # the file extension stays .rsc — no format change needed on decode.
    lines: list[str] = []
    lines.append(
        f"-- Roblox Sine Codec (RSC) v1  |  {sample_rate} Hz  |  "
        f"{n_partials} partials/frame  |  {len(frames)} frames"
    )
    lines.append(json.dumps(metadata))
    for frame in frames:
        parts = [f"{{f={freq:.2f},a={amp:.3f},p={phase:.4f}}}"
                 for (freq, amp, phase) in frame]
        lines.append("{" + ",".join(parts) + "}")

    raw_text    = "\n".join(lines) + "\n"
    compressed  = gzip.compress(raw_text.encode("utf-8"), compresslevel=9)

    with open(path, "wb") as f:
        f.write(compressed)

    ratio = len(compressed) / len(raw_text) * 100
    print(f"  ✅ Wrote {len(frames)} frames → {path}  "
          f"({len(compressed)//1024} KB compressed, {ratio:.0f}% of raw)")


# ─────────────────────────────────────────────
#  Main Encode Pipeline
# ─────────────────────────────────────────────
def encode(
    input_path: str,
    output_path: str,
    n_partials: int,
    target_sample_rate: int,
) -> None:
    print(f"🎵 RSC Encoder  —  {input_path}")
    print(f"   Partials/frame : {n_partials}")
    print(f"   Target SR      : {target_sample_rate} Hz")

    # 1. Load WAV
    samples, native_sr = load_wav(input_path)
    print(f"   Native SR      : {native_sr} Hz  |  {len(samples)} samples  "
          f"({len(samples)/native_sr:.2f}s)")

    # 2. Resample if needed (simple linear interp — scipy not required)
    if native_sr != target_sample_rate:
        print(f"   Resampling {native_sr} → {target_sample_rate} Hz …")
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(target_sample_rate, native_sr)
        up, down = target_sample_rate // g, native_sr // g
        samples = resample_poly(samples, up, down).astype(np.float32)
        sample_rate = target_sample_rate
    else:
        sample_rate = native_sr

    # 3. Normalise
    samples = normalize(samples)
    total_samples = len(samples)

    # 4. Compute frame size for ~60 FPS
    frame_size = int(round(sample_rate / TARGET_FPS))   # ≈735 @ 44100
    print(f"   Frame size     : {frame_size} samples  ({1000*frame_size/sample_rate:.2f} ms)")

    # 5. Slice into frames & analyse
    n_frames = math.ceil(total_samples / frame_size)
    print(f"   Total frames   : {n_frames}")

    # Pad samples so last frame is full
    pad_len = n_frames * frame_size - total_samples
    if pad_len > 0:
        samples = np.concatenate([samples, np.zeros(pad_len, dtype=np.float32)])

    print(f"   Analysis win   : {ANALYSIS_WIN} samples  "
          f"({1000*ANALYSIS_WIN/sample_rate:.1f} ms, "
          f"{sample_rate/ANALYSIS_WIN:.1f} Hz/bin)")

    all_frames: list[list[tuple[float, float, float]]] = []
    prev_partials: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * n_partials

    for i in range(n_frames):
        # Centre of this hop in the full audio array
        center = i * frame_size + frame_size // 2

        partials = analyse_frame(
            audio=samples,
            center=center,
            sample_rate=sample_rate,
            n_partials=n_partials,
            prev_partials=prev_partials,
        )
        all_frames.append(partials)
        prev_partials = partials

        if (i + 1) % 500 == 0 or (i + 1) == n_frames:
            print(f"   … encoded frame {i+1}/{n_frames}", end="\r")

    print()

    # 6. Write .rsc
    write_rsc(
        path=output_path,
        frames=all_frames,
        sample_rate=sample_rate,
        frame_size=frame_size,
        n_partials=n_partials,
        total_samples=total_samples,
    )

    # 7. Stats
    rsc_size_kb = sum(
        len("{" + ",".join(f"{{f={f:.2f},a={a:.3f},p={p:.4f}}}" for f,a,p in fr) + "}\n")
        for fr in all_frames
    ) / 1024
    print(f"   📦 Approx data  : {rsc_size_kb:.1f} KB  "
          f"({rsc_size_kb/1024:.2f} MB)")
    print("   🎉 Encoding complete!")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Roblox Sine Codec (RSC) — Encoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",      "-i", required=True,
                        help="Input .wav file")
    parser.add_argument("--output",     "-o", default=None,
                        help="Output .rsc file (defaults to input name with .rsc)")
    parser.add_argument("--partials",   "-n", type=int, default=DEFAULT_PARTIALS,
                        help="Number of sine partials per frame (8–64 recommended)")
    parser.add_argument("--samplerate", "-r", type=int, default=DEFAULT_SAMPLERATE,
                        choices=[22050, 44100],
                        help="Target sample rate (22050 or 44100)")

    args = parser.parse_args()

    output = args.output
    if output is None:
        base = args.input
        if base.lower().endswith(".wav"):
            base = base[:-4]
        output = base + RSC_EXTENSION

    encode(
        input_path=args.input,
        output_path=output,
        n_partials=args.partials,
        target_sample_rate=args.samplerate,
    )


if __name__ == "__main__":
    main()