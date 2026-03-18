#!/usr/bin/env python3
"""
Barebones RSC6 encoder — keeps same bitstream, drops tracking / peak logic
Only uses DPSS window + ERB-weighted sorting → take top N partials each frame
"""

import argparse
import math
import struct
import numpy as np
from scipy.signal import windows
import librosa

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
TARGET_FPS       = 60
DEFAULT_PARTIALS = 384
DEFAULT_SR       = 44100
RSC_EXTENSION    = ".rsc"
ANALYSIS_WIN     = 4096
MU               = 255.0
_LOG1P_MU        = math.log1p(MU)
ALIVE_THRESHOLD  = 0     # you can raise to 1–3 if you want to kill very quiet partials

# ─────────────────────────────────────────────────────────────
# Mu-law compression
# ─────────────────────────────────────────────────────────────
def mulaw_encode(x: np.ndarray) -> np.ndarray:
    """float32 [0,1] → uint8 [0,255]"""
    x = np.clip(x.astype(np.float64), 0.0, 1.0)
    return np.clip(np.round(MU * np.log1p(MU * x) / _LOG1P_MU), 0, 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────
# Rice coding helpers
# ─────────────────────────────────────────────────────────────
def zigzag(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.int32)
    return ((a << 1) ^ (a >> 31)).astype(np.uint32)

def optimal_rice_k(vals: np.ndarray) -> int:
    if len(vals) == 0:
        return 0
    v = vals.astype(np.int64)
    best_k, best_bits = 0, float("inf")
    for k in range(16):
        bits = int(np.sum(v >> k)) + len(v) * (1 + k)
        if bits < best_bits:
            best_bits, best_k = bits, k
    return best_k

def rice_encode(vals: np.ndarray, k: int) -> bytearray:
    if len(vals) == 0:
        return bytearray()
    v = vals.astype(np.int64)
    q = v >> k
    r = v & ((1 << k) - 1)
    code_lens = q + 1 + k
    total_bits = int(code_lens.sum())
    bits = np.zeros(total_bits, dtype=np.uint8)
    starts = np.cumsum(np.concatenate(([0], code_lens[:-1]))).astype(np.int64)
    bits[starts + q] = 1
    for bit_idx in range(k):
        shift = k - 1 - bit_idx
        pos = starts + q + 1 + bit_idx
        bits[pos.astype(int)] = ((r >> shift) & 1).astype(np.uint8)
    pad = (-total_bits) % 8
    if pad:
        bits = np.append(bits, np.zeros(pad, dtype=np.uint8))
    return bytearray(np.packbits(bits, bitorder="big"))

# ─────────────────────────────────────────────────────────────
# Simple analysis state (only what's needed)
# ─────────────────────────────────────────────────────────────
class AnalysisState:
    def __init__(self, sr: int, win_len: int = ANALYSIS_WIN, NW: float = 4.0):
        self.sr = sr
        self.win_len = win_len
        self.window = windows.dpss(win_len, NW, sym=False).astype(np.float32)
        self.win_scale = 1.0 / np.sum(self.window)
        self.freqs = np.fft.rfftfreq(win_len, 1.0 / sr).astype(np.float32)
        self.bin_width = sr / win_len
        self.nyquist = sr / 2.0
        self.erb = 21.4 * np.log10(4.37e-3 * self.freqs + 1)
        self.pad = np.zeros(win_len, dtype=np.float32)

# ─────────────────────────────────────────────────────────────
# Extract top N candidates — no peak picking, just sort by ERB-weighted power
# ─────────────────────────────────────────────────────────────
def get_top_partials(audio: np.ndarray, center: int, state: AnalysisState, n_take: int):
    half = state.win_len // 2
    s = center - half
    e = center + half
    n = len(audio)

    if s < 0 or e > n:
        chunk = state.pad.copy()
        ss, se = max(0, s), min(n, e)
        chunk[ss - s : ss - s + (se - ss)] = audio[ss:se]
    else:
        chunk = audio[s:e]

    # FFT + window
    spec = np.fft.rfft(chunk.astype(np.float64) * state.window)
    mags = np.abs(spec).astype(np.float32) * state.win_scale

    # ERB-weighted "loudness" for sorting
    power = mags ** 2
    hfc_equiv = power * state.erb

    # Take top N (no min distance, no interpolation)
    idx = np.argpartition(hfc_equiv, -n_take)[-n_take:]
    sort_order = np.argsort(hfc_equiv[idx])[::-1]
    best_idx = idx[sort_order]

    freqs = state.freqs[best_idx]
    amps  = mags[best_idx]

    # Basic plausibility filter
    mask = (freqs >= 20.0) & (freqs <= state.nyquist - state.bin_width)
    freqs = freqs[mask][:n_take]
    amps  = amps[mask][:n_take]

    # If we got fewer than requested, we just live with it (no fallback)
    return freqs.astype(np.float32), np.clip(amps, 0.0, 1.0).astype(np.float32)

# ─────────────────────────────────────────────────────────────
# Write RSC6 file — same format as original
# ─────────────────────────────────────────────────────────────
def write_rsc(
    path: str,
    frame_freqs: np.ndarray,        # [n_frames, n_partials] Hz
    frame_amps:  np.ndarray,        # [n_frames, n_partials] linear [0,1]
    sample_rate: int,
    frame_size: int,
    total_samples: int,
) -> None:
    n_frames, n_partials = frame_freqs.shape
    mask_sz = (n_partials + 7) // 8

    freq_scale = 65535.0 / (sample_rate / 2.0)
    f_q = np.clip(np.round(frame_freqs * freq_scale), 0, 65535).astype(np.int32)
    a_mu = mulaw_encode(frame_amps)   # uint8

    bitmask_buf = bytearray()
    born_buf    = bytearray()
    freq_deltas = []
    amp_deltas  = []

    prev_fq  = np.zeros(n_partials, dtype=np.int32)
    prev_amu = np.zeros(n_partials, dtype=np.int32)
    was_alive = np.zeros(n_partials, dtype=bool)

    for i in range(n_frames):
        alive = frame_amps[i] > ALIVE_THRESHOLD
        born = alive & ~was_alive

        for slot in np.where(alive)[0]:
            fq  = int(f_q[i, slot])
            amu = int(a_mu[i, slot])

            if born[slot]:
                born_buf += struct.pack("<HB", fq, amu)
            else:
                df = fq - prev_fq[slot]
                da = amu - prev_amu[slot]
                if -32768 <= df <= 32767 and -128 <= da <= 127:
                    freq_deltas.append(df)
                    amp_deltas.append(da)
                else:
                    # overflow → treat as birth
                    born_buf += struct.pack("<HB", fq, amu)
                    born[slot] = True

            prev_fq[slot]  = fq
            prev_amu[slot] = amu

        # dead partials → reset trackers
        dead = was_alive & ~alive
        prev_fq[dead] = 0
        prev_amu[dead] = 0

        # bitmasks (little-endian, padded with zeros)
        alive_pad = np.zeros(mask_sz * 8, dtype=np.uint8)
        born_pad  = np.zeros(mask_sz * 8, dtype=np.uint8)
        alive_pad[:n_partials] = alive.astype(np.uint8)
        born_pad[:n_partials]  = born.astype(np.uint8)

        bitmask_buf += bytes(np.packbits(alive_pad, bitorder="little"))
        bitmask_buf += bytes(np.packbits(born_pad,  bitorder="little"))

        was_alive = alive

        if (i + 1) % 500 == 0 or (i + 1) == n_frames:
            print(f" … frame {i+1}/{n_frames}", end="\r")

    print()

    # Rice encode deltas
    fd = np.array(freq_deltas, dtype=np.int32) if freq_deltas else np.array([])
    ad = np.array(amp_deltas, dtype=np.int32) if amp_deltas else np.array([])

    fd_zz = zigzag(fd)
    ad_zz = zigzag(ad)

    k_freq = optimal_rice_k(fd_zz)
    k_amp  = optimal_rice_k(ad_zz)

    print(f" Rice   k_freq={k_freq}  k_amp={k_amp}   | "
          f"{len(fd)} freq Δ   {len(ad)} amp Δ")

    rice_freq = rice_encode(fd_zz, k_freq)
    rice_amp  = rice_encode(ad_zz, k_amp)

    # Header
    header = struct.pack(
        "<4sBIIHIIHBBII",
        b"RSC6", 6,
        sample_rate, frame_size, n_partials,
        total_samples, n_frames,
        mask_sz, k_freq, k_amp,
        len(born_buf), len(rice_freq)
    )

    with open(path, "wb") as f:
        f.write(header)
        f.write(bitmask_buf)
        f.write(born_buf)
        f.write(rice_freq)
        f.write(rice_amp)

    total_sz = len(header) + len(bitmask_buf) + len(born_buf) + len(rice_freq) + len(rice_amp)
    kb = total_sz / 1024
    print(f" Wrote {path}  —  {kb:.1f} KB   {n_frames} frames")

# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────
def encode_wav_to_rsc(input_path: str, output_path: str, n_partials: int, target_sr: int):
    print(f" Barebones RSC6  —  {input_path}  →  {output_path}")
    print(f" partials: {n_partials}   target sr: {target_sr} Hz")

    y, sr = librosa.load(input_path, sr=target_sr, mono=True)
    y = y.astype(np.float32)
    if np.max(np.abs(y)) > 1e-9:
        y /= np.max(np.abs(y))

    total_samples = len(y)
    frame_size = round(target_sr / TARGET_FPS)
    n_frames = math.ceil(total_samples / frame_size)

    # pad last frame if needed
    pad = n_frames * frame_size - total_samples
    if pad > 0:
        y = np.pad(y, (0, pad), mode='constant')

    state = AnalysisState(target_sr, ANALYSIS_WIN)

    all_freqs = np.zeros((n_frames, n_partials), dtype=np.float32)
    all_amps  = np.zeros((n_frames, n_partials), dtype=np.float32)

    for i in range(n_frames):
        center = i * frame_size + frame_size // 2
        freqs, amps = get_top_partials(y, center, state, n_partials)

        # pad with zeros if fewer than n_partials
        n_got = len(freqs)
        if n_got < n_partials:
            freqs = np.pad(freqs, (0, n_partials - n_got))
            amps  = np.pad(amps,  (0, n_partials - n_got))

        all_freqs[i] = freqs
        all_amps[i]  = amps

        if (i + 1) % 500 == 0 or (i + 1) == n_frames:
            print(f" … frame {i+1}/{n_frames}", end="\r")

    print()
    write_rsc(output_path, all_freqs, all_amps, target_sr, frame_size, total_samples)

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser("Barebones RSC6 encoder")
    p.add_argument("--input",    "-i", required=True)
    p.add_argument("--output",   "-o", default=None)
    p.add_argument("--partials", "-n", type=int, default=DEFAULT_PARTIALS)
    p.add_argument("--samplerate","-r", type=int, default=DEFAULT_SR, choices=[22050,44100])
    args = p.parse_args()

    out = args.output or (args.input.removesuffix(".wav") + RSC_EXTENSION)
    encode_wav_to_rsc(args.input, out, args.partials, args.samplerate)

    print("Done! 🔥")