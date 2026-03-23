from __future__ import annotations

"""
rmdctsc_encoder_minimal.py -- RMDCTSCv1 Minimal Encoder

Bare-minimum implementation: DPSS-windowed FFT top-N peak picking +
simple greedy frequency tracker. Same RSC6 bitstream as the full encoder.

Usage:
    python rsc_encoder_test_version.py --input audio.wav --output audio.rsc
    python rsc_encoder_test_version.py --input audio.wav --output audio.rsc --partials 384 --samplerate 44100
"""
import argparse
import math
import os
import struct

import librosa
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm
from scipy.signal import windows

# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────
TARGET_FPS         = 60
DEFAULT_PARTIALS   = 384
DEFAULT_SAMPLERATE = 44100
RMDCTSC_EXTENSION  = ".rmdctsc"
ANALYSIS_WIN       = 2048
SLOT_COOLDOWN      = 1
MU                 = 255.0
ALIVE_THRESHOLD    = 0
_LOG1P_MU          = math.log1p(MU)

# ─────────────────────────────────────────────────────────────
#  Mu-law
# ─────────────────────────────────────────────────────────────
def _mulaw_encode(x: np.ndarray) -> np.ndarray:
    """float32 [0,1] → uint8 [0,255]"""
    x = np.clip(x.astype(np.float64), 0.0, 1.0)
    return np.clip(np.round(MU * np.log1p(MU * x) / _LOG1P_MU), 0, 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────
#  Rice helpers
# ─────────────────────────────────────────────────────────────
def _zigzag(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.int32)
    return ((a << 1) ^ (a >> 31)).astype(np.uint32)

def _optimal_k(vals: np.ndarray) -> int:
    n = len(vals)
    if n == 0:
        return 0
    v64 = vals.astype(np.int64)
    best_k, best_bits = 0, float("inf")
    for k in range(16):
        bits = int((v64 >> k).sum()) + n * (1 + k)
        if bits < best_bits:
            best_bits, best_k = bits, k
    return best_k

def _rice_encode(vals: np.ndarray, k: int) -> bytearray:
    n = len(vals)
    if n == 0:
        return bytearray()
    v64        = vals.astype(np.int64)
    quotients  = v64 >> k
    remainders = v64 & ((1 << k) - 1)
    code_lens  = quotients + 1 + k
    total_bits = int(code_lens.sum())
    bits   = np.zeros(total_bits, dtype=np.uint8)
    starts = np.empty(n, dtype=np.int64)
    starts[0] = 0
    if n > 1:
        starts[1:] = np.cumsum(code_lens[:-1])
    bits[(starts + quotients).astype(int)] = 1
    for bit_idx in range(k):
        shift     = k - 1 - bit_idx
        positions = (starts + quotients + 1 + bit_idx).astype(int)
        bits[positions] = ((remainders >> shift) & 1).astype(np.uint8)
    pad = (-total_bits) % 8
    if pad:
        bits = np.append(bits, np.zeros(pad, dtype=np.uint8))
    return bytearray(np.packbits(bits, bitorder="big"))

# ─────────────────────────────────────────────────────────────
#  Audio loading
# ─────────────────────────────────────────────────────────────
def load_audio(path: str, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    peak = np.max(np.abs(y))
    if peak > 0.0:
        y = y / peak
    return y.astype(np.float32), target_sr

# ─────────────────────────────────────────────────────────────
#  FFT candidate extraction  (Hann window, top-N by amplitude)
# ─────────────────────────────────────────────────────────────
def _fft_candidates(
    audio: np.ndarray,
    center: int,
    win: int,
    hann: np.ndarray,
    win_scale: float,
    bin_width: float,
    nyquist: float,
    min_dist: int,
) -> tuple[np.ndarray, np.ndarray]:
    half = win // 2
    s, e = center - half, center + half
    n = len(audio)
    if s < 0 or e > n:
        chunk = np.zeros(win, dtype=np.float32)
        ss, se = max(0, s), min(n, e)
        chunk[ss - s: ss - s + (se - ss)] = audio[ss:se]
    else:
        chunk = audio[s:e]

    spec  = np.fft.rfft(chunk.astype(np.float64) * hann)
    mags  = (np.abs(spec) * win_scale).astype(np.float32)

    peak_idx, _ = find_peaks(mags, distance=min_dist, height=1e-6)
    if len(peak_idx) == 0:
        return np.array([], np.float32), np.array([], np.float32)

    # Parabolic interpolation for frequency and amplitude
    freqs = np.empty(len(peak_idx), dtype=np.float64)
    amps  = np.empty(len(peak_idx), dtype=np.float64)
    for i, k in enumerate(peak_idx):
        if 1 <= k < len(mags) - 1:
            alpha, beta, gamma = float(mags[k-1]), float(mags[k]), float(mags[k+1])
            denom = alpha - 2.0 * beta + gamma
            offset = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-12 else 0.0
            freqs[i] = (k + offset) * bin_width
            amps[i]  = beta - 0.25 * (alpha - gamma) * offset
        else:
            freqs[i] = k * bin_width
            amps[i]  = float(mags[k])

    in_band = (freqs >= 20.0) & (freqs <= nyquist - bin_width)
    freqs = freqs[in_band].astype(np.float32)
    amps  = np.clip(amps[in_band], 0.0, 1.0).astype(np.float32)

    # Sort by amplitude descending (simple top-N selection happens in tracker)
    order = np.argsort(amps)[::-1]
    return freqs[order], amps[order]

# ─────────────────────────────────────────────────────────────
#  Greedy tracker  (fixed Hz tolerance: 1% of frequency)
# ─────────────────────────────────────────────────────────────
def _track_greedy(
    cand_f: np.ndarray, cand_a: np.ndarray,
    prev_f: np.ndarray, prev_a: np.ndarray,
    n_partials: int,
    cooldowns: np.ndarray,
    cooldown_frames: int = SLOT_COOLDOWN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_f     = np.zeros(n_partials, dtype=np.float32)
    out_a     = np.zeros(n_partials, dtype=np.float32)
    cooldowns = np.maximum(0, cooldowns - 1)

    if len(cand_f) == 0:
        return out_f, out_a, cooldowns

    claimed = np.zeros(len(cand_f), dtype=bool)

    # Continue active slots (loudest first)
    active = np.where(prev_a > 0)[0]
    active = active[np.argsort(prev_a[active])[::-1]]
    for slot in active:
        if claimed.all():
            break
        f0  = float(prev_f[slot])
        tol = max(f0 * 0.01, 20.0)   # 1% tolerance, min 20 Hz
        dists = np.where(~claimed, np.abs(cand_f - f0), np.inf)
        bi    = int(np.argmin(dists))
        if dists[bi] <= tol:
            out_f[slot] = cand_f[bi]
            out_a[slot] = cand_a[bi]
            claimed[bi] = True
        else:
            cooldowns[slot] = cooldown_frames

    # Birth unclaimed candidates into empty slots
    births = np.where(~claimed)[0]
    if len(births):
        empty    = np.where((out_a == 0) & (cooldowns == 0))[0]
        n_assign = min(len(births), len(empty))
        out_f[empty[:n_assign]] = cand_f[births[:n_assign]]
        out_a[empty[:n_assign]] = cand_a[births[:n_assign]]

    return out_f, out_a, cooldowns

# ─────────────────────────────────────────────────────────────
#  RSC6 bitstream writer  (identical to full encoder)
# ─────────────────────────────────────────────────────────────
def write_rsc(
    path: str,
    frame_freqs: np.ndarray,
    frame_amps:  np.ndarray,
    sample_rate: int,
    frame_size:  int,
    total_samples: int,
) -> None:
    n_frames, n_partials = frame_freqs.shape
    mask_sz    = (n_partials + 7) // 8
    freq_scale = 65535.0 / (sample_rate / 2.0)
    f_q  = np.clip(np.round(frame_freqs * freq_scale), 0, 65535).astype(np.int32)
    a_mu = _mulaw_encode(frame_amps)

    alive      = frame_amps > ALIVE_THRESHOLD
    was_alive  = np.vstack([np.zeros((1, n_partials), bool), alive[:-1]])
    nat_born   = alive & ~was_alive
    continuing = alive & was_alive

    f_q_prev = np.vstack([np.zeros((1, n_partials), np.int32), f_q[:-1]])
    amu_prev = np.vstack([np.zeros((1, n_partials), np.int32), a_mu[:-1].astype(np.int32)])
    df_mat   = (f_q - f_q_prev).astype(np.int32)
    da_mat   = (a_mu.astype(np.int32) - amu_prev)

    overflow = continuing & (
        (df_mat < -32768) | (df_mat > 32767) |
        (da_mat <   -255) | (da_mat >    255)
    )
    born_bits_mat = nat_born | overflow
    valid_cont    = continuing & ~overflow

    pad_w     = mask_sz * 8
    alive_pad = np.zeros((n_frames, pad_w), np.uint8)
    born_pad  = np.zeros((n_frames, pad_w), np.uint8)
    alive_pad[:, :n_partials] = alive
    born_pad [:, :n_partials] = born_bits_mat
    alive_packed = np.packbits(alive_pad, axis=1, bitorder="little")
    born_packed  = np.packbits(born_pad,  axis=1, bitorder="little")
    bitmask_buf  = np.stack([alive_packed, born_packed], axis=1).tobytes()

    br, bc = np.where(born_bits_mat)
    if len(br):
        bfq  = f_q [br, bc].astype(np.uint16)
        bamu = a_mu[br, bc]
        raw  = np.empty(len(br) * 3, np.uint8)
        raw[0::3] = (bfq & 0xFF).astype(np.uint8)
        raw[1::3] = (bfq >> 8  ).astype(np.uint8)
        raw[2::3] = bamu
        born_buf = raw.tobytes()
    else:
        born_buf = b""

    cr, cc      = np.where(valid_cont)
    freq_deltas = df_mat[cr, cc].astype(np.int32) if len(cr) else np.array([], np.int32)
    amp_deltas  = da_mat[cr, cc].astype(np.int32) if len(cr) else np.array([], np.int32)
    print(f"   Pass 1 done  |  {len(br)} births  |  {len(cr)} continuing deltas")

    fd_zz  = _zigzag(freq_deltas)
    ad_zz  = _zigzag(amp_deltas)
    k_freq = _optimal_k(fd_zz)
    k_amp  = _optimal_k(ad_zz)
    print(f"   Rice k_freq={k_freq}  k_amp={k_amp}"
          f"  |  {len(freq_deltas)} freq deltas  {len(amp_deltas)} amp deltas")
    rice_freq = _rice_encode(fd_zz, k_freq)
    rice_amp  = _rice_encode(ad_zz, k_amp)

    born_data_sz = len(born_buf)
    rice_freq_sz = len(rice_freq)
    header = struct.pack(
        "<4sBIIHIIHBBII",
        b"RSC6", 6,
        sample_rate, frame_size, n_partials,
        total_samples, n_frames,
        mask_sz, k_freq, k_amp,
        born_data_sz, rice_freq_sz,
    )
    assert len(header) == 35, f"Header size wrong: {len(header)}"
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(bitmask_buf)
        fh.write(born_buf)
        fh.write(rice_freq)
        fh.write(rice_amp)

    total_sz = 35 + len(bitmask_buf) + born_data_sz + rice_freq_sz + len(rice_amp)
    rsc4_sz  = 23 + n_frames * n_partials * 4
    kb       = total_sz / 1024
    saving4  = 100.0 * (1.0 - total_sz / rsc4_sz)
    print(f"  ✅ Wrote {n_frames} frames → {path}")
    print(f"     {kb:.1f} KB  ({saving4:.1f}% smaller than RSC4  {kb/60:.2f} KB/s avg)")
    print(f"     Bitmasks {len(bitmask_buf)/1024:.1f} KB  |  Born {born_data_sz/1024:.1f} KB"
          f"  |  Rice-freq {rice_freq_sz/1024:.1f} KB  |  Rice-amp {len(rice_amp)/1024:.1f} KB")

# ─────────────────────────────────────────────────────────────
#  Main encode pipeline
# ─────────────────────────────────────────────────────────────
def encode(
    input_path: str,
    output_path: str,
    n_partials: int,
    target_sr: int,
) -> None:
    print(f"RMDCTSCv1 Minimal Encoder  --  {input_path}")
    print(f"   Output slots : {n_partials}  |  Target SR: {target_sr} Hz")

    samples, sr   = load_audio(input_path, target_sr)
    total_samples = len(samples)
    print(f"   Loaded       : {total_samples} samples  ({total_samples/target_sr:.2f}s)")

    frame_size = int(round(target_sr / TARGET_FPS))
    n_frames   = math.ceil(total_samples / frame_size)
    pad        = n_frames * frame_size - total_samples
    if pad > 0:
        samples = np.concatenate([samples, np.zeros(pad, np.float32)])
    print(f"   Frame size   : {frame_size} samp  |  {n_frames} frames")

    win       = ANALYSIS_WIN
    hann      = windows.dpss(win, 4).astype(np.float64)
    win_scale = 1.0 / float(np.sum(hann))
    bin_width = float(target_sr) / win
    nyquist   = target_sr / 2.0
    min_dist  = max(2, int(round(25.0 / bin_width)))

    centers = [i * frame_size + frame_size // 2 for i in range(n_frames)]

    # Phase 1: FFT candidate extraction
    candidates = []
    for center in tqdm(centers, desc="   Analysing    ", unit="frame", dynamic_ncols=True,
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} frames  [{elapsed}<{remaining}  {rate_fmt}]"):
        cf, ca = _fft_candidates(samples, center, win, hann, win_scale, bin_width, nyquist, min_dist)
        candidates.append((cf, ca))

    # Phase 2: greedy tracking
    all_f     = np.zeros((n_frames, n_partials), dtype=np.float32)
    all_a     = np.zeros((n_frames, n_partials), dtype=np.float32)
    prev_f    = np.zeros(n_partials, np.float32)
    prev_a    = np.zeros(n_partials, np.float32)
    cooldowns = np.zeros(n_partials, np.int32)

    for i, (cf, ca) in enumerate(tqdm(candidates, desc="   Tracking     ", unit="frame", dynamic_ncols=True,
                                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} frames  [{elapsed}<{remaining}  {rate_fmt}]")):
        of, oa, cooldowns = _track_greedy(cf, ca, prev_f, prev_a, n_partials, cooldowns)
        all_f[i] = of
        all_a[i] = oa
        prev_f   = of
        prev_a   = oa

    write_rsc(output_path, all_f, all_a, target_sr, frame_size, total_samples)

# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="RMDCTSCv1 Minimal — Hann FFT top-N + greedy tracker, RSC6 bitstream",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",      "-i", required=True)
    p.add_argument("--output",     "-o", default=None)
    p.add_argument("--partials",   "-n", type=int, default=DEFAULT_PARTIALS)
    p.add_argument("--samplerate", "-r", type=int, default=DEFAULT_SAMPLERATE,
                   choices=[22050, 44100])
    args = p.parse_args()
    out  = args.output or (os.path.splitext(args.input)[0] + RMDCTSC_EXTENSION)
    encode(args.input, out, args.partials, args.samplerate)

if __name__ == "__main__":
    main()