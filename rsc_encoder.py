"""
rsc_encoder.py -- Roblox Sine Codec (RSC) Encoder

Usage:
    python rsc_encoder.py --input audio.wav --output audio.rsc
    python rsc_encoder.py --input audio.wav --output audio.rsc --partials 384 --samplerate 44100 --workers 4
"""

import argparse
import math
import os
import struct
import librosa
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.signal import find_peaks, windows


# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────
TARGET_FPS         = 60
DEFAULT_PARTIALS   = 384
DEFAULT_SAMPLERATE = 44100
RSC_EXTENSION      = ".rsc"
ANALYSIS_WIN       = 4096
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
    """signed int32 → non-negative uint32  (0→0, -1→1, 1→2, -2→3, …)"""
    a = arr.astype(np.int32)
    return ((a << 1) ^ (a >> 31)).astype(np.uint32)


def _optimal_k(vals: np.ndarray) -> int:
    """Brute-force optimal Rice parameter k in [0, 15]."""
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
    """
    Vectorised Rice encoder — MSB-first bit packing.
    vals  : 1-D uint32 array (non-negative, already zigzag encoded)
    k     : Rice parameter
    returns: packed bytes
    """
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
#  WAV Loading
# ─────────────────────────────────────────────────────────────


def load_audio(path: str, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    # y: float32 array, sr: sample rate
    y, sr = librosa.load(path, sr=target_sr, mono=True)  # auto-resamples & mono
    # normalize to [-1, 1]
    peak = np.max(np.abs(y))
    y = y / peak
    return y.astype(np.float32), target_sr



# ─────────────────────────────────────────────────────────────
#  Analysis State
# ─────────────────────────────────────────────────────────────
class AnalysisState:
    def __init__(self, sample_rate: int, analysis_win: int = ANALYSIS_WIN, NW: float = 4.0):
        self.win       = analysis_win
        self.sr        = sample_rate
        self.window    = windows.dpss(analysis_win, NW, sym=False).astype(np.float32)
        self.win_scale = 1.0 / float(np.sum(self.window))
        self.freqs     = np.fft.rfftfreq(analysis_win, d=1.0 / sample_rate).astype(np.float32)
        self.bin_width = float(sample_rate) / analysis_win
        self.min_dist  = max(2, int(round(25.0 / self.bin_width)))
        self.nyquist   = sample_rate / 2.0
        self.pad_buf   = np.zeros(analysis_win, dtype=np.float32)
        self.erb       = 21.4 * np.log10(4.37e-3 * self.freqs + 1)


# ─────────────────────────────────────────────────────────────
#  Parabolic Peak Interpolation (module-level — no closure overhead)
# ─────────────────────────────────────────────────────────────
def _parabolic_interp(
    idx_arr: np.ndarray,
    mags: np.ndarray,
    bin_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised quadratic peak interpolation → (Hz freqs, amplitudes)."""
    idx      = idx_arr.astype(np.int32)
    ref_bins = idx.astype(np.float64)
    ref_mags = mags[idx].astype(np.float64)

    valid = (idx >= 1) & (idx < len(mags) - 1)
    if valid.any():
        k     = idx[valid]
        alpha = mags[k - 1].astype(np.float64)
        beta  = mags[k    ].astype(np.float64)
        gamma = mags[k + 1].astype(np.float64)
        denom = alpha - 2.0 * beta + gamma
        safe  = np.abs(denom) > 1e-12

        offset = np.zeros(valid.sum(), dtype=np.float64)
        offset[safe] = 0.5 * (alpha[safe] - gamma[safe]) / denom[safe]

        ref_bins[valid] = k + offset
        ref_mags[valid] = beta - 0.25 * (alpha - gamma) * offset

    return ref_bins * bin_width, ref_mags


# ─────────────────────────────────────────────────────────────
#  FFT Candidate Extraction
# ─────────────────────────────────────────────────────────────
def _fft_candidates(
    audio: np.ndarray,
    center: int,
    state: AnalysisState,
    n_candidates: int,
) -> tuple[np.ndarray, np.ndarray]:
    half = state.win // 2
    s, e = center - half, center + half
    n    = len(audio)

    if s < 0 or e > n:
        # state.pad_buf is read-only here — safe for concurrent use
        chunk = state.pad_buf.copy()
        ss, se = max(0, s), min(n, e)
        chunk[ss - s : ss - s + (se - ss)] = audio[ss:se]
    else:
        chunk = audio[s:e]

    spec  = np.fft.rfft(chunk.astype(np.float64) * state.window)
    mags  = np.abs(spec).astype(np.float32) * state.win_scale
    hfc   = mags**2 * state.erb

    peak_idx, _ = find_peaks(mags, distance=state.min_dist, height=1e-6)
    if len(peak_idx) == 0:
        peak_idx = np.argpartition(hfc, -n_candidates)[-n_candidates:]

    peak_hfc      = hfc[peak_idx]
    sort_order    = np.argsort(peak_hfc)[::-1]
    sorted_idx    = peak_idx[sort_order]
    n_peaks_total = len(sorted_idx)

    if n_peaks_total <= n_candidates:
        oversample_factor = 1.0
    else:
        loudest           = peak_hfc[sort_order[0]] + 1e-12
        rel               = peak_hfc[sort_order] / loudest
        cum               = np.cumsum(rel)
        knee              = np.searchsorted(cum, 0.80 * cum[-1])
        sig               = max(n_candidates, knee + 1)
        n_top             = min(n_candidates * 2, n_peaks_total // 2 + 1)
        top_ratio         = np.sum(peak_hfc[sort_order[:n_top]]) / (np.sum(peak_hfc) + 1e-12)
        spread            = 1.0 / (top_ratio + 0.05)
        oversample_factor = 1.0 + (sig / n_candidates) * spread

    n_take = min(n_peaks_total, int(n_candidates * oversample_factor + 0.5))
    n_take = max(n_take, min(n_candidates, n_peaks_total))

    pool_freqs, pool_amps = _parabolic_interp(sorted_idx[:n_take], mags, state.bin_width)

    mask      = (pool_freqs >= 20.0) & (pool_freqs <= state.nyquist - state.bin_width)
    top_freqs = pool_freqs[mask].astype(np.float32)
    top_mags  = pool_amps[mask].astype(np.float32)

    if len(top_freqs) < n_candidates:
        extra_needed  = n_candidates - len(top_freqs)
        remaining_idx = sorted_idx[n_take:]
        if len(remaining_idx) > 0:
            rem_freqs, rem_amps = _parabolic_interp(remaining_idx, mags, state.bin_width)
            rem_mask   = (rem_freqs >= 20.0) & (rem_freqs <= state.nyquist - state.bin_width)
            top_freqs  = np.concatenate([top_freqs, rem_freqs[rem_mask][:extra_needed].astype(np.float32)])
            top_mags   = np.concatenate([top_mags,  rem_amps[rem_mask][:extra_needed].astype(np.float32)])

    return top_freqs, np.clip(top_mags, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────
#  Greedy Peak Tracker
# ─────────────────────────────────────────────────────────────
def _track_greedy(
    cand_f: np.ndarray, cand_a: np.ndarray,
    prev_f: np.ndarray, prev_a: np.ndarray,
    prevprev_f: np.ndarray,
    n_partials: int,
    cooldowns: np.ndarray,
    cooldown_frames: int = SLOT_COOLDOWN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_f = np.zeros(n_partials, dtype=np.float32)
    out_a = np.zeros(n_partials, dtype=np.float32)
    cooldowns = np.maximum(0, cooldowns - 1)

    if len(cand_f) == 0:
        return out_f, out_a, cooldowns

    claimed = np.zeros(len(cand_f), dtype=bool)
    active  = np.where(prev_a > 0)[0]
    active  = active[np.argsort(prev_a[active])[::-1]]

    for slot in active:
        if claimed.all():
            break


        predicted_f = prev_f[slot]

        sc            = predicted_f
        tol           = max(sc * 0.038, (24.7 + 0.108 * sc) * 0.55)
        dists         = np.where(~claimed, np.abs(cand_f - sc), np.inf)
        bi            = int(np.argmin(dists))
        tol          *= 1.25 if cand_f[bi] > sc else 0.85
        eps = 1e-12  # avoid log(0)
        tol *= 1 - np.log1p(prev_a[slot] + eps)/np.log1p(9 + eps)
        tol          *= min(2.0, 1.0 + (abs(prev_f[slot] - prevprev_f[slot])
                             if prevprev_f[slot] > 1e-3 else 0.0) / 80.0)

        if dists[bi] <= tol * 1.8:
            out_f[slot] = cand_f[bi]
            out_a[slot] = cand_a[bi]
            claimed[bi] = True
        else:
            cooldowns[slot] = cooldown_frames

    births = np.where(~claimed)[0]
    if len(births):
        births   = births[np.argsort(cand_a[births])[::-1]]
        empty    = np.where((out_a == 0) & (cooldowns == 0))[0]
        n_assign = min(len(births), len(empty))
        bi_valid = births[:n_assign]
        mask     = (cand_a[bi_valid] > 1e-6) & (cand_f[bi_valid] > 1e-3)
        sl, bi_v = empty[:n_assign][mask], bi_valid[mask]
        out_f[sl] = cand_f[bi_v]
        out_a[sl] = cand_a[bi_v]

    return out_f, out_a, cooldowns


# ─────────────────────────────────────────────────────────────
#  RSC6 Binary Writer
#
#  Header (35 bytes):
#    "RSC6" | u8 ver | u32 sr | u32 frame_sz | u16 n_partials |
#    u32 total_samples | u32 total_frames | u16 mask_sz |
#    u8 k_freq | u8 k_amp | u32 born_data_sz | u32 rice_freq_sz
#
#  Section 1 — Bitmasks  : nF * 2 * mask_sz  bytes
#  Section 2 — Born data : born_data_sz       bytes  (uint16 fq + uint8 amu, uncompressed)
#  Section 3 — Rice freq : rice_freq_sz       bytes  (zigzag+Rice(k_freq) freq deltas)
#  Section 4 — Rice amp  : remaining          bytes  (zigzag+Rice(k_amp)  amp  deltas)
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
    a_mu = _mulaw_encode(frame_amps)                                # (F, P) uint8

    # ── Vectorised Pass 1 ────────────────────────────────────────────────
    #
    # was_alive[i, slot] = alive[i-1, slot] (False for frame 0 — no prior context).
    # Continuing slots: alive at both i-1 and i.
    # Delta validity range matches original int16/int8 checks exactly.
    # np.where returns indices in row-major (frame-first, ascending slot) order,
    # so born_buf and delta list order are identical to the original Python loop.
    #
    alive      = frame_amps > ALIVE_THRESHOLD                           # (F, P) bool
    was_alive  = np.vstack([np.zeros((1, n_partials), bool), alive[:-1]])
    nat_born   = alive & ~was_alive
    continuing = alive & was_alive

    f_q_prev  = np.vstack([np.zeros((1, n_partials), np.int32), f_q[:-1]])
    amu_prev  = np.vstack([np.zeros((1, n_partials), np.int32), a_mu[:-1].astype(np.int32)])

    df_mat = (f_q - f_q_prev).astype(np.int32)
    da_mat = (a_mu.astype(np.int32) - amu_prev)

    overflow = continuing & (
        (df_mat < -32768) | (df_mat > 32767) |
        (da_mat <   -128) | (da_mat >    127)
    )

    born_bits_mat = nat_born | overflow
    valid_cont    = continuing & ~overflow

    # ── Bitmasks (vectorised, no per-frame loop) ─────────────────────────
    pad_w     = mask_sz * 8
    alive_pad = np.zeros((n_frames, pad_w), np.uint8)
    born_pad  = np.zeros((n_frames, pad_w), np.uint8)
    alive_pad[:, :n_partials] = alive
    born_pad [:, :n_partials] = born_bits_mat

    alive_packed = np.packbits(alive_pad, axis=1, bitorder="little")  # (F, mask_sz)
    born_packed  = np.packbits(born_pad,  axis=1, bitorder="little")  # (F, mask_sz)

    # np.stack → (F, 2, mask_sz); .tobytes() in C order yields
    # alive[0], born[0], alive[1], born[1], ... — identical to original.
    stacked     = np.stack([alive_packed, born_packed], axis=1)
    bitmask_buf = stacked.tobytes()

    # ── Born buffer (batch uint16-LE + uint8, no struct loop) ────────────
    br, bc = np.where(born_bits_mat)          # ascending frame then slot
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

    # ── Delta arrays ──────────────────────────────────────────────────────
    cr, cc      = np.where(valid_cont)
    freq_deltas = df_mat[cr, cc].astype(np.int32) if len(cr) else np.array([], np.int32)
    amp_deltas  = da_mat[cr, cc].astype(np.int32) if len(cr) else np.array([], np.int32)

    print(f"   Pass 1 done  |  {len(br)} births  |  {len(cr)} continuing deltas")

    # ── Pass 2: zigzag + Rice encode delta streams ────────────────────────
    fd_zz  = _zigzag(freq_deltas)
    ad_zz  = _zigzag(amp_deltas)
    k_freq = _optimal_k(fd_zz)
    k_amp  = _optimal_k(ad_zz)
    print(f"   Rice k_freq={k_freq}  k_amp={k_amp}"
          f"  |  {len(freq_deltas)} freq deltas  {len(amp_deltas)} amp deltas")

    rice_freq = _rice_encode(fd_zz, k_freq)
    rice_amp  = _rice_encode(ad_zz, k_amp)

    # ── Write file ────────────────────────────────────────────────────────
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
#  Main Encode Pipeline
# ─────────────────────────────────────────────────────────────
def encode(
    input_path: str,
    output_path: str,
    n_partials: int,
    target_sr: int,
    n_workers: int,
) -> None:
    print(f"RSC Encoder  --  {input_path}")
    print(f"   Partials/frame : {n_partials}  |  Target SR: {target_sr} Hz")

    samples, native_sr = load_audio(input_path, target_sr)
    print(f"   Native SR      : {native_sr} Hz  |  {len(samples)} samples  "
          f"({len(samples)/native_sr:.2f}s)")

    if native_sr != target_sr:
        print(f"   Resampling {native_sr} → {target_sr} Hz ...")
        from math import gcd
        from scipy.signal import resample_poly
        g       = gcd(target_sr, native_sr)
        samples = resample_poly(samples, target_sr // g, native_sr // g).astype(np.float32)
        peak    = np.max(np.abs(samples))
        samples /= peak

    sample_rate   = target_sr
    total_samples = len(samples)
    frame_size    = int(round(sample_rate / TARGET_FPS))
    n_frames      = math.ceil(total_samples / frame_size)
    pad           = n_frames * frame_size - total_samples
    if pad > 0:
        samples = np.concatenate([samples, np.zeros(pad, np.float32)])

    print(f"   Frame size     : {frame_size} samp ({1000*frame_size/sample_rate:.2f} ms)"
          f"  |  {n_frames} frames")

    state = AnalysisState(sample_rate)

    max_meaningful = int(state.nyquist / state.bin_width / state.min_dist)
    if n_partials > max_meaningful:
        print(f"   ⚠  Clamping partials {n_partials} → {max_meaningful} "
              f"(max find_peaks can deliver at this window size)")
        n_partials = max_meaningful

    n_cand = n_partials
    print(f"   Analysis win   : {ANALYSIS_WIN} samp ({state.bin_width:.1f} Hz/bin)"
          f"  |  n_cand={n_cand}  cooldown={SLOT_COOLDOWN}  workers={n_workers}")

    # ── Phase 1: parallel FFT candidate extraction ────────────────────────
    # numpy's FFT releases the GIL, so threads genuinely parallelise here.
    # _fft_candidates only reads state.pad_buf (via .copy()) — thread-safe.
    centers = [i * frame_size + frame_size // 2 for i in range(n_frames)]
    print(f"   Extracting FFT candidates ({n_workers} thread(s)) ...")

    def _extract(center: int) -> tuple[np.ndarray, np.ndarray]:
        return _fft_candidates(samples, center, state, n_cand)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        candidates = list(pool.map(_extract, centers))

    # ── Phase 2: sequential greedy tracking ──────────────────────────────
    # Tracking is inherently serial (each frame depends on the previous),
    # but the FFT work above is already done.
    print(f"   Tracking partials ...")
    all_f = np.zeros((n_frames, n_partials), dtype=np.float32)
    all_a = np.zeros((n_frames, n_partials), dtype=np.float32)

    prev_f     = np.zeros(n_partials, np.float32)
    prev_a     = np.zeros(n_partials, np.float32)
    prevprev_f = np.zeros(n_partials, np.float32)
    cooldowns  = np.zeros(n_partials, np.int32)

    for i, (cf, ca) in enumerate(candidates):
        of, oa, cooldowns = _track_greedy(
            cf, ca, prev_f, prev_a, prevprev_f, n_partials, cooldowns
        )
        all_f[i]   = of
        all_a[i]   = oa
        prevprev_f = prev_f
        prev_f     = of
        prev_a     = oa

        if (i + 1) % 500 == 0 or (i + 1) == n_frames:
            print(f"   ... tracked frame {i+1}/{n_frames}", end="\r")
    print()

    write_rsc(output_path, all_f, all_a, sample_rate, frame_size, total_samples)


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="RSC6 Encoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",      "-i", required=True)
    p.add_argument("--output",     "-o", default=None)
    p.add_argument("--partials",   "-n", type=int, default=DEFAULT_PARTIALS)
    p.add_argument("--samplerate", "-r", type=int, default=DEFAULT_SAMPLERATE,
                   choices=[22050, 44100])
    p.add_argument("--workers",    "-w", type=int,
                   default=min(8, os.cpu_count() or 1),
                   help="Thread count for parallel FFT candidate extraction")
    args = p.parse_args()
    out  = args.output or (args.input.removesuffix(".wav") + RSC_EXTENSION)
    encode(args.input, out, args.partials, args.samplerate, args.workers)


if __name__ == "__main__":
    main()