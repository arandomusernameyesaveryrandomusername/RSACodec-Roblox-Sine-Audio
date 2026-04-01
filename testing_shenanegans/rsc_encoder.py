from __future__ import annotations

"""
rsc_encoder.py -- Roblox Sine Codec (RSC) Encoder  [OPTIMIZED]

Key optimisations vs. reference encoder
========================================
1.  @njit on every hot path
    • _zigzag_njit            — no Python boxing, single typed loop
    • _optimal_k_njit         — tight loop, zero numpy intermediate arrays
    • _rice_encode_njit       — manual MSB-first bit-writer (two-pass);
                                  avoids the 5 large intermediate numpy arrays
                                  of the reference; fully cache-friendly
    • _parabolic_interp_njit  — replaces masked-array numpy version;
                                  one allocation per call instead of ~6
    • _score_all_frames_njit  — fused score computation + custom local-max
                                  peak finder; replaces scipy.signal.find_peaks
                                  (~50 µs Python overhead eliminated per frame);
                                  processes ALL frames in one JIT call so that
                                  prev_mags state is maintained correctly and
                                  n_frames Python→JIT boundary crossings become 1
    • _track_greedy           — replaces O(n²) bubble-sort with np.argsort
                                  O(n log n); argsort used for both active-slot
                                  ordering and birth candidate ordering

2.  Two-phase FFT pipeline
    Phase A — FFT only: all n_frames spectra computed in a
              ThreadPoolExecutor (numpy.fft releases the GIL → true
              multi-core parallelism).  Writes into a single pre-allocated
              (n_frames × n_bins) float32 matrix, no per-frame heap alloc.
    Phase B — Scoring + peak-finding: single @njit call over all frames
              (must be sequential to preserve spectral-flux state).

3.  All float32 in analysis; float64 kept only for the FFT itself
    (numerical accuracy) and one-time ATH precomputation.

4.  RTF (real-time factor) reported at completion.

Usage:
    python rsc_encoder.py --input audio.wav --output audio.rsc
    python rsc_encoder.py --input audio.wav --output audio.rsc \\
        --partials 384 --samplerate 44100
"""

import argparse
import math
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor

import librosa
import numpy as np
from numba import njit
from scipy.signal import windows
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
TARGET_FPS         = 60
DEFAULT_PARTIALS   = 384
DEFAULT_SAMPLERATE = 44100
RSC_EXTENSION      = ".rsc"
ANALYSIS_WIN       = 2048
SLOT_COOLDOWN      = 1
ALIVE_THRESHOLD    = 0


# ─────────────────────────────────────────────────────────────────────────────
#  ATH  (one-time, stays in Python/NumPy)
# ─────────────────────────────────────────────────────────────────────────────
def _ath_db(freq: np.ndarray) -> np.ndarray:
    f  = np.maximum(np.asarray(freq, dtype=np.float64), 20.0)
    fk = f / 1000.0
    return np.clip(
        3.64 * fk**-0.8
        - 6.5 * np.exp(-0.6 * (fk - 3.3)**2)
        + 1e-3 * fk**4,
        -90.0, 90.0,
    )


def _ath_linear(n_bins: int, sample_rate: int, win: int,
                ath_gain_db: float = 0.0) -> np.ndarray:
    bin_freqs = np.arange(n_bins, dtype=np.float64) * sample_rate / win
    ath_dbfs  = _ath_db(np.maximum(bin_freqs, 20.0)) - 96.0 + ath_gain_db
    return (10.0 ** (ath_dbfs / 20.0)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Quantisation
# ─────────────────────────────────────────────────────────────────────────────

def _mu_encode(x: np.ndarray) -> np.ndarray:
    # mu-law style: preserves quiet partials much better
    mu = np.float32(65535.0)
    x  = np.clip(x, 0.0, 1.0)
    return np.clip(np.round(np.log1p(mu * x) / np.log1p(mu) * 65535.0), 0, 65535).astype(np.uint16)


# ─────────────────────────────────────────────────────────────────────────────
#  JIT — Zigzag encode
# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=True)
def _zigzag_njit(arr: np.ndarray) -> np.ndarray:
    """signed int32 -> non-negative uint32  (0->0, -1->1, 1->2, -2->3, ...)"""
    out = np.empty(len(arr), dtype=np.uint32)
    for i in range(len(arr)):
        a = np.int32(arr[i])
        out[i] = np.uint32((a << np.int32(1)) ^ (a >> np.int32(31)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  JIT — Optimal Rice k
# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=True)
def _optimal_k_njit(vals: np.ndarray) -> int:
    """Brute-force optimal Rice k in [0,16].  vals must be uint32."""
    n = len(vals)
    if n == 0:
        return 0
    best_k    = 0
    best_bits = np.int64(1) << np.int64(62)   # sentinel
    for k in range(17):
        bits = np.int64(n) * np.int64(1 + k)
        for i in range(n):
            bits += np.int64(vals[i]) >> np.int64(k)
        if bits < best_bits:
            best_bits = bits
            best_k    = k
    return best_k


# ─────────────────────────────────────────────────────────────────────────────
#  JIT — Rice encoder  (manual MSB-first bit-writer)
# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=True)
def _rice_encode_njit(vals: np.ndarray, k: int) -> np.ndarray:
    """
    Encode a uint32 array with Rice(k) coding into an MSB-first byte stream.

    Two passes share the same inner logic:
      Pass 1 — count total bits  (one shift + one add per value)
      Pass 2 — write bits into a zeroed buffer
    Avoids the 5 large intermediate arrays of the numpy reference
    (quotients, remainders, code_lens, starts, bits).
    """
    n = len(vals)
    if n == 0:
        return np.empty(0, dtype=np.uint8)

    # Pass 1 — count bits
    total_bits = np.int64(0)
    for i in range(n):
        total_bits += (np.int64(vals[i]) >> np.int64(k)) + np.int64(1 + k)

    out = np.zeros(int((total_bits + 7) >> 3), dtype=np.uint8)

    # Pass 2 — write bits
    bit_pos = np.int64(0)
    k_mask  = np.int64((1 << k) - 1)

    for i in range(n):
        v = np.int64(vals[i])
        q = int(v >> np.int64(k))
        r = int(v  &  k_mask)

        bit_pos += np.int64(q)                       # unary zeros (buffer is 0)
        bpos = int(bit_pos)
        out[bpos >> 3] |= np.uint8(np.uint8(1) << np.uint8(7 - (bpos & 7)))
        bit_pos += np.int64(1)

        for b in range(k - 1, -1, -1):
            if (r >> b) & 1:
                bpos = int(bit_pos)
                out[bpos >> 3] |= np.uint8(np.uint8(1) << np.uint8(7 - (bpos & 7)))
            bit_pos += np.int64(1)

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  JIT — Score all frames + peak-find  (replaces scipy.signal.find_peaks loop)
# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=True)
def _score_all_frames_njit(
    all_mags:     np.ndarray,   # float32 (n_frames, n_bins)
    ath_lin:      np.ndarray,   # float32 (n_bins,)
    bin_width:    np.float32,
    nyquist:      np.float32,
    n_candidates: int,
    cand_freqs:   np.ndarray,   # float32 (n_frames, n_candidates) — output
    cand_amps:    np.ndarray,   # float32 (n_frames, n_candidates) — output
    cand_counts:  np.ndarray,   # int32   (n_frames,)              — output
) -> None:
    """
    For every frame:
      1. Compute per-bin combined score (SNR-weighted, flux-boosted).
      2. Find local maxima (equivalent to scipy find_peaks with distance=1).
      3. Sort by descending score, parabolic-interpolate, frequency-filter,
         and write into the output candidate arrays.

    Sequential loop maintains prev_mags across frames for spectral flux.
    Replaces n_frames individual Python/scipy calls with one JIT call.
    """
    n_frames, n_bins = all_mags.shape
    prev_mags = np.zeros(n_bins, dtype=np.float32)
    score     = np.empty(n_bins, dtype=np.float32)
    f_low     = np.float32(20.0)
    f_high    = nyquist - bin_width

    for fi in range(n_frames):
        mags = all_mags[fi]

        # ── Spectral flux (scalar for this frame) ─────────────────────────
        flux = np.float32(0.0)
        for b in range(n_bins):
            d = np.log1p(mags[b]) - np.log1p(prev_mags[b])
            if d > np.float32(0.0):
                flux += (d * (np.float32(1.0)) + 1)

        # ── Per-bin combined score + update prev_mags ─────────────────────
        for b in range(n_bins):
            snr      = mags[b] / max(ath_lin[b], np.float32(1e-12))
            score[b] = mags[b] * flux * np.log1p(snr)
            prev_mags[b] = mags[b]

        # ── Find local maxima (find_peaks distance=1) ─────────────────────
        n_peaks = 0
        for b in range(1, n_bins - 1):
            if (score[b] > score[b - 1] and score[b] >= score[b + 1]
                    and score[b] > np.float32(1e-12)):
                n_peaks += 1

        if n_peaks == 0:
            # Fallback: take top n_candidates by raw score
            top = np.argsort(-score)
            ci  = 0
            for pi in range(min(n_bins, n_candidates * 4)):
                if ci >= n_candidates:
                    break
                bi = top[pi]
                if bi < 1 or bi >= n_bins - 1:
                    continue
                alpha = np.float64(mags[bi - 1])
                beta  = np.float64(mags[bi    ])
                gamma = np.float64(mags[bi + 1])
                denom = alpha - 2.0 * beta + gamma
                if abs(denom) > 1e-12:
                    off = 0.5 * (alpha - gamma) / denom
                    f   = np.float32((bi + off) * bin_width)
                    a   = np.float32(beta - 0.25 * (alpha - gamma) * off)
                else:
                    f = np.float32(bi * bin_width)
                    a = mags[bi]
                if f >= f_low and f <= f_high:
                    cand_freqs[fi, ci] = f
                    cand_amps[fi, ci]  = min(a, np.float32(1.0))
                    ci += 1
            cand_counts[fi] = ci
            continue

        # ── Collect peak indices ──────────────────────────────────────────
        peak_idx = np.empty(n_peaks, dtype=np.int32)
        j = 0
        for b in range(1, n_bins - 1):
            if (score[b] > score[b - 1] and score[b] >= score[b + 1]
                    and score[b] > np.float32(1e-12)):
                peak_idx[j] = b
                j += 1

        # Sort peaks by descending combined score
        order = np.argsort(-score[peak_idx])
        sorted_peaks = peak_idx[order]

        # ── Parabolic interpolation + filter → candidate slots ────────────
        ci = 0
        for pi in range(len(sorted_peaks)):
            if ci >= n_candidates:
                break
            bi = sorted_peaks[pi]
            alpha = np.float64(mags[bi - 1])
            beta  = np.float64(mags[bi    ])
            gamma = np.float64(mags[bi + 1])
            denom = alpha - 2.0 * beta + gamma
            if abs(denom) > 1e-12:
                off = 0.5 * (alpha - gamma) / denom
                f   = np.float32((bi + off) * bin_width)
                a   = np.float32(beta - 0.25 * (alpha - gamma) * off)
            else:
                f = np.float32(bi * bin_width)
                a = mags[bi]
            if f >= f_low and f <= f_high:
                cand_freqs[fi, ci] = f
                cand_amps[fi, ci]  = min(a, np.float32(1.0))
                ci += 1

        cand_counts[fi] = ci


# ─────────────────────────────────────────────────────────────────────────────
#  JIT — Greedy tracker  (argsort replaces O(n²) bubble-sort)
# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=True)
def _track_greedy(
    cand_f:         np.ndarray,   # float32 (nc,) — valid candidates this frame
    cand_a:         np.ndarray,   # float32 (nc,)
    prev_f:         np.ndarray,   # float32 (n_partials,)
    prev_a:         np.ndarray,   # float32 (n_partials,)
    prevprev_f:     np.ndarray,   # float32 (n_partials,)
    n_partials:     int,
    cooldowns:      np.ndarray,   # int32   (n_partials,)
    cooldown_frames: int = 1,
) -> tuple:
    out_f = np.zeros(n_partials, dtype=np.float32)
    out_a = np.zeros(n_partials, dtype=np.float32)

    for i in range(n_partials):
        cd = cooldowns[i] - 1
        cooldowns[i] = cd if cd > 0 else 0

    nc = len(cand_f)
    if nc == 0:
        return out_f, out_a, cooldowns

    claimed = np.zeros(nc, dtype=np.bool_)

    # Active slots sorted by descending amplitude via argsort  O(P log P)
    active_slots = np.where(prev_a > np.float32(0.0))[0].astype(np.int32)
    n_active = len(active_slots)
    if n_active > 1:
        active_slots = active_slots[np.argsort(-prev_a[active_slots])]

    # Match each active slot to its nearest unclaimed candidate
    for ii in range(n_active):
        slot = active_slots[ii]
        sc   = prev_f[slot]

        best_d  = np.float32(1e12)
        best_bi = -1
        for ci in range(nc):
            if not claimed[ci]:
                d = cand_f[ci] - sc
                if d < np.float32(0.0):
                    d = -d
                if d < best_d:
                    best_d  = d
                    best_bi = ci

        if best_bi < 0:
            break

        # ERB-rate of the predicted centre frequency (Cams)
        sc_erb  = np.float32(21.4) * np.log10(np.float32(4.37e-3) * sc + np.float32(1.0))

        # 1-Cam bandwidth expressed back in Hz at sc — this IS the perceptual channel width
        one_cam_hz = (np.float32(10.0) ** ((sc_erb + np.float32(1.0)) / np.float32(21.4)) - \
                    np.float32(10.0) ** (sc_erb / np.float32(21.4))) / np.float32(4.37e-3)

        # Base tolerance: half a Cam (symmetric perceptual unit, no arbitrary Hz value)
        tol = one_cam_hz * 0.5

        # Direction asymmetry: rising = wider (ratio of adjacent Cam boundaries, not a picked number)
        sc_erb_up   = sc_erb + np.float32(0.5)
        sc_erb_down = sc_erb - np.float32(0.5)
        hz_up   = (np.float32(10.0) ** (sc_erb_up   / np.float32(21.4)) - np.float32(1.0)) / np.float32(4.37e-3)
        hz_down = (np.float32(10.0) ** (sc_erb_down / np.float32(21.4)) - np.float32(1.0)) / np.float32(4.37e-3)
        asym    = (hz_up - sc) / (sc - hz_down)   # >1 above sc, <1 below — derived from ERB curvature
        tol    *= asym if cand_f[best_bi] > sc else (np.float32(1.0) / asym)

        # Amplitude weighting: log1p is its own scale — ratio of log1p(prev_a) to log1p(1.0) (unit amplitude)
        tol    *= np.float32(1.0) - np.log1p(prev_a[slot]) / np.log1p(np.float32(1.0))

        # Velocity: express drift as fraction of one_cam_hz per frame — no Hz/frame magic number
        vel  = abs(prev_f[slot] - prevprev_f[slot]) if prevprev_f[slot] > np.finfo(np.float32).eps else np.float32(0.0)
        tol *= np.float32(1.0) + vel / one_cam_hz

        if best_d <= tol * np.float32(1.8):
            out_f[slot]      = cand_f[best_bi]
            out_a[slot]      = cand_a[best_bi]
            claimed[best_bi] = True
        else:
            cooldowns[slot] = cooldown_frames

    # Births: unclaimed candidates sorted by descending amplitude
    n_unclaimed = 0
    for ci in range(nc):
        if not claimed[ci]:
            n_unclaimed += 1
    if n_unclaimed == 0:
        return out_f, out_a, cooldowns

    unclaimed = np.empty(n_unclaimed, dtype=np.int32)
    j = 0
    for ci in range(nc):
        if not claimed[ci]:
            unclaimed[j] = ci
            j += 1
    if n_unclaimed > 1:
        unclaimed = unclaimed[np.argsort(-cand_a[unclaimed])]

    # Empty slots (out_a==0 and not cooling down)
    n_empty = 0
    for s in range(n_partials):
        if out_a[s] == np.float32(0.0) and cooldowns[s] == 0:
            n_empty += 1
    empty = np.empty(n_empty, dtype=np.int32)
    j = 0
    for s in range(n_partials):
        if out_a[s] == np.float32(0.0) and cooldowns[s] == 0:
            empty[j] = s
            j += 1

    for i in range(min(n_unclaimed, n_empty)):
        bi = unclaimed[i]
        sl = empty[i]
        if cand_a[bi] > np.float32(1e-6) and cand_f[bi] > np.float32(1e-3):
            out_f[sl] = cand_f[bi]
            out_a[sl] = cand_a[bi]

    return out_f, out_a, cooldowns


# ─────────────────────────────────────────────────────────────────────────────
#  Analysis state  (pre-computed constants, shared read-only across threads)
# ─────────────────────────────────────────────────────────────────────────────
class AnalysisState:
    def __init__(self, sample_rate: int, analysis_win: int = ANALYSIS_WIN):
        self.win       = analysis_win
        self.sr        = sample_rate
        self.window    = windows.dpss(analysis_win, 3).astype(np.float32)
        self.win_scale = np.float32(1.0 / float(np.sum(self.window)))
        self.bin_width = np.float32(float(sample_rate) / analysis_win)
        self.nyquist   = np.float32(sample_rate / 2.0)
        self.min_dist  = 1
        n_bins         = analysis_win // 2 + 1
        freqs          = np.fft.rfftfreq(analysis_win, d=1.0 / sample_rate).astype(np.float32)
        self.erb       = (21.4 * np.log10(4.37e-3 * freqs + 1)).astype(np.float32)
        self.ath_lin   = _ath_linear(n_bins, sample_rate, analysis_win)


# ─────────────────────────────────────────────────────────────────────────────
#  Phase A — parallel FFT extraction
# ─────────────────────────────────────────────────────────────────────────────
def _compute_all_spectra(
    audio:    np.ndarray,
    centers:  list,
    state:    AnalysisState,
) -> np.ndarray:
    """
    Compute all FFT magnitude spectra in parallel via ThreadPoolExecutor.
    numpy.fft releases the GIL -> true multi-core parallelism.
    Pre-allocates a single (n_frames x n_bins) matrix; each thread writes
    its own row with no locking.
    """
    n_frames = len(centers)
    n_bins   = state.win // 2 + 1
    all_mags = np.empty((n_frames, n_bins), dtype=np.float32)
    half     = state.win // 2
    n_audio  = len(audio)
    window   = state.window        # read-only
    scale    = state.win_scale

    def _fft_one(i: int) -> None:
        c = centers[i]
        s, e = c - half, c + half
        if s < 0 or e > n_audio:
            chunk = np.zeros(state.win, dtype=np.float32)
            ss, se = max(0, s), min(n_audio, e)
            chunk[ss - s : ss - s + (se - ss)] = audio[ss:se]
        else:
            chunk = audio[s:e]
        spec = np.fft.rfft(chunk.astype(np.float64) * window)
        all_mags[i] = np.abs(spec).astype(np.float32) * scale

    n_workers = min(os.cpu_count() or 4, n_frames)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(tqdm(
            pool.map(_fft_one, range(n_frames)),
            total=n_frames, desc="   FFT      ",
            unit="frame", dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} frames  [{elapsed}<{remaining}  {rate_fmt}]",
        ))
    return all_mags


# ─────────────────────────────────────────────────────────────────────────────
#  WAV Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_audio(path: str, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    peak  = np.max(np.abs(y))
    return (y / peak).astype(np.float32), target_sr


# ─────────────────────────────────────────────────────────────────────────────
#  RSC6 Binary Writer
# ─────────────────────────────────────────────────────────────────────────────
def write_rsc(
    path:          str,
    frame_freqs:   np.ndarray,
    frame_amps:    np.ndarray,
    sample_rate:   int,
    frame_size:    int,
    total_samples: int,
) -> None:
    n_frames, n_partials = frame_freqs.shape
    mask_sz    = (n_partials + 7) // 8
    freq_scale = 65535.0 / (sample_rate / 2.0)

    f_q   = np.clip(np.round(frame_freqs * freq_scale), 0, 65535).astype(np.int32)
    a_lin = _mu_encode(frame_amps)

    # Vectorised delta computation (unchanged — already optimal numpy)
    alive      = frame_amps > ALIVE_THRESHOLD
    was_alive  = np.vstack([np.zeros((1, n_partials), bool), alive[:-1]])
    nat_born   = alive & ~was_alive
    continuing = alive & was_alive

    f_q_prev  = np.vstack([np.zeros((1, n_partials), np.int32), f_q[:-1]])
    amu_prev  = np.vstack([np.zeros((1, n_partials), np.int32), a_lin[:-1].astype(np.int32)])

    df_mat = (f_q - f_q_prev).astype(np.int32)
    da_mat = (a_lin.astype(np.int32) - amu_prev)

    overflow = continuing & (
        (df_mat < -32768) | (df_mat > 32767) |
        (da_mat < -32768) | (da_mat >  32767)
    )
    born_bits_mat = nat_born | overflow
    valid_cont    = continuing & ~overflow

    # Bitmasks (vectorised packbits)
    pad_w     = mask_sz * 8
    alive_pad = np.zeros((n_frames, pad_w), np.uint8)
    born_pad  = np.zeros((n_frames, pad_w), np.uint8)
    alive_pad[:, :n_partials] = alive
    born_pad [:, :n_partials] = born_bits_mat
    alive_packed = np.packbits(alive_pad, axis=1, bitorder="little")
    born_packed  = np.packbits(born_pad,  axis=1, bitorder="little")
    bitmask_buf  = np.stack([alive_packed, born_packed], axis=1).tobytes()

    # Born buffer
    br, bc = np.where(born_bits_mat)
    if len(br):
        bfq  = f_q[br, bc].astype(np.uint16)
        bamu = a_lin[br, bc]
        raw  = np.empty(len(br) * 4, np.uint8)
        raw[0::4] = (bfq  & 0xFF).astype(np.uint8)
        raw[1::4] = (bfq  >>  8 ).astype(np.uint8)
        raw[2::4] = (bamu & 0xFF).astype(np.uint8)
        raw[3::4] = (bamu >>  8 ).astype(np.uint8)
        born_buf = raw.tobytes()
    else:
        born_buf = b""

    # Delta arrays
    cr, cc      = np.where(valid_cont)
    freq_deltas = df_mat[cr, cc].astype(np.int32) if len(cr) else np.array([], np.int32)
    amp_deltas  = da_mat[cr, cc].astype(np.int32) if len(cr) else np.array([], np.int32)
    print(f"   Delta pass  |  {len(br)} births  |  {len(cr)} continuing")

    # Zigzag + optimal k (JIT)
    fd_zz  = _zigzag_njit(freq_deltas)
    ad_zz  = _zigzag_njit(amp_deltas)
    k_freq = _optimal_k_njit(fd_zz)
    k_amp  = _optimal_k_njit(ad_zz)
    print(f"   Rice k_freq={k_freq}  k_amp={k_amp}")

    # Rice encode (JIT manual bit-writer)
    rice_freq = bytes(_rice_encode_njit(fd_zz, k_freq))
    rice_amp  = bytes(_rice_encode_njit(ad_zz, k_amp))

    # Write file
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
    assert len(header) == 35, f"Header size mismatch: {len(header)}"
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
    print(f"  ✅ Wrote {n_frames} frames -> {path}")
    print(f"     {kb:.1f} KB  ({saving4:.1f}% smaller than RSC4  "
          f"{kb / (n_frames / TARGET_FPS):.2f} KB/s avg)")
    print(f"     Bitmasks {len(bitmask_buf)/1024:.1f} KB  |  Born {born_data_sz/1024:.1f} KB"
          f"  |  Rice-freq {rice_freq_sz/1024:.1f} KB  |  Rice-amp {len(rice_amp)/1024:.1f} KB")


# ─────────────────────────────────────────────────────────────────────────────
#  Main encode pipeline
# ─────────────────────────────────────────────────────────────────────────────
def encode(
    input_path:  str,
    output_path: str,
    n_partials:  int,
    target_sr:   int,
) -> None:
    print(f"RSC Encoder (optimised)  --  {input_path}")
    print(f"   Partials/frame : {n_partials}  |  Target SR: {target_sr} Hz")

    t0 = time.perf_counter()

    samples, sr = load_audio(input_path, target_sr)
    print(f"   {len(samples)} samples  ({len(samples)/sr:.2f}s)  @{sr} Hz")

    total_samples = len(samples)
    frame_size    = int(round(sr / TARGET_FPS))
    n_frames      = math.ceil(total_samples / frame_size)
    pad           = n_frames * frame_size - total_samples
    if pad > 0:
        samples = np.concatenate([samples, np.zeros(pad, np.float32)])
    print(f"   Frame size     : {frame_size} samp ({1000*frame_size/sr:.2f} ms)"
          f"  |  {n_frames} frames")

    state = AnalysisState(sr)
    max_meaningful = state.win // state.min_dist // 2
    if n_partials > max_meaningful:
        print(f"   Clamping partials {n_partials} -> {max_meaningful}")
        n_partials = max_meaningful

    # ── Phase A: parallel FFT ─────────────────────────────────────────────
    centers = [i * frame_size + frame_size // 2 for i in range(n_frames)]
    print(f"   Phase A - parallel FFT ({os.cpu_count()} threads) ...")
    t1 = time.perf_counter()
    all_mags = _compute_all_spectra(samples, centers, state)
    print(f"   Phase A done in {time.perf_counter() - t1:.2f}s")

    # ── Phase B: score + peak-find (JIT, one call for all frames) ─────────
    cand_freqs  = np.zeros((n_frames, n_partials), dtype=np.float32)
    cand_amps   = np.zeros((n_frames, n_partials), dtype=np.float32)
    cand_counts = np.zeros(n_frames, dtype=np.int32)
    print(f"   Phase B - scoring + peak-finding (JIT) ...")
    t2 = time.perf_counter()
    _score_all_frames_njit(
        all_mags, state.ath_lin,
        state.bin_width, state.nyquist, n_partials,
        cand_freqs, cand_amps, cand_counts,
    )
    print(f"   Phase B done in {time.perf_counter() - t2:.2f}s")
    del all_mags   # free ~(n_frames * n_bins * 4) bytes

    # ── Phase C: greedy tracking (JIT, sequential) ────────────────────────
    print(f"   Phase C - greedy tracking (JIT) ...")
    t3 = time.perf_counter()
    all_f = np.zeros((n_frames, n_partials), dtype=np.float32)
    all_a = np.zeros((n_frames, n_partials), dtype=np.float32)

    prev_f     = np.zeros(n_partials, np.float32)
    prev_a     = np.zeros(n_partials, np.float32)
    prevprev_f = np.zeros(n_partials, np.float32)
    cooldowns  = np.zeros(n_partials, np.int32)

    for i in tqdm(range(n_frames), desc="   Tracking  ",
                  unit="frame", dynamic_ncols=True,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} frames  [{elapsed}<{remaining}  {rate_fmt}]"):
        nc  = int(cand_counts[i])
        of, oa, cooldowns = _track_greedy(
            cand_freqs[i, :nc], cand_amps[i, :nc],
            prev_f, prev_a, prevprev_f, n_partials, cooldowns,
        )
        all_f[i]   = of
        all_a[i]   = oa
        prevprev_f = prev_f.copy()
        prev_f     = of.copy()
        prev_a     = oa.copy()

    print(f"   Phase C done in {time.perf_counter() - t3:.2f}s\n")

    # ── Phase D: encode + write ───────────────────────────────────────────
    write_rsc(output_path, all_f, all_a, sr, frame_size, total_samples)

    elapsed = time.perf_counter() - t0
    dur     = total_samples / sr
    print(f"   Done in {elapsed:.2f}s  (RTF {elapsed/dur:.3f}x)")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="RSC6 Encoder (optimised)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",      "-i", required=True)
    p.add_argument("--output",     "-o", default=None)
    p.add_argument("--partials",   "-n", type=int, default=DEFAULT_PARTIALS)
    p.add_argument("--samplerate", "-r", type=int, default=DEFAULT_SAMPLERATE,
                   choices=[22050, 44100])
    args = p.parse_args()
    out  = args.output or (args.input.removesuffix(".wav") + RSC_EXTENSION)
    encode(args.input, out, args.partials, args.samplerate)


if __name__ == "__main__":
    main()