from __future__ import annotations

"""
rsc_encoder.py -- Roblox Sine Codec (RSC) Encoder  —  RSC7
Usage:
    python rsc_encoder.py --input audio.wav --output audio.rsc
    python rsc_encoder.py --input audio.wav --output audio.rsc --partials 384 --samplerate 44100 --workers 4

RSC7 adds a Sinusoidal+Noise (SMS-style) residual model on top of RSC6:

  For each frame:
    1. Synthesise the sine model (MQ interpolated, same as decoder).
    2. Compute residual = original_frame − sine_frame.
    3. FFT the residual with a Hann window.
    4. Compute per-band RMS energy over N_NOISE_BANDS log-spaced bands.
    5. Mu-law quantise each band energy to uint8 and store in Section 5.

  Section 5 layout: nF * N_NOISE_BANDS bytes, one uint8 per band per frame.
  Header gains two new fields (total = 37 bytes):
    u8  n_noise_bands   (= N_NOISE_BANDS, currently 32)
    u32 noise_data_sz   (= nF * N_NOISE_BANDS bytes)

  The decoder reads Section 5, dequantises, generates per-band shaped white
  noise, and adds it to the sinusoidal output.

RSC7 Header (37 bytes):
  "RSC7" | u8 ver | u32 sr | u32 frame_sz | u16 n_partials |
  u32 total_samples | u32 total_frames | u16 mask_sz |
  u8 k_freq | u8 k_amp | u32 born_data_sz | u32 rice_freq_sz |
  u8 n_noise_bands | u32 noise_data_sz
"""

import argparse
import math
import os
import struct
import librosa
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
from scipy.signal import find_peaks, windows
from numba import njit

# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────
TARGET_FPS         = 60
DEFAULT_PARTIALS   = 384
DEFAULT_SAMPLERATE = 44100
RSC_EXTENSION      = ".rsc"
ANALYSIS_WIN       = 2048
SLOT_COOLDOWN      = 1
MU                 = 255.0
ALIVE_THRESHOLD    = 0
_LOG1P_MU          = math.log1p(MU)

# ── Noise model ───────────────────────────────────────────────
N_NOISE_BANDS      = 32          # log-spaced bands stored per frame
NOISE_FLOOR_DB     = -80.0       # below this → treat as silence (uint8 = 0)
NOISE_REF_DB       = 0.0         # 0 dB FS reference → uint8 = 255

def _ath_db(freq: np.ndarray) -> np.ndarray:
    """
    Terhardt (1979) approximation of the ISO quiet-listening threshold.
    Returns dB SPL at each frequency (Hz). Values are in the range
    ~0 dB (1–4 kHz) up to ~80 dB near DC and the Nyquist edge.
    """
    f   = np.asarray(freq, dtype=np.float64)
    f   = np.maximum(f, 20.0)
    fk  = f / 1000.0
    ath = (
          3.64  * fk ** -0.8
        - 6.5   * np.exp(-0.6 * (fk - 3.3) ** 2)
        + 1e-3  * fk ** 4
    )
    return np.clip(ath, -90.0, 90.0)


def _ath_linear(n_bins: int, sample_rate: int, win: int,
                ath_gain_db: float = 0.0) -> np.ndarray:
    bin_freqs = np.arange(n_bins, dtype=np.float64) * sample_rate / win
    ath_spl   = _ath_db(np.maximum(bin_freqs, 20.0))
    ath_dbfs  = ath_spl - 96.0 + ath_gain_db
    ath_lin   = 10.0 ** (ath_dbfs / 20.0)
    return ath_lin.astype(np.float32)

# ─────────────────────────────────────────────────────────────
#  Mu-law
# ─────────────────────────────────────────────────────────────
def _mulaw_encode(x: np.ndarray) -> np.ndarray:
    """float32 [0,1] → uint8 [0,255]"""
    x = np.clip(x.astype(np.float64), 0.0, 1.0)
    return np.clip(np.round(MU * np.log1p(MU * x) / _LOG1P_MU), 0, 255).astype(np.uint8)

def _mulaw_decode(u: np.ndarray) -> np.ndarray:
    """uint8 [0,255] → float32 [0,1]"""
    u_norm = u.astype(np.float32) / np.float32(MU)
    return (np.exp(u_norm * np.float32(_LOG1P_MU)) - 1.0) / np.float32(MU)

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
#  WAV Loading
# ─────────────────────────────────────────────────────────────
def load_audio(path: str, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    peak = np.max(np.abs(y))
    y = y / peak
    return y.astype(np.float32), target_sr

# ─────────────────────────────────────────────────────────────
#  Analysis State
# ─────────────────────────────────────────────────────────────
class AnalysisState:
    def __init__(self, sample_rate: int, analysis_win: int = ANALYSIS_WIN):
        self.win       = analysis_win
        self.sr        = sample_rate
        self.window    = windows.hann(analysis_win).astype(np.float32)
        self.win_scale = 1.0 / float(np.sum(self.window))
        self.freqs     = np.fft.rfftfreq(analysis_win, d=1.0 / sample_rate).astype(np.float32)
        self.bin_width = float(sample_rate) / analysis_win
        self.min_dist  = 1
        self.nyquist   = sample_rate / 2.0
        self.pad_buf   = np.zeros(analysis_win, dtype=np.float32)
        self.erb       = 21.4 * np.log10(4.37e-3 * self.freqs + 1)
        self.prev_mags = np.zeros(len(self.freqs), dtype=np.float32)
        self.ath_lin    = _ath_linear(len(self.freqs), self.sr, self.win)


# ─────────────────────────────────────────────────────────────
#  Noise-band edge pre-computation
# ─────────────────────────────────────────────────────────────
def _noise_band_edges(sample_rate: int, n_bands: int,
                      f_lo: float = 20.0) -> np.ndarray:
    """
    Return (n_bands+1,) array of log-spaced frequency edges in Hz,
    from f_lo to Nyquist.  Used by both encoder and (implicitly) decoder.
    """
    f_hi = sample_rate / 2.0
    return np.exp(np.linspace(np.log(f_lo), np.log(f_hi), n_bands + 1)).astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  Sinusoidal frame synthesiser (for residual computation)
#  — single-frame version used inside the encoder.
# ─────────────────────────────────────────────────────────────
def _synth_frame(freqs_prev: np.ndarray, amps_prev: np.ndarray,
                 freqs_curr: np.ndarray, amps_curr: np.ndarray,
                 phases: np.ndarray,
                 frame_size: int, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesise one frame of MQ interpolated sines.
    Returns (samples float32, updated_phases float32).
    Mirrors the decoder's synthesis exactly so the residual is accurate.
    """
    T      = frame_size
    sr_f32 = np.float32(sample_rate)
    TWO_PI = np.float32(2.0 * math.pi)
    T_sec  = np.float32(T / sample_rate)

    t_sec  = np.arange(T, dtype=np.float32) / sr_f32
    t_norm = np.arange(T, dtype=np.float32) / np.float32(T)

    # Hold freq on death (like decoder)
    f_use  = np.where(freqs_curr > 0, freqs_curr, freqs_prev).astype(np.float32)
    active = (amps_curr > 0) | (amps_prev > 0)

    out    = np.zeros(T, dtype=np.float32)

    if active.any():
        act    = active
        pa     = amps_prev[act]
        ca     = amps_curr[act]
        fu     = f_use[act]
        ph     = phases[act].copy()

        # Born partials: reset phase
        born   = (amps_prev[act] == 0) & (amps_curr[act] > 0)
        ph[born] = np.float32(0.0)

        phase   = ph[:, None] + TWO_PI * fu[:, None] * t_sec[None, :]
        amp_env = pa[:, None] + (ca - pa)[:, None] * t_norm[None, :]
        out     = (amp_env * np.sin(phase)).sum(axis=0)

    # Advance phases
    new_phases = phases.copy()
    new_phases[active] = (phases[active] + TWO_PI * f_use[active] * T_sec) % TWO_PI
    new_phases[amps_curr == 0] = np.float32(0.0)

    return out, new_phases


# ─────────────────────────────────────────────────────────────
#  Noise-band energy encoder
# ─────────────────────────────────────────────────────────────
def encode_noise_bands(
    samples:      np.ndarray,   # full audio float32 [-1,1]
    all_f:        np.ndarray,   # (nF, nP) float32 Hz
    all_a:        np.ndarray,   # (nF, nP) float32 linear
    frame_size:   int,
    sample_rate:  int,
    n_bands:      int = N_NOISE_BANDS,
) -> np.ndarray:
    """
    For each frame:
      1. Synthesise the sine model.
      2. Residual = original − sine.
      3. FFT the residual (Hann-windowed).
      4. Compute RMS per log-spaced band.
      5. Mu-law quantise → uint8.

    Returns noise_bands (nF, n_bands) uint8.
    """
    nF, nP     = all_f.shape
    T          = frame_size
    win        = windows.hann(T).astype(np.float32)
    win_scale  = np.float32(1.0 / np.sum(win))
    fft_freqs  = np.fft.rfftfreq(T, d=1.0 / sample_rate).astype(np.float32)
    edges      = _noise_band_edges(sample_rate, n_bands)

    # Bin index ranges for each band
    band_slices = []
    for b in range(n_bands):
        lo_idx = int(np.searchsorted(fft_freqs, edges[b]))
        hi_idx = int(np.searchsorted(fft_freqs, edges[b + 1]))
        hi_idx = max(hi_idx, lo_idx + 1)   # at least 1 bin
        band_slices.append((lo_idx, hi_idx))

    # Pad audio to cover all frames
    total_needed = nF * T
    if len(samples) < total_needed:
        samples = np.concatenate([samples, np.zeros(total_needed - len(samples), np.float32)])

    noise_bands = np.zeros((nF, n_bands), dtype=np.uint8)

    phases = np.zeros(nP, dtype=np.float32)
    prev_f = np.zeros(nP, dtype=np.float32)
    prev_a = np.zeros(nP, dtype=np.float32)

    # Reference amplitude: RMS of a full-scale sine ≈ 1/√2
    # We normalise band RMS to [0,1] relative to that
    ref_amp = np.float32(1.0 / math.sqrt(2.0))

    for i in tqdm(range(nF), desc="   Noise enc",
                  unit="frame", dynamic_ncols=True,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} frames  [{elapsed}<{remaining}  {rate_fmt}]"):
        orig_frame  = samples[i * T : (i + 1) * T]
        sine_frame, phases = _synth_frame(
            prev_f, prev_a, all_f[i], all_a[i], phases, T, sample_rate
        )
        residual    = orig_frame - sine_frame

        # Windowed FFT of residual
        spec  = np.fft.rfft((residual * win).astype(np.float64))
        mags  = (np.abs(spec) * win_scale).astype(np.float32)

        for b, (lo, hi) in enumerate(band_slices):
            band_mags = mags[lo:hi]
            rms       = float(np.sqrt(np.mean(band_mags ** 2))) if len(band_mags) > 0 else 0.0
            # Normalise to [0,1]: full-scale ref = ref_amp
            norm      = min(rms / float(ref_amp), 1.0)
            noise_bands[i, b] = int(_mulaw_encode(np.array([norm], dtype=np.float32))[0])

        prev_f = all_f[i]
        prev_a = all_a[i]

    return noise_bands


# ─────────────────────────────────────────────────────────────
#  Parabolic Peak Interpolation
# ─────────────────────────────────────────────────────────────
def _parabolic_interp(
    idx_arr: np.ndarray,
    mags: np.ndarray,
    bin_width: float,
) -> tuple[np.ndarray, np.ndarray]:
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
        chunk = state.pad_buf.copy()
        ss, se = max(0, s), min(n, e)
        chunk[ss - s : ss - s + (se - ss)] = audio[ss:se]
    else:
        chunk = audio[s:e]
    spec  = np.fft.rfft(chunk.astype(np.float64) * state.window)
    mags  = np.abs(spec).astype(np.float32) * state.win_scale
    snr_ratio = mags / np.maximum(state.ath_lin, 1e-12)
    snr_score = np.log1p(snr_ratio)
    hfc   = mags * state.erb
    diff  = np.maximum(mags - state.prev_mags, 0.0)
    spectral_flux = float(np.sum(diff * state.erb + 1.0))
    state.prev_mags[:] = mags
    combined_score = hfc * spectral_flux * snr_score
    peak_idx, _ = find_peaks(combined_score, distance=state.min_dist, height=1e-12)
    if len(peak_idx) == 0:
        peak_idx = np.argpartition(combined_score, -n_candidates)[-n_candidates:]

    peak_scores = combined_score[peak_idx]
    sort_order  = np.argsort(peak_scores)[::-1]
    sorted_idx  = peak_idx[sort_order]
    n_peaks_total = len(sorted_idx)
    n_take = n_peaks_total
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
@njit(cache=True)
def _track_greedy(
    cand_f, cand_a,
    prev_f, prev_a,
    prevprev_f,
    n_partials,
    cooldowns,
    cooldown_frames=1,
):
    out_f = np.zeros(n_partials, dtype=np.float32)
    out_a = np.zeros(n_partials, dtype=np.float32)
    cooldowns[:] = np.maximum(0, cooldowns - 1)

    if len(cand_f) == 0:
        return out_f, out_a, cooldowns

    claimed = np.zeros(len(cand_f), dtype=np.bool_)
    active = np.where(prev_a > 0)[0]
    for i in range(len(active)):
        for j in range(i+1, len(active)):
            if prev_a[active[j]] > prev_a[active[i]]:
                active[i], active[j] = active[j], active[i]

    for slot_idx in active:
        if np.all(claimed):
            break
        predicted_f = prev_f[slot_idx]
        sc = predicted_f
        tol = max(sc * 0.038, (24.7 + 0.108 * sc) * 0.55)
        dists = np.where(~claimed, np.abs(cand_f - sc), 1e12)
        bi = np.argmin(dists)
        tol *= 1.25 if cand_f[bi] > sc else 0.85
        eps = 1e-12
        tol *= 1 - np.log1p(prev_a[slot_idx] + eps)/np.log1p(9 + eps)
        tol *= min(2.0, 1.0 + (abs(prev_f[slot_idx] - prevprev_f[slot_idx])
                    if prevprev_f[slot_idx] > 1e-3 else 0.0) / 80.0)
        if dists[bi] <= tol * 1.8:
            out_f[slot_idx] = cand_f[bi]
            out_a[slot_idx] = cand_a[bi]
            claimed[bi] = True
        else:
            cooldowns[slot_idx] = cooldown_frames

    births_idx = []
    for i in range(len(claimed)):
        if not claimed[i]:
            births_idx.append(i)
    for i in range(len(births_idx)):
        for j in range(i+1, len(births_idx)):
            if cand_a[births_idx[j]] > cand_a[births_idx[i]]:
                births_idx[i], births_idx[j] = births_idx[j], births_idx[i]

    empty_slots = []
    for i in range(n_partials):
        if out_a[i] == 0 and cooldowns[i] == 0:
            empty_slots.append(i)

    n_assign = min(len(births_idx), len(empty_slots))
    for i in range(n_assign):
        bi = births_idx[i]
        sl = empty_slots[i]
        if cand_a[bi] > 1e-6 and cand_f[bi] > 1e-3:
            out_f[sl] = cand_f[bi]
            out_a[sl] = cand_a[bi]

    return out_f, out_a, cooldowns


# ─────────────────────────────────────────────────────────────
#  RSC7 Binary Writer
#
#  Header (37 bytes):
#    "RSC7" | u8 ver | u32 sr | u32 frame_sz | u16 n_partials |
#    u32 total_samples | u32 total_frames | u16 mask_sz |
#    u8 k_freq | u8 k_amp | u32 born_data_sz | u32 rice_freq_sz |
#    u8 n_noise_bands | u32 noise_data_sz
#
#  Section 1 — Bitmasks    : nF * 2 * mask_sz  bytes
#  Section 2 — Born data   : born_data_sz       bytes
#  Section 3 — Rice freq   : rice_freq_sz       bytes
#  Section 4 — Rice amp    : remaining_rice_amp bytes
#  Section 5 — Noise bands : nF * n_noise_bands bytes  (uint8)
# ─────────────────────────────────────────────────────────────
def write_rsc(
    path:          str,
    frame_freqs:   np.ndarray,
    frame_amps:    np.ndarray,
    noise_bands:   np.ndarray,   # (nF, N_NOISE_BANDS) uint8
    sample_rate:   int,
    frame_size:    int,
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

    pad_w     = mask_sz * 8
    alive_pad = np.zeros((n_frames, pad_w), np.uint8)
    born_pad  = np.zeros((n_frames, pad_w), np.uint8)
    alive_pad[:, :n_partials] = alive
    born_pad [:, :n_partials] = born_bits_mat
    alive_packed = np.packbits(alive_pad, axis=1, bitorder="little")
    born_packed  = np.packbits(born_pad,  axis=1, bitorder="little")
    stacked      = np.stack([alive_packed, born_packed], axis=1)
    bitmask_buf  = stacked.tobytes()

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

    # ── Noise section ─────────────────────────────────────────
    n_noise_bands = noise_bands.shape[1]
    noise_buf     = noise_bands.astype(np.uint8).tobytes()
    noise_data_sz = len(noise_buf)
    print(f"   Noise bands: {n_noise_bands}  |  {noise_data_sz/1024:.1f} KB")

    # ── Write file ────────────────────────────────────────────
    born_data_sz = len(born_buf)
    rice_freq_sz = len(rice_freq)
    # RSC7 header = 37 bytes (RSC6 was 35; +u8 n_noise_bands +u32 noise_data_sz)
    header = struct.pack(
        "<4sBIIHIIHBBIIBI",
        b"RSC7", 7,
        sample_rate, frame_size, n_partials,
        total_samples, n_frames,
        mask_sz, k_freq, k_amp,
        born_data_sz, rice_freq_sz,
        n_noise_bands, noise_data_sz,
    )
    assert len(header) == 40, f"Header size wrong: {len(header)}"
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(bitmask_buf)
        fh.write(born_buf)
        fh.write(rice_freq)
        fh.write(rice_amp)
        fh.write(noise_buf)   # Section 5

    total_sz = len(header) + len(bitmask_buf) + born_data_sz + rice_freq_sz + len(rice_amp) + noise_data_sz
    kb       = total_sz / 1024
    print(f"  ✅ Wrote {n_frames} frames → {path}")
    print(f"     {kb:.1f} KB  total")
    print(f"     Bitmasks {len(bitmask_buf)/1024:.1f} KB  |  Born {born_data_sz/1024:.1f} KB"
          f"  |  Rice-freq {rice_freq_sz/1024:.1f} KB  |  Rice-amp {len(rice_amp)/1024:.1f} KB"
          f"  |  Noise {noise_data_sz/1024:.1f} KB")


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
    max_meaningful = int(state.win // state.min_dist // 2)
    if n_partials > max_meaningful:
        print(f"   ⚠  Clamping partials {n_partials} → {max_meaningful} "
              f"(max find_peaks can deliver at this window size)")
        n_partials = max_meaningful
    n_cand = n_partials
    print(f"   Analysis win   : {ANALYSIS_WIN} samp ({state.bin_width:.1f} Hz/bin)"
          f"  |  n_cand={n_cand}  cooldown={SLOT_COOLDOWN}  workers={n_workers}")

    # ── Phase 1: parallel FFT candidate extraction ────────────────────────
    centers = [i * frame_size + frame_size // 2 for i in range(n_frames)]
    print(f"   Extracting FFT candidates ({n_workers} thread(s)) ...")

    def _extract(center: int) -> tuple[np.ndarray, np.ndarray]:
        return _fft_candidates(samples, center, state, n_cand)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        candidates = list(pool.map(_extract, tqdm(centers, desc="FFT extraction")))

    # ── Phase 2: sequential greedy tracking ──────────────────────────────
    print(f"   Tracking partials ...")
    all_f = np.zeros((n_frames, n_partials), dtype=np.float32)
    all_a = np.zeros((n_frames, n_partials), dtype=np.float32)
    prev_f     = np.zeros(n_partials, np.float32)
    prev_a     = np.zeros(n_partials, np.float32)
    prevprev_f = np.zeros(n_partials, np.float32)
    cooldowns  = np.zeros(n_partials, np.int32)
    for i, (cf, ca) in enumerate(tqdm(candidates, desc="Tracking partials")):
        of, oa, cooldowns = _track_greedy(
            cf, ca, prev_f, prev_a, prevprev_f, n_partials, cooldowns
        )
        all_f[i]   = of
        all_a[i]   = oa
        prevprev_f = prev_f
        prev_f     = of
        prev_a     = oa
    print()

    # ── Phase 3: noise residual encoding ─────────────────────────────────
    print("   Computing noise residual ...")
    noise_bands = encode_noise_bands(
        samples, all_f, all_a, frame_size, sample_rate, N_NOISE_BANDS
    )

    write_rsc(output_path, all_f, all_a, noise_bands, sample_rate, frame_size, total_samples)


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="RSC7 Encoder",
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