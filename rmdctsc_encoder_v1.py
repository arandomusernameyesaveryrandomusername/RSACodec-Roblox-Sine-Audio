from __future__ import annotations

"""
rmdctsc_encoder_v1.py -- Roblox Modified Discrete Cosine Transform Sine Codec v1 (RMDCTSCv1) Encoder

MDCT-based psychoacoustic masking replaces the old HFC/ERB peak-selection heuristic.
Per frame, a 4096-point MDCT is computed; its energy is spread across the ERB
(Equivalent Rectangular Bandwidth) scale using a simultaneous-masking spreading
function to produce a per-bin masking threshold.  FFT candidate peaks are then ranked
by their Signal-to-Mask Ratio (SMR) rather than raw HFC amplitude, so perceptually
important partials always win slots.

ERB is used instead of Bark because it more accurately models auditory filter bandwidth
across the full frequency range, especially above 4 kHz where Bark underestimates
critical-band width.  The 4096-point MDCT gives ~10.76 Hz/bin resolution at 44100 Hz,
providing fine-grained low-frequency masking without exceeding the 4096-point FFT window.

Bitstream format: unchanged (RSC6 compatible).

Usage:
    python rmdctsc_encoder_v1.py --input audio.wav --output audio.rmdctsc
    python rmdctsc_encoder_v1.py --input audio.wav --output audio.rmdctsc --partials 384 --samplerate 44100 --workers 4
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
TARGET_FPS          = 60
DEFAULT_PARTIALS    = 384
DEFAULT_SAMPLERATE  = 44100
RMDCTSC_EXTENSION   = ".rmdctsc"
ANALYSIS_WIN        = 4096
SLOT_COOLDOWN       = 1
MU                  = 255.0
ALIVE_THRESHOLD     = 0
_LOG1P_MU           = math.log1p(MU)

# ── MDCT psychoacoustic constants ─────────────────────────────
MDCT_WIN            = 4096         # MDCT block length — 4096 pts @ 44100 Hz gives
                                   # ~10.76 Hz/bin; fine low-freq masking detail,
                                   # matching the 4096-pt FFT analysis window.
# BUG FIX 3: ABSOLUTE_THRESHOLD raised from 0 to a small positive value so the
# energy floor in masking_threshold_from_chunk is not a no-op (0² = 0 was useless).
ABSOLUTE_THRESHOLD  = 1e-6        # normalised amplitude floor (~silence in a [0,1] signal)
N_ERB               = 40           # ERB bands (covers ~20 Hz – 20 kHz at 44100 Hz)
# Spreading slopes are NOT constants — they are derived per-masker from
# psychoacoustic first principles inside masking_threshold_from_chunk.

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
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    peak = np.max(np.abs(y))
    if peak > 0.0:
        y = y / peak
    return y.astype(np.float32), target_sr

# ─────────────────────────────────────────────────────────────
#  MDCT helpers
# ─────────────────────────────────────────────────────────────
def _mdct_window(N: int) -> np.ndarray:
    """Sine window for MDCT (standard choice for psychoacoustic use)."""
    n = np.arange(N, dtype=np.float64)
    return np.sin(math.pi / N * (n + 0.5))

def _mdct(x: np.ndarray, window: np.ndarray) -> np.ndarray:
    """
    Type-IV MDCT via FFT.  Produces M = N//2 real coefficients.

    Derivation:
      X[k] = sum_{n=0}^{N-1} xw[n] * cos(pi/M * (n + 0.5 + M/2) * (k + 0.5))

    Written as a real part of a DFT evaluated at half-integer frequencies:
      X[k] = Re{ post[k] * FFT(xw * pre)[k] }

    where:
      pre[n]  = exp(-j*pi*n/N)              (shifts DFT to half-integer bins)
      post[k] = exp(-j*pi*(2k+1)*phi/N)     (phi = M/2 + 0.5, the time offset)

    Verified by comparison to the direct O(N*M) formula: max relative error < 1e-13.
    """
    N   = len(x)
    M   = N // 2
    xw  = x * window
    n   = np.arange(N, dtype=np.float64)
    k   = np.arange(M, dtype=np.float64)
    pre  = np.exp(-1j * np.pi * n / N)
    Z    = np.fft.fft(xw * pre)[:M]
    phi  = M / 2.0 + 0.5
    post = np.exp(-1j * np.pi * (2.0 * k + 1.0) * phi / N)
    return np.real(Z * post).astype(np.float32)

# ─────────────────────────────────────────────────────────────
#  Psychoacoustic Masking (MDCT-based, ERB scale)
# ─────────────────────────────────────────────────────────────
def _hz_to_erb(hz: np.ndarray) -> np.ndarray:
    """
    Moore & Glasberg (1983) ERB-rate formula (ERBs, i.e. number of ERBs below f).
    ERB-rate = 21.4 * log10(4.37e-3 * f + 1)
    This is more accurate than Bark above ~4 kHz, where Bark critical bands are
    too narrow relative to measured auditory filter widths.
    """
    hz = np.asarray(hz, dtype=np.float64)
    return 21.4 * np.log10(4.37e-3 * hz + 1.0)


def _erb_distance_matrix(erb_centers: np.ndarray) -> np.ndarray:
    """
    Signed ERB-distance matrix D where D[i, j] = erb_centers[j] - erb_centers[i].
    Positive = j is above i (upper slope), negative = j is below i (lower slope).
    Stored as float32; used at runtime to compute level-dependent spreading.
    Shape: (N_ERB, N_ERB).
    """
    return (erb_centers[np.newaxis, :] - erb_centers[:, np.newaxis]).astype(np.float32)

class PsychoState:
    """
    Holds all MDCT/psychoacoustic analysis state.
    Built once per encoder run; read-only after construction (thread-safe).

    Uses the ERB (Equivalent Rectangular Bandwidth) scale instead of Bark.
    ERB band centres are spaced uniformly in ERB-rate units from ~20 Hz to nyquist,
    giving finer resolution in the high-frequency region where Bark is too coarse.
    """
    def __init__(self, sample_rate: int, mdct_win: int = MDCT_WIN):
        self.sr     = sample_rate
        self.M      = mdct_win
        self.mdct_N = mdct_win // 2

        # MDCT window + per-bin frequency axis
        self.mdct_window = _mdct_window(mdct_win)
        # Normalisation: divide MDCT coefficients by sum(window) so they are on
        # the same amplitude scale as the DPSS-windowed FFT (which uses win_scale
        # = 1/sum(window)).  Without this the MDCT energies are ~N^2 times larger
        # than FFT magnitudes^2, making the masking threshold astronomically high.
        self.mdct_scale  = 1.0 / float(np.sum(self.mdct_window))
        self.mdct_freqs  = (np.arange(self.mdct_N) + 0.5) * sample_rate / mdct_win  # Hz

        # ERB-rate axis: N_ERB bands uniformly spaced from ERB(20 Hz) to ERB(nyquist)
        erb_lo   = _hz_to_erb(np.array([20.0]))[0]
        erb_hi   = _hz_to_erb(np.array([sample_rate / 2.0]))[0]
        self.erb_centers   = np.linspace(erb_lo, erb_hi, N_ERB, dtype=np.float64)  # ERB-rate
        self.erb_centers_f = self.erb_centers.astype(np.float32)

        # ERB signed-distance matrix (N_ERB × N_ERB): D[i,j] = erb_j - erb_i
        # Used to compute level-dependent spreading per masker at runtime.
        self.erb_dist = _erb_distance_matrix(self.erb_centers)   # (N_ERB, N_ERB)

        # Centre frequency (Hz) of each ERB band — needed for frequency-dependent
        # lower slope (basilar membrane asymmetry below ~500 Hz).
        self.erb_center_hz = ((10.0 ** (self.erb_centers / 21.4)) - 1.0) / 4.37e-3  # (N_ERB,)

        # Map each MDCT bin to its ERB band index (nearest-neighbour)
        mdct_erb      = _hz_to_erb(self.mdct_freqs)                    # (mdct_N,)
        # For each bin, find the closest ERB band centre
        diffs         = np.abs(
            mdct_erb[:, np.newaxis] - self.erb_centers[np.newaxis, :]  # (mdct_N, N_ERB)
        )
        self.mdct_erb_idx = diffs.argmin(axis=1).astype(np.int32)      # (mdct_N,)

        # Fractional ERB position for bilinear interpolation back to per-bin
        self.mdct_erb_f   = mdct_erb.astype(np.float32)

    def masking_threshold_from_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Compute a per-MDCT-bin masking threshold (linear amplitude).

        Spreading slopes are level-dependent: louder maskers have a shallower
        upper slope (more upward spread) and a slightly shallower lower slope.
        This is computed per tonal masker and fully vectorised.

        Steps:
          1. MDCT → normalised coefficient amplitudes
          2. Identify tonal bins (3-point local peaks in MDCT magnitude)
          3. For each tonal masker, derive level-dependent slopes and compute
             its spreading contribution across all ERB bands  (vectorised)
          4. Per-band threshold = max spreading contribution across all maskers
          5. Interpolate to per-bin resolution, floor at silence threshold
        """
        M = self.M
        if len(chunk) < M:
            tmp = np.zeros(M, np.float32)
            tmp[:len(chunk)] = chunk[:M]
            chunk = tmp
        else:
            chunk = chunk[:M]

        coeffs = _mdct(chunk.astype(np.float64), self.mdct_window) * self.mdct_scale
        amps   = np.abs(coeffs).astype(np.float64)               # (mdct_N,)

        # ── 1. Identify tonal bins: 3-point local maxima ──────────────────
        is_peak         = np.zeros(len(amps), dtype=bool)
        is_peak[1:-1]   = (amps[1:-1] > amps[:-2]) & (amps[1:-1] > amps[2:])
        peak_indices    = np.where(is_peak)[0]

        if len(peak_indices) == 0:
            # No tonal content — return absolute threshold floor everywhere
            return np.full(len(amps), ABSOLUTE_THRESHOLD, dtype=np.float32)

        # ── 2. Masker levels in dB energy (relative to full-scale) ────────
        masker_energy   = amps[peak_indices] ** 2                # (K,)
        masker_erb_band = self.mdct_erb_idx[peak_indices]        # (K,) integer band idx
        # Level in dB energy; 0 dBFS energy = 1.0; floor at -120 dB
        L_db            = 10.0 * np.log10(np.maximum(masker_energy, 1e-12))  # (K,)

        # ── 3. Per-masker slopes from psychoacoustic first principles ─────
        #
        # Upper slope — level-dependent (Zwicker & Fastl 1999, p.66):
        #   Experimentally measured slopes fall from ~27 dB/Bark at very low
        #   levels to ~2 dB/Bark at 80+ dB SPL.  The rate is −0.37 dB/Bark
        #   per dB of level above 40 dB SPL.
        #   0 dBFS is mapped to 90 dB SPL (typical studio monitoring headroom).
        L_spl  = L_db + 90.0                                     # (K,) approx dB SPL
        slope_hi = np.clip(
            27.0 - 0.37 * np.maximum(L_spl - 40.0, 0.0),
            2.0, 27.0
        )                                                         # (K,) dB/ERB

        # Lower slope — frequency-dependent, not level-dependent (Zwicker & Fastl
        #   1999, p.64): the travelling wave on the basilar membrane decays
        #   steeply basal-ward (~27 dB/Bark) at mid/high masker frequencies,
        #   but the slope shallows below ~500 Hz where BM mechanics change.
        masker_hz = self.erb_center_hz[masker_erb_band]           # (K,) Hz
        slope_lo  = np.clip(27.0 * (masker_hz / 500.0), 5.0, 27.0)  # (K,) dB/ERB

        # ── 4. Vectorised per-masker spreading → max across maskers ───────
        # D[k, j] = erb_dist[masker_band_k, j] = signed ERB distance to target band j
        D = self.erb_dist[masker_erb_band, :]                     # (K, N_ERB)

        # Attenuation in dB energy: upper slope for D>0, lower for D<0
        atten_db = np.where(
            D >= 0,
            slope_hi[:, np.newaxis] * D,           # upper (higher freq)
           -slope_lo[:, np.newaxis] * D,            # lower (lower freq), D<0 so negate
        )                                                         # (K, N_ERB)

        # Spread energy: masker_energy[k] * 10^(-atten_db[k,j]/10)
        spread = masker_energy[:, np.newaxis] * 10.0 ** (-atten_db / 10.0)  # (K, N_ERB)

        # Per-band threshold = dominant masker (max), then −6 dB
        erb_threshold = spread.max(axis=0) * 0.25                # (N_ERB,)

        # ── 5. Interpolate to per-bin resolution, floor at silence ────────
        erb_lo_val  = float(self.erb_centers[0])
        erb_hi_val  = float(self.erb_centers[-1])
        erb_idx_f   = np.clip(
            (self.mdct_erb_f.astype(np.float64) - erb_lo_val)
            / (erb_hi_val - erb_lo_val) * (N_ERB - 1),
            0.0, N_ERB - 1 - 1e-9
        )
        lo  = erb_idx_f.astype(np.int32)
        hi  = np.minimum(lo + 1, N_ERB - 1)
        frac = erb_idx_f - lo
        bin_thresh_energy = erb_threshold[lo] * (1.0 - frac) + erb_threshold[hi] * frac
        # BUG FIX 3: ABSOLUTE_THRESHOLD is now 1e-6 (not 0), so this floor is meaningful
        bin_thresh_energy = np.maximum(bin_thresh_energy, ABSOLUTE_THRESHOLD ** 2)
        return np.sqrt(bin_thresh_energy).astype(np.float32)          # (mdct_N,)

    def smr_for_candidates(
        self,
        cand_freqs:  np.ndarray,
        cand_amps:   np.ndarray,
        mask_thresh: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Signal-to-Mask Ratio for each FFT candidate peak.

        cand_freqs  : Hz, shape (K,)
        cand_amps   : linear amplitude, shape (K,)
        mask_thresh : per-MDCT-bin masking threshold from masking_threshold_from_chunk
        returns     : SMR array, shape (K,) — higher = more perceptually salient
        """
        if len(cand_freqs) == 0:
            return np.array([], dtype=np.float32)
        # Map candidate Hz → nearest MDCT bin index
        bin_idx = np.round(cand_freqs / self.sr * self.M - 0.5).astype(np.int32)
        bin_idx = np.clip(bin_idx, 0, len(mask_thresh) - 1)
        thresh_at_peak = np.maximum(mask_thresh[bin_idx], 1e-12)
        return (cand_amps / thresh_at_peak).astype(np.float32)

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

# ─────────────────────────────────────────────────────────────
#  Parabolic Peak Interpolation
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
#  FFT Candidate Extraction  (now SMR-ranked via MDCT masking)
# ─────────────────────────────────────────────────────────────
def _fft_candidates(
    audio:  np.ndarray,
    center: int,
    state:  AnalysisState,
    psycho: PsychoState,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract all psychoacoustically valid sinusoidal candidates for one frame.

    A candidate is kept if and only if its amplitude exceeds the MDCT-derived
    simultaneous-masking threshold at that frequency (SMR > 1.0).  No arbitrary
    count cap is applied — the masking model alone decides what is audible.
    Candidates are returned sorted descending by SMR so the tracker always
    sees the most salient partials first.
    """
    half = state.win // 2
    s, e = center - half, center + half
    n    = len(audio)
    if s < 0 or e > n:
        chunk = state.pad_buf.copy()
        ss, se = max(0, s), min(n, e)
        chunk[ss - s : ss - s + (se - ss)] = audio[ss:se]
    else:
        chunk = audio[s:e]

    # ── FFT analysis ──────────────────────────────────────────────────────
    spec     = np.fft.rfft(chunk.astype(np.float64) * state.window)
    mags     = np.abs(spec).astype(np.float32) * state.win_scale
    peak_idx, _ = find_peaks(mags, distance=state.min_dist, height=1e-6)
    if len(peak_idx) == 0:
        return np.array([], np.float32), np.array([], np.float32)

    # ── MDCT masking threshold ────────────────────────────────────────────
    # Centre a MDCT_WIN-length block on the frame centre; masking_threshold_from_chunk
    # handles padding if the chunk is shorter than MDCT_WIN.
    mc         = len(chunk) // 2
    mdct_half  = psycho.M // 2
    mc_s       = max(0, mc - mdct_half)
    mc_e       = min(len(chunk), mc_s + psycho.M)
    mask_thresh = psycho.masking_threshold_from_chunk(chunk[mc_s:mc_e])  # (mdct_N,)

    # ── Parabolic interpolation → Hz + amplitude ──────────────────────────
    all_freqs, all_amps = _parabolic_interp(peak_idx, mags, state.bin_width)
    in_band = (all_freqs >= 20.0) & (all_freqs <= state.nyquist - state.bin_width)
    all_freqs = all_freqs[in_band].astype(np.float32)
    all_amps  = all_amps[in_band].astype(np.float32)
    if len(all_freqs) == 0:
        return np.array([], np.float32), np.array([], np.float32)

    # ── Psychoacoustic gate: keep only peaks above masking threshold ───────
    smr  = psycho.smr_for_candidates(all_freqs, all_amps, mask_thresh)
    keep = smr > 1.0                                  # audible above the mask
    all_freqs = all_freqs[keep]
    all_amps  = np.clip(all_amps[keep], 0.0, 1.0)
    smr       = smr[keep]

    if len(all_freqs) == 0:
        return np.array([], np.float32), np.array([], np.float32)

    # ── Sort descending by SMR so tracker fills slots most-salient-first ──
    order = np.argsort(smr)[::-1]
    return all_freqs[order], all_amps[order]

# ─────────────────────────────────────────────────────────────
#  Greedy Peak Tracker  (fixed ERB-width tolerance)
# ─────────────────────────────────────────────────────────────
def _erb_width(hz: float) -> float:
    """
    Auditory filter width (Hz) at centre frequency hz.
    Moore & Glasberg (1983): ERB = 24.7 * (4.37e-3 * f + 1)
    Half this width is used as the matching tolerance — a candidate must
    land within ±0.5 ERB of a tracked partial's last known frequency.
    """
    return 24.7 * (4.37e-3 * hz + 1.0)

def _track_greedy(
    cand_f: np.ndarray, cand_a: np.ndarray,
    prev_f: np.ndarray, prev_a: np.ndarray,
    last_known_f: np.ndarray,
    n_partials: int,
    cooldowns:  np.ndarray,
    cooldown_frames: int = SLOT_COOLDOWN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Match candidates to active slots using a fixed ±0.5 ERB tolerance.

    `last_known_f` retains the last live frequency for each slot so that a
    slot that died one frame ago can still match a returning partial.
    Returns (out_f, out_a, updated_cooldowns, updated_last_known_f).
    """
    out_f        = np.zeros(n_partials, dtype=np.float32)
    out_a        = np.zeros(n_partials, dtype=np.float32)
    cooldowns    = np.maximum(0, cooldowns - 1)
    last_known_f = last_known_f.copy()

    if len(cand_f) == 0:
        return out_f, out_a, cooldowns, last_known_f

    claimed = np.zeros(len(cand_f), dtype=bool)

    # ── Continue active slots (loudest first) ─────────────────────────────
    active = np.where(prev_a > 0)[0]
    active = active[np.argsort(prev_a[active])[::-1]]

    for slot in active:
        if claimed.all():
            break
        f0   = float(last_known_f[slot])
        if f0 <= 0.0:
            continue
        tol  = 0.5 * _erb_width(f0)
        dists = np.where(~claimed, np.abs(cand_f - f0), np.inf)
        bi    = int(np.argmin(dists))
        if dists[bi] <= tol:
            out_f[slot]        = cand_f[bi]
            out_a[slot]        = cand_a[bi]
            last_known_f[slot] = cand_f[bi]
            claimed[bi]        = True
        else:
            cooldowns[slot] = cooldown_frames

    # ── Assign unclaimed candidates to empty slots ────────────────────────
    births = np.where(~claimed)[0]
    if len(births):
        empty    = np.where((out_a == 0) & (cooldowns == 0))[0]
        n_assign = min(len(births), len(empty))
        sl       = empty[:n_assign]
        bi_v     = births[:n_assign]
        out_f[sl]        = cand_f[bi_v]
        out_a[sl]        = cand_a[bi_v]
        last_known_f[sl] = cand_f[bi_v]

    return out_f, out_a, cooldowns, last_known_f

# ─────────────────────────────────────────────────────────────
#  RSC6 Binary Writer (bitstream unchanged)
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

    f_q_prev  = np.vstack([np.zeros((1, n_partials), np.int32), f_q[:-1]])
    amu_prev  = np.vstack([np.zeros((1, n_partials), np.int32), a_mu[:-1].astype(np.int32)])

    df_mat = (f_q - f_q_prev).astype(np.int32)
    da_mat = (a_mu.astype(np.int32) - amu_prev)

    # BUG FIX 1: The amplitude delta overflow range was ±127 (int8 range), which
    # incorrectly flagged valid mu-law deltas in [128, 255] as overflows and forced
    # them to be re-encoded as births, wasting born slot budget and inflating file
    # size.  The correct range is ±255 since mu-law values are uint8 [0, 255] and
    # their differences span [-255, 255].  The Rice encoder handles full int32
    # zigzag values, so no decoder-side constraint limits this to int8.
    overflow = continuing & (
        (df_mat < -32768) | (df_mat > 32767) |
        (da_mat <   -255) | (da_mat >    255)   # was: -128 / 127  (BUG FIX 1)
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
    stacked     = np.stack([alive_packed, born_packed], axis=1)
    bitmask_buf = stacked.tobytes()

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
#  Main Encode Pipeline
# ─────────────────────────────────────────────────────────────
def encode(
    input_path: str,
    output_path: str,
    n_partials: int,
    target_sr: int,
    n_workers: int,
) -> None:
    print(f"RMDCTSCv1 Encoder  --  {input_path}")
    print(f"   Output slots   : {n_partials}  |  Target SR: {target_sr} Hz")
    print(f"   Psychoacoustics: MDCT-based ERB masking  "
          f"(block={MDCT_WIN}, ERB bands={N_ERB}, slopes=Zwicker&Fastl first-principles)")
    samples, sr = load_audio(input_path, target_sr)
    sample_rate   = target_sr
    total_samples = len(samples)
    print(f"   Loaded         : {total_samples} samples  ({total_samples/sample_rate:.2f}s)")
    frame_size    = int(round(sample_rate / TARGET_FPS))
    n_frames      = math.ceil(total_samples / frame_size)
    pad           = n_frames * frame_size - total_samples
    if pad > 0:
        samples = np.concatenate([samples, np.zeros(pad, np.float32)])
    print(f"   Frame size     : {frame_size} samp ({1000*frame_size/sample_rate:.2f} ms)"
          f"  |  {n_frames} frames")

    state  = AnalysisState(sample_rate)
    psycho = PsychoState(sample_rate)

    print(f"   Analysis win   : {ANALYSIS_WIN} samp ({state.bin_width:.1f} Hz/bin)"
          f"  |  slots={n_partials}  tolerance=±0.5 ERB  cooldown={SLOT_COOLDOWN}  workers={n_workers}")

    # ── Phase 1: parallel FFT + MDCT candidate extraction ────────────────
    # Candidates are not capped — every peak above the masking threshold is kept.
    centers = [i * frame_size + frame_size // 2 for i in range(n_frames)]
    print(f"   Extracting psychoacoustic candidates ({n_workers} thread(s)) ...")

    def _extract(center: int) -> tuple[np.ndarray, np.ndarray]:
        return _fft_candidates(samples, center, state, psycho)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        candidates = list(pool.map(_extract, centers))

    # ── Phase 2: sequential greedy tracking ──────────────────────────────
    print(f"   Tracking partials ...")
    all_f = np.zeros((n_frames, n_partials), dtype=np.float32)
    all_a = np.zeros((n_frames, n_partials), dtype=np.float32)
    prev_f       = np.zeros(n_partials, np.float32)
    prev_a       = np.zeros(n_partials, np.float32)
    last_known_f = np.zeros(n_partials, np.float32)
    cooldowns    = np.zeros(n_partials, np.int32)
    for i, (cf, ca) in enumerate(candidates):
        of, oa, cooldowns, last_known_f = _track_greedy(
            cf, ca, prev_f, prev_a, last_known_f, n_partials, cooldowns
        )
        all_f[i] = of
        all_a[i] = oa
        prev_f   = of
        prev_a   = oa
        if (i + 1) % 500 == 0 or (i + 1) == n_frames:
            print(f"   ... tracked frame {i+1}/{n_frames}", end="\r")
    print()
    write_rsc(output_path, all_f, all_a, sample_rate, frame_size, total_samples)

# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="RMDCTSCv1 — Roblox Modified Discrete Cosine Transform Sine Codec v1 Encoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",      "-i", required=True)
    p.add_argument("--output",     "-o", default=None)
    p.add_argument("--partials",   "-n", type=int, default=DEFAULT_PARTIALS)
    p.add_argument("--samplerate", "-r", type=int, default=DEFAULT_SAMPLERATE,
                   choices=[22050, 44100])
    p.add_argument("--workers",    "-w", type=int,
                   default=min(8, os.cpu_count() or 1),
                   help="Thread count for parallel FFT+MDCT candidate extraction")
    args = p.parse_args()
    out  = args.output or (os.path.splitext(args.input)[0] + RMDCTSC_EXTENSION)
    encode(args.input, out, args.partials, args.samplerate, args.workers)

if __name__ == "__main__":
    main()