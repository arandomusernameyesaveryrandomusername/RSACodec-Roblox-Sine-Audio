from __future__ import annotations

from rsc_decoder import _BitReader
"""
rsc_encoder.py -- Roblox Sine Codec (RSC) Encoder
Format: RSC7 (RSC6 + ERB residual band envelope)

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
TARGET_FPS           = 60
DEFAULT_PARTIALS     = 384
DEFAULT_SAMPLERATE   = 44100
RSC_EXTENSION        = ".rsc"
ANALYSIS_WIN         = 4096
SLOT_COOLDOWN        = 1
MU                   = 255.0
ALIVE_THRESHOLD      = 0
N_RESIDUAL_BANDS     = 24          # ERB bands for residual envelope
TWO_PI               = 2.0 * math.pi
_LOG1P_MU            = math.log1p(MU)

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
def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio at its native sample rate, normalised to [-1, 1]."""
    y, sr = librosa.load(path, sr=None, mono=True)
    peak = np.max(np.abs(y))
    if peak > 1e-9:
        y = y / peak
    return y.astype(np.float32), sr

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
    hfc   = mags**2 * state.erb
    peak_idx, _ = find_peaks(mags, distance=state.min_dist, height=1e-6)
    if len(peak_idx) == 0:
        k_part   = min(n_candidates, len(hfc))
        peak_idx = np.argpartition(hfc, -k_part)[-k_part:]
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
            rem_mask  = (rem_freqs >= 20.0) & (rem_freqs <= state.nyquist - state.bin_width)
            top_freqs = np.concatenate([top_freqs, rem_freqs[rem_mask][:extra_needed].astype(np.float32)])
            top_mags  = np.concatenate([top_mags,  rem_amps[rem_mask][:extra_needed].astype(np.float32)])
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
        sc    = prev_f[slot]
        tol   = max(sc * 0.038, (24.7 + 0.108 * sc) * 0.55)
        dists = np.where(~claimed, np.abs(cand_f - sc), np.inf)
        bi    = int(np.argmin(dists))
        tol  *= 1.25 if cand_f[bi] > sc else 0.85
        eps   = 1e-12
        tol  *= 1 - np.log1p(prev_a[slot] + eps) / np.log1p(9 + eps)
        tol  *= min(2.0, 1.0 + (abs(prev_f[slot] - prevprev_f[slot])
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
#  Fast sine resynthesis  (for residual computation)
#  Phase logic must stay identical to the decoder's synthesize().
# ─────────────────────────────────────────────────────────────
def _synthesize_fast(
    all_f: np.ndarray,
    all_a: np.ndarray,
    frame_size: int,
    sample_rate: int,
) -> np.ndarray:
    n_frames, n_partials = all_f.shape
    output = np.zeros(frame_size * n_frames, dtype=np.float64)
    t      = np.arange(frame_size, dtype=np.float64) / sample_rate
    phi    = np.zeros(n_partials, dtype=np.float64)
    for i in range(n_frames):
        f      = all_f[i].astype(np.float64)
        a      = all_a[i].astype(np.float64)
        active = a > 1e-6
        if active.any():
            fa = f[active]; aa = a[active]; pa = phi[active]
            output[i * frame_size : (i + 1) * frame_size] = (
                aa[:, None] * np.sin(TWO_PI * fa[:, None] * t + pa[:, None])
            ).sum(axis=0)
        phi = (phi + TWO_PI * f * frame_size / sample_rate) % TWO_PI
        phi[~active] = 0.0
        if (i + 1) % 500 == 0 or (i + 1) == n_frames:
            print(f"   ... resynth frame {i+1}/{n_frames}", end="\r")
    print()
    return output.astype(np.float32)

# ─────────────────────────────────────────────────────────────
#  ERB band encoding  (from residual + A-weighting)
# ─────────────────────────────────────────────────────────────
def _erb_band_bins(
    frame_size: int,
    sample_rate: int,
    n_bands: int,
) -> list[tuple[int, int]]:
    """Map FFT bins to n_bands ERB-spaced bands. Returns list of (lo, hi) pairs."""
    freqs   = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    erb     = 21.4 * np.log10(4.37e-3 * np.maximum(freqs, 1e-3) + 1)
    erb_min = erb[1]       # skip DC bin
    erb_max = erb[-1]
    edges   = np.linspace(erb_min, erb_max, n_bands + 1)
    bands: list[tuple[int, int]] = []
    for b in range(n_bands):
        lo = int(np.searchsorted(erb, edges[b]))
        hi = int(np.searchsorted(erb, edges[b + 1]))
        hi = max(hi, lo + 1)
        # Last band must extend all the way to Nyquist —
        # searchsorted returns len(freqs)-1 for erb_max, missing the last bin.
        if b == n_bands - 1:
            hi = len(freqs)
        else:
            hi = min(hi, len(freqs))
        bands.append((lo, hi))
    return bands


def _a_weights_for_bands(
    bands: list[tuple[int, int]],
    frame_size: int,
    sample_rate: int,
) -> np.ndarray:
    """
    Compute normalised A-weighting coefficient for each ERB band.

    Uses the centre frequency of each band derived from the same rfftfreq
    array that was used to build `bands`, so bin indices stay consistent.
    Peak weight is normalised to 1.0.

    A(f) = 12194² f⁴ / [(f²+20.6²) √((f²+107.7²)(f²+737.9²)) (f²+12194²)]
    """
    # Use the same frequency array that _erb_band_bins used — avoids any
    # size mismatch if frame_size produces an odd-length rfft output.
    freqs   = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    weights = np.zeros(len(bands), dtype=np.float64)
    for b, (lo, hi) in enumerate(bands):
        fc  = float(np.mean(freqs[lo:hi]))     # centre freq of this band in Hz
        fc  = max(fc, 10.0)                     # guard against f≈0 on band 0
        f2  = fc * fc
        num = (12194.0 ** 2) * (f2 ** 2)
        den = ((f2 + 20.6  ** 2)
               * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
               * (f2 + 12194.0 ** 2))
        weights[b] = num / den if den > 0 else 0.0

    peak = weights.max()
    if peak > 1e-12:
        weights /= peak
    return weights.astype(np.float32)


def _band_energies_from_audio(
    audio: np.ndarray,
    frame_size: int,
    bands: list[tuple[int, int]],
    a_weights: np.ndarray,
) -> np.ndarray:
    """
    Compute A-weighted ERB band energy envelope of the residual.

    Uses a DPSS (Slepian) window — same family as the analysis window —
    for better sidelobe rejection than Hann.

    Returns (n_frames, n_bands) int16 in [0, 32767], globally normalised
    so the loudest weighted band = 32767.
    """
    n_frames = len(audio) // frame_size
    frames   = audio[:n_frames * frame_size].reshape(n_frames, frame_size).astype(np.float64)
    win      = windows.dpss(frame_size, 4.0, sym=False).astype(np.float64)
    frames   = frames * win[np.newaxis, :]
    specs    = np.fft.rfft(frames, axis=1)
    mags2    = np.abs(specs) ** 2

    n_bands  = len(bands)
    energies = np.zeros((n_frames, n_bands), dtype=np.float64)
    for b, (lo, hi) in enumerate(bands):
        energies[:, b] = mags2[:, lo:hi].sum(axis=1) * float(a_weights[b])

    peak = energies.max()
    if peak > 1e-12:
        energies /= peak

    return np.clip(np.round(energies * 32767), 0, 32767).astype(np.int16)

# ─────────────────────────────────────────────────────────────
#  Residual band Rice encoder
#
#  Delta-encodes each band time-series independently (frame-to-frame),
#  then concatenates all bands into one zigzag+Rice stream.
#  This exploits temporal smoothness of band envelopes.
# ─────────────────────────────────────────────────────────────
def _encode_residual_bands(band_energies: np.ndarray) -> tuple[bytearray, int]:
    """
    band_energies : (n_frames, n_bands) int16 [0, 32767]
    Returns (rice_bytes, k_residual).
    Delta is along the frame axis per band, concatenated band-by-band.
    """
    n_frames, n_bands = band_energies.shape
    vals = band_energies.astype(np.int32)           # (n_frames, n_bands)
    # Delta along time axis: first row stored absolute, rest are diffs
    deltas = np.empty_like(vals)
    deltas[0, :]  = vals[0, :]                      # absolute first frame
    deltas[1:, :] = vals[1:, :] - vals[:-1, :]      # frame-to-frame diff
    # Concatenate band by band (each band's full time series in order)
    flat   = deltas.T.ravel().astype(np.int32)       # (n_bands * n_frames,)
    zz     = _zigzag(flat)
    k      = _optimal_k(zz)
    return _rice_encode(zz, k), k


def _decode_residual_bands_check(rice_bytes: bytearray, k: int,
                                  n_frames: int, n_bands: int) -> np.ndarray:
    """Round-trip check only — not used in production encoder."""
    from io import BytesIO
    reader = _BitReader(bytes(rice_bytes), 0)
    flat   = np.array([reader.read_rice(k) for _ in range(n_bands * n_frames)], dtype=np.int32)
    deltas = flat.reshape(n_bands, n_frames)
    vals   = np.cumsum(deltas, axis=1).T             # (n_frames, n_bands)
    return np.clip(vals, 0, 32767).astype(np.int16)


# ─────────────────────────────────────────────────────────────
#  RSC7 Binary Writer
#
#  Header (40 bytes):
#    "RSC7" | u8 ver | u32 sr | u32 frame_sz | u16 n_partials |
#    u32 total_samples | u32 total_frames | u16 mask_sz |
#    u8 k_freq | u8 k_amp | u32 born_data_sz | u32 rice_freq_sz |
#    u8 n_bands | u32 residual_sz
#
#  Section 1 — Bitmasks  : nF * 2 * mask_sz  bytes
#  Section 2 — Born data : born_data_sz       bytes  (uint16 fq + uint8 amu)
#  Section 3 — Rice freq : rice_freq_sz       bytes
#  Section 4 — Rice amp  : rice_amp_sz        bytes
#  Section 5 — Residual  : residual_sz        bytes  (nF * n_bands uint8, row-major)
# ─────────────────────────────────────────────────────────────
def write_rsc(
    path: str,
    frame_freqs:   np.ndarray,
    frame_amps:    np.ndarray,
    band_energies: np.ndarray,       # (n_frames, n_bands) int16
    sample_rate:   int,
    frame_size:    int,
    total_samples: int,
) -> None:
    n_frames, n_partials = frame_freqs.shape
    n_bands  = band_energies.shape[1]
    mask_sz  = (n_partials + 7) // 8

    freq_scale = 65535.0 / (sample_rate / 2.0)
    f_q  = np.clip(np.round(frame_freqs * freq_scale), 0, 65535).astype(np.int32)
    a_mu = _mulaw_encode(frame_amps)

    alive      = frame_amps > ALIVE_THRESHOLD
    was_alive  = np.vstack([np.zeros((1, n_partials), bool), alive[:-1]])
    nat_born   = alive & ~was_alive
    continuing = alive & was_alive

    f_q_prev  = np.vstack([np.zeros((1, n_partials), np.int32), f_q[:-1]])
    amu_prev  = np.vstack([np.zeros((1, n_partials), np.int32), a_mu[:-1].astype(np.int32)])
    df_mat    = (f_q - f_q_prev).astype(np.int32)
    da_mat    = (a_mu.astype(np.int32) - amu_prev)

    overflow = continuing & (
        (df_mat < -32768) | (df_mat > 32767) |
        (da_mat <   -128) | (da_mat >    127)
    )
    born_bits_mat = nat_born | overflow
    valid_cont    = continuing & ~overflow

    # ── Bitmasks ──────────────────────────────────────────────────────
    pad_w     = mask_sz * 8
    alive_pad = np.zeros((n_frames, pad_w), np.uint8)
    born_pad  = np.zeros((n_frames, pad_w), np.uint8)
    alive_pad[:, :n_partials] = alive
    born_pad [:, :n_partials] = born_bits_mat
    alive_packed = np.packbits(alive_pad, axis=1, bitorder="little")
    born_packed  = np.packbits(born_pad,  axis=1, bitorder="little")
    bitmask_buf  = np.stack([alive_packed, born_packed], axis=1).tobytes()

    # ── Born buffer ───────────────────────────────────────────────────
    br, bc = np.where(born_bits_mat)
    if len(br):
        bfq  = f_q[br, bc].astype(np.uint16)
        bamu = a_mu[br, bc]
        raw  = np.empty(len(br) * 3, np.uint8)
        raw[0::3] = (bfq & 0xFF).astype(np.uint8)
        raw[1::3] = (bfq >> 8).astype(np.uint8)
        raw[2::3] = bamu
        born_buf = raw.tobytes()
    else:
        born_buf = b""

    # ── Delta streams ─────────────────────────────────────────────────
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

    # ── Residual section: delta + zigzag + Rice ──────────────────────
    residual_buf, k_residual = _encode_residual_bands(band_energies)

    # ── Header (41 bytes) ─────────────────────────────────────────────
    #   "RSC7" | u8 ver | u32 sr | u32 frame_sz | u16 n_partials |
    #   u32 total_samples | u32 total_frames | u16 mask_sz |
    #   u8 k_freq | u8 k_amp | u32 born_data_sz | u32 rice_freq_sz |
    #   u8 n_bands | u8 k_residual | u32 residual_sz
    born_data_sz = len(born_buf)
    rice_freq_sz = len(rice_freq)
    residual_sz  = len(residual_buf)

    header = struct.pack(
        "<4sBIIHIIHBBIIBBI",
        b"RSC7", 7,
        sample_rate, frame_size, n_partials,
        total_samples, n_frames,
        mask_sz, k_freq, k_amp,
        born_data_sz, rice_freq_sz,
        n_bands, k_residual, residual_sz,
    )
    assert len(header) == 41, f"Header size wrong: {len(header)}"

    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(bitmask_buf)
        fh.write(born_buf)
        fh.write(rice_freq)
        fh.write(rice_amp)
        fh.write(residual_buf)

    total_sz = 41 + len(bitmask_buf) + born_data_sz + rice_freq_sz + len(rice_amp) + residual_sz
    kb       = total_sz / 1024
    res_kb   = residual_sz / 1024
    print(f"  ✅ Wrote {n_frames} frames → {path}")
    print(f"     {kb:.1f} KB total  ({kb/60:.2f} KB/s avg)")
    print(f"     Bitmasks {len(bitmask_buf)/1024:.1f} KB  |  Born {born_data_sz/1024:.1f} KB"
          f"  |  Rice-freq {rice_freq_sz/1024:.1f} KB  |  Rice-amp {len(rice_amp)/1024:.1f} KB"
          f"  |  Residual {res_kb:.1f} KB ({n_bands} bands, k={k_residual})")

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

    samples, native_sr = load_audio(input_path)
    print(f"   Native SR      : {native_sr} Hz  |  {len(samples)} samples  "
          f"({len(samples)/native_sr:.2f}s)")

    if native_sr != target_sr:
        print(f"   Resampling {native_sr} → {target_sr} Hz ...")
        from math import gcd
        from scipy.signal import resample_poly
        g       = gcd(target_sr, native_sr)
        samples = resample_poly(samples, target_sr // g, native_sr // g).astype(np.float32)
        # Re-normalise after resample — poly resampling can push peaks slightly above 1.0
        peak = np.max(np.abs(samples))
        if peak > 1e-9:
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
        print(f"   ⚠  Clamping partials {n_partials} → {max_meaningful}")
        n_partials = max_meaningful
    n_cand = n_partials

    print(f"   Analysis win   : {ANALYSIS_WIN} samp ({state.bin_width:.1f} Hz/bin)"
          f"  |  n_cand={n_cand}  cooldown={SLOT_COOLDOWN}  workers={n_workers}")

    # ── Phase 1: parallel FFT candidate extraction ────────────────────
    centers = [i * frame_size + frame_size // 2 for i in range(n_frames)]
    print(f"   Extracting FFT candidates ({n_workers} thread(s)) ...")
    def _extract(center: int) -> tuple[np.ndarray, np.ndarray]:
        return _fft_candidates(samples, center, state, n_cand)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        candidates = list(pool.map(_extract, centers))

    # ── Phase 2: sequential greedy tracking ───────────────────────────
    print(f"   Tracking partials ...")
    all_f      = np.zeros((n_frames, n_partials), dtype=np.float32)
    all_a      = np.zeros((n_frames, n_partials), dtype=np.float32)
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

    # ── Phase 3: residual = original − sines, encode band envelope ───────
    print(f"   Resynthesizing sines ...")
    sine_sum  = _synthesize_fast(all_f, all_a, frame_size, sample_rate)
    # Match scales before subtracting — sine_sum is normalised to its own peak,
    # samples is normalised to [-1, 1]. Align sine peak to sample peak so the
    # subtraction actually cancels the tonal content.
    sine_peak = np.max(np.abs(sine_sum))
    if sine_peak > 1e-9:
        sine_sum *= (np.max(np.abs(samples[:len(sine_sum)])) / sine_peak)
    residual  = samples[:len(sine_sum)] - sine_sum

    print(f"   Computing A-weighted ERB residual envelope ({N_RESIDUAL_BANDS} bands) ...")
    bands         = _erb_band_bins(frame_size, sample_rate, N_RESIDUAL_BANDS)
    a_weights     = _a_weights_for_bands(bands, frame_size, sample_rate)
    band_energies = _band_energies_from_audio(residual, frame_size, bands, a_weights)

    write_rsc(output_path, all_f, all_a, band_energies,
              sample_rate, frame_size, total_samples)

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