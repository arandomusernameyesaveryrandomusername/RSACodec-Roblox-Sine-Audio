from __future__ import annotations

"""
rmdctsc_encoder_minimal.py -- RMDCTSCv1 Minimal Encoder  (Full-Partials edition)

Bare-minimum implementation: Hann-windowed FFT top-N peak picking +
simple greedy frequency tracker + ATH (Absolute Threshold of Hearing)
suppression for bit savings.  Same RSC6 bitstream as the full encoder.

Usage:
    python rsc_encoder_ath.py --input audio.wav --output audio.rsc
    python rsc_encoder_ath.py --input audio.wav --output audio.rsc \\
        --partials 384 --samplerate 44100 --ath-gain 6
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
#  Absolute Threshold of Hearing
# ─────────────────────────────────────────────────────────────

def _ath_db(freq: np.ndarray) -> np.ndarray:
    """
    Terhardt (1979) approximation of the ISO quiet-listening threshold.
    Returns dB SPL at each frequency (Hz). Values are in the range
    ~0 dB (1–4 kHz) up to ~80 dB near DC and the Nyquist edge.

    Formula:
        ATH(f) =  3.64*(f/1000)**-0.8
                - 6.5 * exp(-0.6*(f/1000 - 3.3)**2)
                + 10**-3 * (f/1000)**4
    Clipped to [-10, 90] dB for numerical safety.
    """
    f   = np.asarray(freq, dtype=np.float64)
    f   = np.maximum(f, 20.0)          # avoid log/power issues near DC
    fk  = f / 1000.0
    ath = (
          3.64  * fk ** -0.8
        - 6.5   * np.exp(-0.6 * (fk - 3.3) ** 2)
        + 1e-3  * fk ** 4
    )
    return np.clip(ath, -90.0, 90.0)


def _ath_linear(n_bins: int, sample_rate: int, win: int,
                ath_gain_db: float = 0.0) -> np.ndarray:
    """
    Pre-compute a per-FFT-bin ATH amplitude threshold (linear, 0-1 scale).

    The encoder normalises audio to peak=1, which we treat as 0 dB FS.
    We map dB SPL → dB FS with the rough assumption that 0 dB FS ≈ 96 dB SPL
    (standard for 16-bit audio), then convert to linear amplitude:

        amp_threshold = 10 ** ((ath_db_spl - 96 + ath_gain_db) / 20)

    ath_gain_db > 0  → more aggressive (raise threshold, drop more partials)
    ath_gain_db < 0  → more permissive (lower threshold, keep more partials)
    """
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
#  FFT candidate extraction  (Hann window, top-N by SNR score)
#  Candidates are ranked by perceptual salience:
#      snr_ratio = amp / ATH_threshold   (how far above hearing floor)
#      score     = amp * log10(snr_ratio + 1)
#  This prioritises partials that are both loud AND well above the
#  hearing threshold over partials that are merely loud.
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
    ath_lin: np.ndarray,          # per-bin ATH thresholds
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

    # ── ATH gate disabled: keep all bins so we can fill all partial slots ──
    n_bins   = len(mags)
    ath_bins = ath_lin[:n_bins]
    # mags = np.where(mags >= ath_bins, mags, 0.0)  # disabled for full slot usage
    # ───────────────────────────────────────────────────────────────────────

    peak_idx, _ = find_peaks(mags, distance=min_dist, height=1e-9)  # lowered for full slot fill
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
    peak_bins = peak_idx[in_band]

    # ── Top-up: if fewer candidates than slots, pad with highest-amplitude
    #    non-peak bins so the tracker always has enough to fill all slots ──
    MIN_CANDIDATES = int(nyquist / bin_width)  # use all in-band bins if needed
    if len(freqs) < MIN_CANDIDATES:
        peak_bin_set = set(peak_bins.tolist())
        all_bins     = np.arange(1, len(mags) - 1)  # skip DC and Nyquist
        extra_mask   = np.array([b not in peak_bin_set for b in all_bins])
        extra_bins   = all_bins[extra_mask]
        extra_freqs  = extra_bins * bin_width
        in_band_mask = (extra_freqs >= 20.0) & (extra_freqs <= nyquist - bin_width)
        extra_bins   = extra_bins[in_band_mask]
        extra_freqs  = extra_freqs[in_band_mask].astype(np.float32)
        extra_amps   = np.clip(mags[extra_bins], 0.0, 1.0)
        # sort extra by amplitude descending, take only as many as needed
        n_need  = MIN_CANDIDATES - len(freqs)
        top_idx = np.argsort(extra_amps)[::-1][:n_need]
        freqs   = np.concatenate([freqs, extra_freqs[top_idx]])
        amps    = np.concatenate([amps,  extra_amps[top_idx]])
        peak_bins = np.concatenate([peak_bins, extra_bins[top_idx]])
    # ─────────────────────────────────────────────────────────────────

    # ── SNR-based perceptual sort ─────────────────────────────────────
    ath_at_peaks = ath_bins[np.clip(peak_bins, 0, len(ath_bins) - 1)].astype(np.float32)
    snr_ratio    = amps / np.maximum(ath_at_peaks, 1e-12)
    score        = amps * np.log10(snr_ratio + 1.0)
    order        = np.argsort(score)[::-1]
    return freqs[order], amps[order]
    # ─────────────────────────────────────────────────────────────────

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
        tol = 652938475983
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

    # Active-partial statistics (partials that survived the ATH gate each frame)
    active_per_frame = alive.sum(axis=1)          # shape: (n_frames,)
    avg_active = float(active_per_frame.mean())
    min_active = int(active_per_frame.min())
    max_active = int(active_per_frame.max())
    utilisation = 100.0 * avg_active / n_partials

    print(f"  ✅ Wrote {n_frames} frames → {path}")
    print(f"     {kb:.1f} KB  ({saving4:.1f}% smaller than RSC4  {kb/60:.2f} KB/s avg)")
    print(f"     Bitmasks {len(bitmask_buf)/1024:.1f} KB  |  Born {born_data_sz/1024:.1f} KB"
          f"  |  Rice-freq {rice_freq_sz/1024:.1f} KB  |  Rice-amp {len(rice_amp)/1024:.1f} KB")
    print(f"     Active partials/frame  avg {avg_active:.1f}  min {min_active}  max {max_active}"
          f"  ({utilisation:.1f}% of {n_partials} slots used)")

# ─────────────────────────────────────────────────────────────
#  Main encode pipeline
# ─────────────────────────────────────────────────────────────
def encode(
    input_path: str,
    output_path: str,
    n_partials: int,
    target_sr: int,
    ath_gain_db: float = 0.0,
) -> None:
    print(f"RMDCTSCv1 Minimal Encoder  (ATH edition)  --  {input_path}")
    print(f"   Output slots : {n_partials}  |  Target SR: {target_sr} Hz"
          f"  |  ATH gain: {ath_gain_db:+.1f} dB")

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
    hann      = windows.hann(win).astype(np.float64)
    win_scale = 1.0 / float(np.sum(hann))
    bin_width = float(target_sr) / win
    nyquist   = target_sr / 2.0
    min_dist  = max(1, int(round(10.0 / bin_width)))  # tighter spacing → more candidates for full slot fill

    # ── Pre-compute ATH threshold array (once, outside the frame loop) ──
    n_bins  = win // 2 + 1
    ath_lin = _ath_linear(n_bins, target_sr, win, ath_gain_db)
    suppressed_count = int(np.sum(ath_lin >= win_scale))   # diagnostic only
    print(f"   ATH          : {suppressed_count}/{n_bins} bins at or above threshold"
          f"  ({100*suppressed_count/n_bins:.1f}% of spectrum pre-gated)")
    # ────────────────────────────────────────────────────────────────────

    centers = [i * frame_size + frame_size // 2 for i in range(n_frames)]

    # Phase 1: FFT candidate extraction
    candidates = []
    for center in tqdm(centers, desc="   Analysing    ", unit="frame", dynamic_ncols=True,
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} frames  [{elapsed}<{remaining}  {rate_fmt}]"):
        cf, ca = _fft_candidates(
            samples, center, win, hann, win_scale,
            bin_width, nyquist, min_dist,
            ath_lin,
        )
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

    # ── Post-tracking ATH gate ────────────────────────────────────────────
    # Slots are already fully packed by the tracker. Now silence any partial
    # whose amplitude is below the ATH floor at its frequency, so the
    # bitstream doesn't waste bits on inaudible content — without starving
    # the tracker of candidates in the first place.
    freq_scale_inv = (target_sr / 2.0) / 65535.0          # quantised → Hz
    for fi in range(n_frames):
        freqs_row = all_f[fi]                              # (n_partials,)
        amps_row  = all_a[fi]
        active    = amps_row > 0
        if not active.any():
            continue
        # map each partial's frequency to its nearest FFT bin
        bins = np.clip(
            np.round(freqs_row[active] / bin_width).astype(np.int32),
            0, len(ath_lin) - 1,
        )
        ath_at_partial = ath_lin[bins]                     # ATH amplitude at each partial's freq
        below_ath      = amps_row[active] < ath_at_partial
        idx            = np.where(active)[0][below_ath]
        all_a[fi, idx] = 0.0                               # silence — slot stays assigned
    # ─────────────────────────────────────────────────────────────────────

    write_rsc(output_path, all_f, all_a, target_sr, frame_size, total_samples)

# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="RMDCTSCv1 Minimal (ATH edition) — Hann FFT + ATH gate + greedy tracker, RSC6 bitstream",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",      "-i", required=True)
    p.add_argument("--output",     "-o", default=None)
    p.add_argument("--partials",   "-n", type=int,   default=DEFAULT_PARTIALS)
    p.add_argument("--samplerate", "-r", type=int,   default=DEFAULT_SAMPLERATE,
                   choices=[22050, 44100])
    p.add_argument(
        "--ath-gain", "-g", type=float, default=0.0, metavar="DB",
        help=(
            "Shift the ATH curve up (+) or down (-) in dB. "
            "+6 dB drops ~2× more partials and saves more bits; "
            "-6 dB is more conservative. Default: 0 (pure ISO threshold)."
        ),
    )
    args = p.parse_args()
    out  = args.output or (os.path.splitext(args.input)[0] + RMDCTSC_EXTENSION)
    encode(args.input, out, args.partials, args.samplerate, args.ath_gain)

if __name__ == "__main__":
    main()