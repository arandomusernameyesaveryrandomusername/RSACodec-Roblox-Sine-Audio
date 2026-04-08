from __future__ import annotations

"""
rsc_encoder.py -- Roblox Sine Codec (RSC) Encoder  [SIMPLIFIED]

No tracking, no Gaussian interpolation, no peak-finding.
Just raw FFT bins sorted by amplitude, top-N written per frame.
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
DEFAULT_PARTIALS   = 192
DEFAULT_SAMPLERATE = 44100
RSC_EXTENSION      = ".rsc"
ANALYSIS_WIN       = 1024
ALIVE_THRESHOLD    = 0


# ─────────────────────────────────────────────────────────────────────────────
#  ATH
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
    mu = np.float32(65535.0)
    x  = np.clip(x, 0.0, 1.0)
    return np.clip(np.round(np.log1p(mu * x) / np.log1p(mu) * 65535.0), 0, 65535).astype(np.uint16)


# ─────────────────────────────────────────────────────────────────────────────
#  JIT — Zigzag encode
# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=True)
def _zigzag_njit(arr: np.ndarray) -> np.ndarray:
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
    n = len(vals)
    if n == 0:
        return 0
    best_k    = 0
    best_bits = np.int64(1) << np.int64(62)
    for k in range(17):
        bits = np.int64(n) * np.int64(1 + k)
        for i in range(n):
            bits += np.int64(vals[i]) >> np.int64(k)
        if bits < best_bits:
            best_bits = bits
            best_k    = k
    return best_k


# ─────────────────────────────────────────────────────────────────────────────
#  JIT — Rice encoder
# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=True)
def _rice_encode_njit(vals: np.ndarray, k: int) -> np.ndarray:
    n = len(vals)
    if n == 0:
        return np.empty(0, dtype=np.uint8)

    total_bits = np.int64(0)
    for i in range(n):
        total_bits += (np.int64(vals[i]) >> np.int64(k)) + np.int64(1 + k)

    out = np.zeros(int((total_bits + 7) >> 3), dtype=np.uint8)

    bit_pos = np.int64(0)
    k_mask  = np.int64((1 << k) - 1)

    for i in range(n):
        v = np.int64(vals[i])
        q = int(v >> np.int64(k))
        r = int(v  &  k_mask)

        bit_pos += np.int64(q)
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
#  Raw bin extraction — top-N bins by amplitude, above ATH, in [20Hz, nyquist)
# ─────────────────────────────────────────────────────────────────────────────
def _extract_top_bins(
    all_mags:    np.ndarray,   # float32 (n_frames, n_bins)
    ath_lin:     np.ndarray,   # float32 (n_bins,)
    bin_width:   float,
    nyquist:     float,
    n_partials:  int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each frame: mask bins outside [20 Hz, nyquist), mask bins below ATH,
    then take the top-N by raw magnitude. No peak-finding, no interpolation,
    no tracking — just sorted bins.
    """
    n_frames, n_bins = all_mags.shape
    freqs = np.arange(n_bins, dtype=np.float32) * bin_width

    freq_mask = (freqs >= 20.0) & (freqs < nyquist)

    out_f = np.zeros((n_frames, n_partials), dtype=np.float32)
    out_a = np.zeros((n_frames, n_partials), dtype=np.float32)

    for fi in range(n_frames):
        mags = all_mags[fi]
        mask = freq_mask & (mags > ath_lin)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            continue

        # Sort valid bins by descending magnitude, take top-N
        top = indices[np.argsort(-mags[indices])][:n_partials]
        n   = len(top)
        out_f[fi, :n] = freqs[top]
        out_a[fi, :n] = np.minimum(mags[top], 1.0)

    return out_f, out_a


# ─────────────────────────────────────────────────────────────────────────────
#  Analysis state
# ─────────────────────────────────────────────────────────────────────────────
class AnalysisState:
    def __init__(self, sample_rate: int, analysis_win: int = ANALYSIS_WIN):
        self.win       = analysis_win
        self.sr        = sample_rate
        self.window    = windows.dpss(analysis_win, 3).astype(np.float32)
        self.win_scale = np.float32(1.0 / float(np.sum(self.window)))
        self.bin_width = np.float32(float(sample_rate) / analysis_win)
        self.nyquist   = np.float32(sample_rate / 2.0)
        n_bins         = analysis_win // 2 + 1
        self.ath_lin   = _ath_linear(n_bins, sample_rate, analysis_win)


# ─────────────────────────────────────────────────────────────────────────────
#  Phase A — parallel FFT extraction
# ─────────────────────────────────────────────────────────────────────────────
def _compute_all_spectra(
    audio:   np.ndarray,
    centers: list,
    state:   AnalysisState,
) -> np.ndarray:
    n_frames = len(centers)
    n_bins   = state.win // 2 + 1
    all_mags = np.empty((n_frames, n_bins), dtype=np.float32)
    half     = state.win // 2
    n_audio  = len(audio)
    window   = state.window
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

    cr, cc      = np.where(valid_cont)
    freq_deltas = df_mat[cr, cc].astype(np.int32) if len(cr) else np.array([], np.int32)
    amp_deltas  = da_mat[cr, cc].astype(np.int32) if len(cr) else np.array([], np.int32)
    print(f"   Delta pass  |  {len(br)} births  |  {len(cr)} continuing")

    fd_zz  = _zigzag_njit(freq_deltas)
    ad_zz  = _zigzag_njit(amp_deltas)
    k_freq = _optimal_k_njit(fd_zz)
    k_amp  = _optimal_k_njit(ad_zz)
    print(f"   Rice k_freq={k_freq}  k_amp={k_amp}")

    rice_freq = bytes(_rice_encode_njit(fd_zz, k_freq))
    rice_amp  = bytes(_rice_encode_njit(ad_zz, k_amp))

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
    print(f"RSC Encoder (simplified)  --  {input_path}")
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

    # ── Phase A: parallel FFT ─────────────────────────────────────────────
    centers = [i * frame_size + frame_size // 2 for i in range(n_frames)]
    print(f"   Phase A - parallel FFT ({os.cpu_count()} threads) ...")
    t1 = time.perf_counter()
    all_mags = _compute_all_spectra(samples, centers, state)
    print(f"   Phase A done in {time.perf_counter() - t1:.2f}s")

    # ── Phase B: raw top-N bin extraction ────────────────────────────────
    print(f"   Phase B - extracting top-{n_partials} bins per frame ...")
    t2 = time.perf_counter()
    all_f, all_a = _extract_top_bins(
        all_mags, state.ath_lin,
        float(state.bin_width), float(state.nyquist),
        n_partials,
    )
    print(f"   Phase B done in {time.perf_counter() - t2:.2f}s")
    del all_mags

    # ── Phase C: encode + write ───────────────────────────────────────────
    write_rsc(output_path, all_f, all_a, sr, frame_size, total_samples)

    elapsed = time.perf_counter() - t0
    dur     = total_samples / sr
    print(f"   Done in {elapsed:.2f}s  (RTF {elapsed/dur:.3f}x)")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="RSC6 Encoder (simplified)",
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