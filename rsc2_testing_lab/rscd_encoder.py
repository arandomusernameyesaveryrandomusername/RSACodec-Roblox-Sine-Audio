#!/usr/bin/env python3
"""
RSC2 → Desmos Encoder
Encodes a WAV file using the RSC2 spectral codec, then outputs a JSON file
containing all partial data ready for the Desmos preprocessor.

Output JSON schema:
{
  "sample_rate": int,
  "n_channels": int,
  "n_frames": int,
  "fft_size": int,
  "hop_size": int,
  "max_partials": int,
  "duration_sec": float,
  "channels": [
    {
      "frames": [
        {
          "frame_idx": int,
          "time_sec": float,
          "peak": float,
          "partials": [
            { "bin": int, "freq_hz": float, "amp_norm": float, "phase_rad": float }
          ]
        }
      ]
    }
  ]
}
"""

import numpy as np
import struct
import wave
import argparse
import json
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from numpy.fft import rfft

# ── Try Numba JIT ─────────────────────────────────────────────────────────────
try:
    from numba import njit as _njit

    @_njit(cache=True, fastmath=True)
    def _score_numba(mags, score_n, score_hole):
        n = len(mags)
        score = np.empty(n, dtype=np.float32)
        for b in range(n):
            b_lo = max(0, b - score_n)
            b_hi = min(n - 1, b + score_n)
            lsum = np.float32(0.0)
            count = 0
            for k in range(b_lo, b_hi + 1):
                if k < (b - score_hole) or k > (b + score_hole):
                    lsum += mags[k]
                    count += 1
            lmean = lsum / (count + np.float32(1e-12))
            log_peak  = np.log(mags[b]  + np.float32(1e-12))
            log_floor = np.log(lmean    + np.float32(1e-12))
            t3 = log_peak - log_floor
            if t3 < 0.0:
                t3 = np.float32(0.0)
            if 0 < b < n - 1:
                t5 = log_peak - (np.log(mags[b-1] + np.float32(1e-12))
                                 + np.log(mags[b+1] + np.float32(1e-12))) * np.float32(0.5)
                if t5 < 0.0:
                    t5 = np.float32(0.0)
            else:
                t5 = np.float32(0.0)
            score[b] = mags[b] * (t3 + t5)
            if (score[b] > score[b - 1] and score[b] > score[b + 1]
                    and score[b] > np.float32(1e-12)):
                score[b-1] = np.float32(0.0)
                score[b+1] = np.float32(0.0)
        return score

    _dummy = np.ones(513, dtype=np.float32)
    _score_numba(_dummy, 8, 3)
    _USE_NUMBA = True
    print("⚡ Numba JIT enabled")

except Exception:
    _USE_NUMBA = False


def _score_numpy(mags: np.ndarray, score_n: int, score_hole: int) -> np.ndarray:
    n     = len(mags)
    b_idx = np.arange(n)
    pad   = score_n
    mpad  = np.pad(mags.astype(np.float64), pad, mode='constant')
    cs    = np.concatenate(([0.0], np.cumsum(mpad)))
    b_lo  = np.maximum(0, b_idx - score_n)
    b_hi  = np.minimum(n - 1, b_idx + score_n)
    win_sum = cs[b_hi + pad + 1] - cs[b_lo + pad]
    win_cnt = b_hi - b_lo + 1
    h_lo     = np.maximum(b_lo, b_idx - score_hole)
    h_hi     = np.minimum(b_hi, b_idx + score_hole)
    has_hole = h_lo <= h_hi
    hole_sum = np.where(has_hole, cs[h_hi + pad + 1] - cs[h_lo + pad], 0.0)
    hole_cnt = np.where(has_hole, h_hi - h_lo + 1, 0)
    lmean    = (win_sum - hole_sum) / np.maximum(win_cnt - hole_cnt, 1).astype(np.float64)
    log_peak  = np.log(mags.astype(np.float64) + 1e-12)
    log_floor = np.log(lmean + 1e-12)
    t3 = np.maximum(0.0, log_peak - log_floor)
    t5 = np.zeros(n, dtype=np.float64)
    if n > 2:
        inner = slice(1, n - 1)
        t5[inner] = np.maximum(0.0, log_peak[inner] - 0.5 * (log_peak[:-2] + log_peak[2:]))
    return ((t3 + t5) * mags).astype(np.float32)


def compute_scores(mags, score_n=6, score_hole=2):
    if _USE_NUMBA:
        return _score_numba(mags, score_n, score_hole)
    return _score_numpy(mags, score_n, score_hole)


# ── Codec constants ───────────────────────────────────────────────────────────
FFT_SIZE     = 2048
HOP_SIZE     = 2048
MAX_PARTIALS = 3
SCORE_N      = 6
SCORE_HOLE   = 2
_TWO_PI      = 2.0 * np.pi
_U16_MAX     = 65535.0


def _encode_channel_to_frames(sig, sample_rate, max_partials=MAX_PARTIALS):
    """Encode one channel, returning rich frame dicts for JSON output."""
    n_samples = len(sig)
    n_bins    = FFT_SIZE // 2 + 1
    n_frames  = max(0, (n_samples - FFT_SIZE) // HOP_SIZE + 1)

    win = np.ones(FFT_SIZE, dtype=np.float32)  # rectangular window

    frame_shape   = (n_frames, FFT_SIZE)
    frame_strides = (sig.strides[0] * HOP_SIZE, sig.strides[0])
    frames_raw    = np.lib.stride_tricks.as_strided(sig, shape=frame_shape, strides=frame_strides)

    spec = rfft(frames_raw, n=FFT_SIZE, axis=1)
    mags = np.abs(spec).astype(np.float32)
    phs  = np.angle(spec).astype(np.float32)

    freq_per_bin = sample_rate / FFT_SIZE
    k = min(max_partials, n_bins)

    frames_out = []
    for fi in range(n_frames):
        sc  = compute_scores(mags[fi], SCORE_N, SCORE_HOLE)
        idx = np.argsort(sc)[-k:][::-1]

        sel_mags = mags[fi][idx]
        peak     = float(sel_mags.max()) if len(sel_mags) else 0.0
        if peak < 1e-12:
            peak = 1.0

        amp_norm  = (sel_mags / peak).tolist()
        phase_rad = phs[fi][idx].tolist()
        freq_hz   = (idx.astype(float) * freq_per_bin).tolist()
        bins      = idx.tolist()

        partials = [
            {
                "bin":       int(b),
                "freq_hz":   round(float(f), 4),
                "amp_norm":  round(float(a), 6),
                "phase_rad": round(float(p), 6),
            }
            for b, f, a, p in zip(bins, freq_hz, amp_norm, phase_rad)
        ]

        frames_out.append({
            "frame_idx": fi,
            "time_sec":  round(fi * HOP_SIZE / sample_rate, 6),
            "peak":      round(peak, 8),
            "partials":  partials,
        })

    return frames_out, n_frames


def load_wav(path: str):
    with wave.open(path, "rb") as wf:
        sr, ch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw == 2:
        s = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        s = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width {sw}")
    return s.reshape(-1, ch), sr, ch


def main():
    ap = argparse.ArgumentParser(description="RSC2 → Desmos JSON encoder")
    ap.add_argument("input",           help="Input .wav file")
    ap.add_argument("output",          help="Output .json file")
    ap.add_argument("--max-partials",  type=int, default=MAX_PARTIALS,
                    help=f"Max partials per frame (default {MAX_PARTIALS})")
    ap.add_argument("--max-frames",    type=int, default=None,
                    help="Truncate to N frames (useful for Desmos import size limits)")
    ap.add_argument("--no-numba",      action="store_true")
    a = ap.parse_args()

    if a.no_numba:
        global _USE_NUMBA
        _USE_NUMBA = False

    print(f"🎵 Loading  {a.input} …")
    pcm, sr, n_ch = load_wav(a.input)
    n_samples      = len(pcm)
    duration       = n_samples / sr
    print(f"   {sr} Hz | {n_ch} ch | {n_samples} samples | {duration:.2f}s")

    channels_out = []
    t0 = time.perf_counter()

    for ch in range(n_ch):
        print(f"⚙️  Encoding channel {ch+1}/{n_ch} …")
        sig    = pcm[:, ch].astype(np.float32)
        frames, n_frames = _encode_channel_to_frames(sig, sr, a.max_partials)
        if a.max_frames:
            frames = frames[:a.max_frames]
        channels_out.append({"frames": frames})
        print(f"   → {len(frames)} frames")

    dt = time.perf_counter() - t0

    out = {
        "sample_rate":  sr,
        "n_channels":   n_ch,
        "n_frames":     len(channels_out[0]["frames"]),
        "fft_size":     FFT_SIZE,
        "hop_size":     HOP_SIZE,
        "max_partials": a.max_partials,
        "duration_sec": round(duration, 6),
        "channels":     channels_out,
    }

    print(f"💾 Writing {a.output} …")
    with open(a.output, "w") as f:
        json.dump(out, f, separators=(",", ":"))

    size_kb = len(json.dumps(out)) / 1024
    print(f"✅ Done in {dt*1000:.1f}ms — {size_kb:.1f} KB")
    print(f"   Frames: {out['n_frames']}  |  Max partials: {a.max_partials}")
    print(f"   → Pass this JSON to  rsc2_desmos_prep.py  next!")


if __name__ == "__main__":
    main()