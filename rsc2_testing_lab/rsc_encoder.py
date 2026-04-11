#!/usr/bin/env python3
"""
RSC2 Audio Codec - Encoder  (v2)
Sine-wave based codec: DPSS window, 1024-pt FFT,
local-crest + curvature scoring, up to 192 partials/frame.

Header (after 4-byte magic):
  version(u8) channels(u8) sample_rate(u32be) fft_size(u16be)
  hop_size(u16be) max_partials(u16be) n_frames(u32be)
  n_samples(u32be) window_sum(f32be)

Per channel x frame:
  n_partials(u16be) [ bin(u16be) amp(f32be) phase(f32be) ] ...
"""

import numpy as np, struct, wave, argparse
from scipy.signal.windows import dpss

RSC2_MAGIC   = b"RSC2"
RSC2_VERSION = 1
FFT_SIZE     = 1024
HOP_SIZE     = FFT_SIZE // 4
MAX_PARTIALS = 192
SCORE_N      = 8
SCORE_HOLE   = 3
DPSS_NW      = 3.5
HEADER_FMT   = ">BBIHHHIIf"

def compute_scores(mags):
    n = len(mags)
    score = np.zeros(n, dtype=np.float32)
    for b in range(n):
        b_lo = max(0, b - SCORE_N)
        b_hi = min(n - 1, b + SCORE_N)
        lsum = np.float32(0.0)
        for k in range(b_lo, b_hi + 1):
            if k < (b - SCORE_HOLE) or k > (b + SCORE_HOLE):
                lsum += mags[k]
        hole_actual = min(b_hi, b+SCORE_HOLE) - max(b_lo, b-SCORE_HOLE) + 1
        lmean = lsum / np.float32(b_hi - b_lo + 1 - hole_actual + 1e-12)
        log_peak  = np.log(mags[b]  + 1e-12)
        log_floor = np.log(lmean    + 1e-12)
        t3 = max(0.0, log_peak - log_floor)
        if 0 < b < n - 1:
            t5 = max(log_peak - (np.log(mags[b-1]+1e-12)+np.log(mags[b+1]+1e-12))*0.5,
                     np.float32(0.0))
        else:
            t5 = np.float32(0.0)
        score[b] = np.log1p(mags[b]) + t3 + t5
    return score

def encode(pcm, sample_rate, n_channels):
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    n_samples, channels = pcm.shape
    tapers, _ = dpss(FFT_SIZE, DPSS_NW, Kmax=1, sym=True, return_ratios=True)
    win = tapers[0].astype(np.float32)
    win_sum = float(win.sum())
    all_frames = []
    for ch in range(channels):
        sig = pcm[:, ch].astype(np.float32)
        frames = []
        pos = 0
        while pos + FFT_SIZE <= n_samples:
            spec  = np.fft.rfft(sig[pos:pos+FFT_SIZE] * win, n=FFT_SIZE)
            mags  = np.abs(spec).astype(np.float32)
            phs   = np.angle(spec).astype(np.float32)
            sc    = compute_scores(mags)
            k     = min(MAX_PARTIALS, len(mags))
            idx   = np.argpartition(sc, -k)[-k:]
            idx   = idx[np.argsort(sc[idx])[::-1]]
            frames.append([(int(b), float(mags[b]), float(phs[b])) for b in idx])
            pos  += HOP_SIZE
        all_frames.append(frames)
    buf = bytearray(RSC2_MAGIC)
    buf += struct.pack(HEADER_FMT, RSC2_VERSION, channels, sample_rate,
                       FFT_SIZE, HOP_SIZE, MAX_PARTIALS,
                       len(all_frames[0]), n_samples, win_sum)
    for ch in range(channels):
        for fp in all_frames[ch]:
            buf += struct.pack(">H", len(fp))
            for b, a, p in fp:
                buf += struct.pack(">Hff", b, a, p)
    return bytes(buf)

def load_wav(path):
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
    ap = argparse.ArgumentParser(description="RSC2 encoder")
    ap.add_argument("input"); ap.add_argument("output")
    a = ap.parse_args()
    print(f"🎵 Loading {a.input}…")
    pcm, sr, ch = load_wav(a.input)
    print(f"   {sr} Hz | {ch} ch | {len(pcm)} samples")
    print(f"⚙️  Encoding (FFT={FFT_SIZE}, hop={HOP_SIZE}, max_partials={MAX_PARTIALS})…")
    data = encode(pcm, sr, ch)
    with open(a.output, "wb") as f:
        f.write(data)
    print(f"✅ Written {len(data):,} bytes → {a.output}")

if __name__ == "__main__":
    main()