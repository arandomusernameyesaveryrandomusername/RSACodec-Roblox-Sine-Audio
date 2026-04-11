#!/usr/bin/env python3
"""
RSC2 Audio Codec - Decoder
Reconstructs audio by IRFFT on selected partials + overlap-add.

The encoder stored:
  raw FFT magnitudes (from a DPSS-windowed frame)
  window_sum in the header for amplitude correction

Reconstruction per frame:
  1. Fill spectrum[bin] = amp * exp(j*phase) for each partial
  2. IRFFT → time-domain frame
  3. Multiply by correction = FFT_SIZE / window_sum  (undo DPSS amplitude scale)
  4. OLA with rectangular accumulation, then divide by overlap count
"""

import numpy as np, struct, wave, argparse

RSC2_MAGIC  = b"RSC2"
HEADER_FMT  = ">BBIHHHIIf"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

def decode(data):
    if data[:4] != RSC2_MAGIC:
        raise ValueError(f"Bad magic {data[:4]!r}")
    off = 4
    (version, channels, sample_rate,
     fft_size, hop_size, max_partials,
     n_frames, n_samples, window_sum) = struct.unpack_from(HEADER_FMT, data, off)
    off += HEADER_SIZE

    print(f"   RSC2 v{version} | {sample_rate} Hz | {channels} ch")
    print(f"   FFT={fft_size} hop={hop_size} max_partials={max_partials} frames={n_frames}")
    print(f"   window_sum={window_sum:.4f}")

    n_bins      = fft_size // 2 + 1
    # amplitude correction: irfft output is scaled by 1/fft_size relative to
    # the input spectrum, and the DPSS window reduced amplitudes by window_sum/fft_size,
    # so the net factor to recover original amplitude is:
    #   amp_correction = fft_size / window_sum
    amp_corr    = fft_size / window_sum
    total       = n_samples + fft_size
    pcm         = np.zeros((total, channels), dtype=np.float32)

    for ch in range(channels):
        ch_pcm  = np.zeros(total, dtype=np.float32)
        ch_cnt  = np.zeros(total, dtype=np.float32)
        pos     = 0
        for _ in range(n_frames):
            (n_p,) = struct.unpack_from(">H", data, off); off += 2
            spec = np.zeros(n_bins, dtype=np.complex64)
            for _ in range(n_p):
                b, amp, ph = struct.unpack_from(">Hff", data, off)
                off += struct.calcsize(">Hff")
                spec[b] = amp * np.exp(1j * ph)
            frame = np.fft.irfft(spec, n=fft_size).astype(np.float32) * amp_corr
            end   = pos + fft_size
            ch_pcm[pos:end] += frame
            ch_cnt[pos:end] += 1.0
            pos += hop_size
        ch_cnt = np.where(ch_cnt < 1e-8, 1.0, ch_cnt)
        pcm[:, ch] = ch_pcm / ch_cnt

    return pcm[:n_samples, :], sample_rate, channels

def save_wav(path, pcm, sr):
    n, ch = pcm.shape
    raw = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    with wave.open(path, "wb") as wf:
        wf.setnchannels(ch); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(raw)

def main():
    ap = argparse.ArgumentParser(description="RSC2 decoder")
    ap.add_argument("input"); ap.add_argument("output")
    a = ap.parse_args()
    print(f"📂 Loading {a.input}…")
    with open(a.input, "rb") as f:
        data = f.read()
    print(f"   {len(data):,} bytes")
    print("⚙️  Decoding…")
    pcm, sr, ch = decode(data)
    print(f"💾 Writing {a.output}…")
    save_wav(a.output, pcm, sr)
    print(f"✅ Done  ({len(pcm)} samples, {sr} Hz, {ch} ch)")

if __name__ == "__main__":
    main()