#!/usr/bin/env python3
"""
RSC2 Audio Codec - Decoder (u16 amp+phase edition)

Binary format per frame:
  peak       f32be   per-frame max FFT magnitude
  nPartials  u16be
  [ bin(u16be)  amp(u16be)  phase(u16be) ] × nPartials

Reconstruction per frame:
  amp   = (amp_u16  / 65535.0) * peak          → absolute FFT magnitude
  phase = (ph_u16   / 65535.0) * 2π  - π       → radians
  spec[bin] = amp * exp(j * phase)
  frame = irfft(spec, n=fft_size)               → no extra amp_corr needed;
                                                   irfft(rfft(x)) = x for the
                                                   rectangular window used by
                                                   the encoder (win_sum = fft_size)
  OLA with rectangular accumulation + divide by overlap count
"""
import numpy as np
import struct
import wave
import argparse

RSC2_MAGIC  = b"RSC2"
HEADER_FMT  = ">BBIHHHIIf"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# Per-frame header written by encoder: peak(f32be) + nPartials(u16be)
FRAME_HDR_FMT  = ">fH"
FRAME_HDR_SIZE = struct.calcsize(FRAME_HDR_FMT)   # 6

# Per-partial record: bin(u16be) amp(u16be) phase(u16be)
PARTIAL_FMT  = ">HHH"
PARTIAL_SIZE = struct.calcsize(PARTIAL_FMT)        # 6

_U16_MAX    = 65535.0
_TWO_PI     = 2.0 * np.pi


def decode(data: bytes):
    if data[:4] != RSC2_MAGIC:
        raise ValueError(f"Bad magic {data[:4]!r}")

    off = 4
    (version, channels, sample_rate,
     fft_size, hop_size, max_partials,
     n_frames, n_samples, window_sum) = struct.unpack_from(HEADER_FMT, data, off)
    off += HEADER_SIZE

    print(f"   RSC2 v{version} | {sample_rate} Hz | {channels} ch")
    print(f"   FFT={fft_size} hop={hop_size} max_partials={max_partials} frames={n_frames}")
    print(f"   window_sum={window_sum:.4f}  (rect window → no amp correction needed)")

    if version != 1:
        raise ValueError(f"Unsupported RSC2 version {version}")

    n_bins = fft_size // 2 + 1
    total  = n_samples + fft_size          # safe output buffer length

    pcm = np.zeros((total, channels), dtype=np.float32)

    for ch in range(channels):
        ch_pcm = np.zeros(total, dtype=np.float32)
        ch_cnt = np.zeros(total, dtype=np.float32)
        pos    = 0

        for _ in range(n_frames):
            # ── Frame header ─────────────────────────────────────────────────
            peak, n_p = struct.unpack_from(FRAME_HDR_FMT, data, off)
            off += FRAME_HDR_SIZE

            # ── Reconstruct spectrum ──────────────────────────────────────────
            spec = np.zeros(n_bins, dtype=np.complex64)
            for _ in range(n_p):
                b, amp_u16, ph_u16 = struct.unpack_from(PARTIAL_FMT, data, off)
                off += PARTIAL_SIZE

                # Recover absolute magnitude and phase
                amp   = (amp_u16 / _U16_MAX) * peak
                phase = (ph_u16  / _U16_MAX) * _TWO_PI - np.pi
                spec[b] = amp * np.exp(1j * phase)

            # ── IRFFT → OLA ───────────────────────────────────────────────────
            # irfft(rfft(x)) = x for the rectangular window used by the encoder,
            # so no extra amplitude correction factor is required.
            frame = np.fft.irfft(spec, n=fft_size).astype(np.float32)
            end   = pos + fft_size
            ch_pcm[pos:end] += frame
            ch_cnt[pos:end] += 1.0
            pos += hop_size

        # Divide by overlap count (always 1 for hop=fft_size, but kept for safety)
        ch_cnt = np.where(ch_cnt < 1e-8, 1.0, ch_cnt)
        pcm[:, ch] = ch_pcm / ch_cnt

    return pcm[:n_samples, :], sample_rate, channels


def save_wav(path: str, pcm: np.ndarray, sr: int):
    n, ch = pcm.shape
    raw = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    with wave.open(path, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(raw)


def main():
    ap = argparse.ArgumentParser(description="RSC2 decoder (u16 amp+phase)")
    ap.add_argument("input")
    ap.add_argument("output")
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