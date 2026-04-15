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
import time
import sys

RSC2_MAGIC  = b"RSC2"
HEADER_FMT  = "<BBIHHHIIf"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# Per-frame header written by encoder: peak(f32be) + nPartials(u16be)
FRAME_HDR_FMT  = "<fH"
FRAME_HDR_SIZE = struct.calcsize(FRAME_HDR_FMT)   # 6

# Per-partial record: bin(u16be) amp(u16be) phase(u16be)
PARTIAL_FMT  = "<HHH"
PARTIAL_SIZE = struct.calcsize(PARTIAL_FMT)        # 6

_U16_MAX    = 65535.0
_TWO_PI     = 2.0 * np.pi


# ── Progress Bar Helper ───────────────────────────────────────────────────────
class ProgressBar:
    def __init__(self, total: int, desc: str = "Progress", width: int = 50, unit: str = "items"):
        self.total = total
        self.desc = desc
        self.width = width
        self.unit = unit
        self.current = 0
        self.start_time = time.perf_counter()
        self.last_update = self.start_time
        self.last_render_time = self.start_time
        self.target_screen_fps = 60.0
        self.screen_interval = 1.0 / self.target_screen_fps

    def update(self, amount: int = 1, force: bool = False) -> None:
        self.current = min(self.current + amount, self.total)
        now = time.perf_counter()
        
        # Update screen at ~60 FPS for smooth animation
        if (now - self.last_render_time) >= self.screen_interval or force or self.current >= self.total:
            self._render(now)
            self.last_render_time = now

    def _render(self, now: float) -> None:
        elapsed = now - self.start_time
        
        # Calculate processing speed (items/second)
        if elapsed > 0.1:  # Only show speed after 0.1s to avoid noise
            speed = self.current / elapsed
        else:
            speed = 0

        # Calculate progress
        pct = self.current / self.total if self.total > 0 else 0.0
        filled = int(self.width * pct)
        bar = "█" * filled + "░" * (self.width - filled)

        # Calculate ETA
        if pct > 0 and elapsed > 0:
            total_time = elapsed / pct
            eta_sec = total_time - elapsed
            eta_str = self._format_time(eta_sec)
        else:
            eta_str = "--:--"

        elapsed_str = self._format_time(elapsed)

        # Build output
        line = (f"\r{self.desc:20} │{bar}│ {self.current:6}/{self.total:6} "
                f"[{elapsed_str} < {eta_str}] {speed:7.1f} {self.unit}/s")
        sys.stdout.write(line)
        sys.stdout.flush()

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as HH:MM:SS or MM:SS"""
        if seconds < 0:
            seconds = 0
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def finish(self) -> None:
        self.current = self.total
        self._render(time.perf_counter())
        print()  # newline


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

    pbar = ProgressBar(channels * n_frames, "Decoding frames", width=40, unit="frames")

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

            pbar.update(1)

        # Divide by overlap count (always 1 for hop=fft_size, but kept for safety)
        ch_cnt = np.where(ch_cnt < 1e-8, 1.0, ch_cnt)
        pcm[:, ch] = ch_pcm / ch_cnt

    pbar.finish()
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
    t0 = time.perf_counter()
    pcm, sr, ch = decode(data)
    dt = time.perf_counter() - t0

    print(f"💾 Writing {a.output}…")
    save_wav(a.output, pcm, sr)
    print(f"✅ Done  ({len(pcm)} samples, {sr} Hz, {ch} ch)")
    print(f"   Decoded in {dt * 1000:.1f} ms")


if __name__ == "__main__":
    main()