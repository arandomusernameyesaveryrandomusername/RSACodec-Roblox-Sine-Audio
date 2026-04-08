import argparse
import numpy as np
import librosa


def load_audio(path, sr=None):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr



def log_spectral_distance(x, y, n_fft=1024, hop_length=256):
    # Match lengths
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    # STFT magnitude
    Sx = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length)) + 1e-8
    Sy = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) + 1e-8

    # Log difference
    log_diff = np.log(Sx) - np.log(Sy)

    # LSD per frame → then mean
    lsd = np.mean(np.sqrt(np.mean(log_diff**2, axis=0)))

    return lsd


def main():
    parser = argparse.ArgumentParser(description="Log Spectral Distance (LSD) calculator")
    parser.add_argument("file1", help="Original/reference audio")
    parser.add_argument("file2", help="Reconstructed/processed audio")
    parser.add_argument("--sr", type=int, default=None, help="Force sample rate (optional)")
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=735)

    args = parser.parse_args()

    print("🔊 Loading audio...")
    x, sr1 = load_audio(args.file1, args.sr)
    y, sr2 = load_audio(args.file2, args.sr)

    # match loudness properly
    peak = max(np.max(np.abs(x)), np.max(np.abs(y))) + 1e-8
    x /= peak
    y /= peak

    if sr1 != sr2:
        print(f"⚠️ Sample rates differ ({sr1} vs {sr2}), resampling...")
        y = librosa.resample(y, orig_sr=sr2, target_sr=sr1)
        sr = sr1
    else:
        sr = sr1

    print("🧮 Computing LSD...")
    lsd_value = log_spectral_distance(x, y, args.n_fft, args.hop)

    print(f"\n📊 Log Spectral Distance: {lsd_value:.6f}")

    # Interpretation guide
    if lsd_value < 0.1:
        quality = "🔥 Extremely close (almost identical)"
    elif lsd_value < 0.3:
        quality = "✅ Very good"
    elif lsd_value < 0.6:
        quality = "⚠️ Noticeable differences"
    else:
        quality = "💀 Large degradation"

    print(f"🎯 Quality: {quality}")


if __name__ == "__main__":
    main()