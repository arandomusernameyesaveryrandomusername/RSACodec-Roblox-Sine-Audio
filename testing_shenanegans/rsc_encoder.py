import numpy as np
import soundfile as sf

def hann_window(n):
    return np.hanning(n)

def greedy_top_n(mag, phase, n_keep):
    idx = np.argsort(mag)[-n_keep:]  # top N magnitudes
    new_mag = np.zeros_like(mag)
    new_phase = np.zeros_like(phase)

    new_mag[idx] = mag[idx]
    new_phase[idx] = phase[idx]

    return new_mag, new_phase

def process_frame(frame, n_keep):
    window = hann_window(len(frame))
    x = frame * window

    fft = np.fft.rfft(x)
    mag = np.abs(fft)
    phase = np.angle(fft)

    mag, phase = greedy_top_n(mag, phase, n_keep)

    rebuilt = mag * np.exp(1j * phase)
    time = np.fft.irfft(rebuilt)

    return time

def overlap_add(frames, hop_size):
    out_len = hop_size * (len(frames) - 1) + len(frames[0])
    out = np.zeros(out_len)

    for i, frame in enumerate(frames):
        start = i * hop_size
        out[start:start + len(frame)] += frame

    return out

def main(input_file, output_file, frame_size=2048, hop_size=735, n_keep=64):
    audio, sr = sf.read(input_file)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono mixdown

    frames = []

    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i + frame_size]
        rebuilt = process_frame(frame, n_keep)
        frames.append(rebuilt)

    out = overlap_add(frames, hop_size)

    # normalize
    out = out / (np.max(np.abs(out)) + 1e-9)

    sf.write(output_file, out, sr)

if __name__ == "__main__":
    main("t1.flac", "output_topn.wav", n_keep=192)