import numpy as np
from scipy.io import wavfile

# ─────────────────────────────────────────────
# CONFIG (tweak if you want king)
# ─────────────────────────────────────────────
FREQUENCIES   = [50, 100, 440, 1000, 5000, 10000]  # Hz
DURATION      = 1.0     # seconds
SAMPLE_RATE   = 44100   # standard CD quality
AMPLITUDE     = 0.999   # almost full scale, avoids clipping on some players
OUTPUT_PREFIX = "sine_" # filename prefix

# ─────────────────────────────────────────────
# Generate & save loop-ready sines
# ─────────────────────────────────────────────
for freq in FREQUENCIES:
    # Time vector — exact integer number of samples
    num_samples = int(SAMPLE_RATE * DURATION)
    t = np.linspace(0, DURATION, num_samples, endpoint=False)
    
    # Pure sine wave
    wave = AMPLITUDE * np.sin(2 * np.pi * freq * t)
    
    # Convert to signed 16-bit integer (WAV standard)
    wave_int16 = np.int16(wave * 32767)
    
    # Filename like sine_440hz_1s.wav
    filename = f"{OUTPUT_PREFIX}{freq}hz_{DURATION}s.wav"
    
    wavfile.write(filename, SAMPLE_RATE, wave_int16)
    print(f"🔊 Generated → {filename}  ({freq} Hz, {num_samples} samples)")

print("\n🎉 All done king! Upload these bad boys to Roblox → get looped asset IDs → plug into multi-band RSC pools 🔥🚀")
print("Pro tip: In Roblox, set Looped = true + RollOffMaxDistance = 0 for 2D everywhere playback 😏")