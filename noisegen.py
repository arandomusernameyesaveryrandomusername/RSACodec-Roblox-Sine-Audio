import numpy as np, wave
s = (np.random.randn(44100) * 32767).astype(np.int16)
w = wave.open('noise.wav','wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100); w.writeframes(s.tobytes()); w.close()
