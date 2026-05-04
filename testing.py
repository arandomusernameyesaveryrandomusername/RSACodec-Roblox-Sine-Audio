import os
import random
import librosa
import numpy as np
import sounddevice as sd
import threading
from collections import deque

FOLDER_PATH = './'
FFT_SIZE = 2048
HOP_SIZE = 512
SR = 44100
TRANSITION_SEC = 1.5

# --- DECOUPLED CURVES ---
# freq morphs FAST (exponential), amp morphs SLOW (sigmoid)
FREQ_SPEED = 3.0   # Frequency morphs 3x faster than amplitude
AMP_CURVE = 'sigmoid'  # slow start, slow end, fast middle
FREQ_CURVE = 'exp'     # fast start, then settles

WINDOW = np.hanning(FFT_SIZE).astype(np.float32)

def amp_weight(t):
    """Slow, smooth sigmoid crossfade."""
    # S-curve: gentle start, gentle end
    return 1.0 / (1.0 + np.exp(-12 * (t - 0.5)))

def freq_weight(t):
    """Fast exponential — frequencies lock in early."""
    # Exponential approach: 80% of freq shift happens in first 30% of time
    return 1.0 - np.exp(-5.0 * t)

class SpectralDJ:
    def __init__(self, folder):
        self.files = [f for f in os.listdir(folder) if f.endswith(('.wav', '.mp3'))]
        if not self.files:
            raise ValueError("No audio files!")
        
        print("🔄 Loading...")
        self.audio_cache = {}
        for f in self.files:
            path = os.path.join(folder, f)
            y, _ = librosa.load(path, sr=SR, mono=True)
            self.audio_cache[f] = np.pad(y.astype(np.float32), (FFT_SIZE, FFT_SIZE))
        
        self.file_list = list(self.audio_cache.keys())
        self.current_file = self.file_list[0]
        self.next_file = None
        self.pointer = FFT_SIZE
        self.current_audio = self.audio_cache[self.current_file]
        
        self.is_morphing = False
        self.morph_t = 0.0  # 0→1 linear time
        self.morph_step = 1.0 / (TRANSITION_SEC * SR / HOP_SIZE)
        
        # OLA + phase vocoder state
        self.ola_buffer = np.zeros(FFT_SIZE, dtype=np.float32)
        self.prev_phase = np.zeros(FFT_SIZE // 2 + 1, dtype=np.float32)
        
        self.output_queue = deque(maxlen=8)
        self.running = True
        self._start_processor()
    
    def _start_processor(self):
        def processor():
            while self.running:
                if len(self.output_queue) < 4:
                    self.output_queue.append(self._generate_frame())
                else:
                    import time
                    time.sleep(0.001)
        self.thread = threading.Thread(target=processor, daemon=True)
        self.thread.start()
    
    def _stft(self, audio, ptr):
        frame = audio[ptr:ptr + FFT_SIZE]
        if len(frame) < FFT_SIZE:
            frame = np.pad(frame, (0, FFT_SIZE - len(frame)))
        windowed = frame * WINDOW
        spec = np.fft.rfft(windowed)
        return np.abs(spec), np.angle(spec)
    
    def _generate_frame(self):
        ptr = self.pointer
        
        mag_a, phase_a = self._stft(self.current_audio, ptr)
        
        if self.is_morphing and self.next_file:
            next_audio = self.audio_cache[self.next_file]
            mag_b, phase_b = self._stft(next_audio, ptr)
            
            # --- DECOUPLED WEIGHTS ---
            w_freq = freq_weight(self.morph_t)   # FAST: freqs shift early
            w_amp = amp_weight(self.morph_t)     # SLOW: volume creeps
            
            print(f"\r⏳ t={self.morph_t:.2f} | freq_w={w_freq:.2f} | amp_w={w_amp:.2f}", end='')
            
            # MAGNITUDE: use AMP weight (slow crossfade)
            mag = (1 - w_amp) * mag_a + w_amp * mag_b
            
            # PHASE: use FREQ weight (fast spectral lock)
            # Interpolate on complex plane with fast weight
            z_a = mag_a * np.exp(1j * phase_a)
            z_b = mag_b * np.exp(1j * phase_b)
            z_blend = (1 - w_freq) * z_a + w_freq * z_b
            phase = np.angle(z_blend)
            
            # Phase vocoder continuity
            freqs = np.fft.rfftfreq(FFT_SIZE, 1/SR)
            expected = self.prev_phase + 2 * np.pi * freqs * HOP_SIZE / SR
            phase_diff = np.angle(np.exp(1j * (phase - expected)))
            phase = expected + phase_diff
            self.prev_phase = phase.copy()
            
            # Advance time
            self.morph_t += self.morph_step
            if self.morph_t >= 1.0:
                self.is_morphing = False
                self.current_file = self.next_file
                self.current_audio = self.audio_cache[self.current_file]
                self.morph_t = 0.0
                self.next_file = None
                print("\n✅ Morph Complete!")
        else:
            mag = mag_a
            phase = phase_a
            freqs = np.fft.rfftfreq(FFT_SIZE, 1/SR)
            expected = self.prev_phase + 2 * np.pi * freqs * HOP_SIZE / SR
            phase_diff = np.angle(np.exp(1j * (phase - expected)))
            phase = expected + phase_diff
            self.prev_phase = phase.copy()
        
        # Synthesize + OLA
        spec = mag * np.exp(1j * phase)
        frame_out = np.fft.irfft(spec, n=FFT_SIZE) * WINDOW
        self.ola_buffer += frame_out
        output = self.ola_buffer[:HOP_SIZE].copy()
        self.ola_buffer = np.roll(self.ola_buffer, -HOP_SIZE)
        self.ola_buffer[-HOP_SIZE:] = 0
        
        self.pointer += HOP_SIZE
        return output.astype(np.float32)
    
    def trigger_next(self):
        if not self.is_morphing:
            candidates = [f for f in self.file_list if f != self.current_file]
            self.next_file = random.choice(candidates) if candidates else self.file_list[0]
            print(f"\n🔥 Transitioning to: {self.next_file}")
            self.morph_t = 0.0
            self.is_morphing = True

    def audio_callback(self, outdata, frames, time_info, status):
        if status:
            print(f"\n⚠️ {status}")
        collected = []
        total = 0
        while total < frames and self.output_queue:
            chunk = self.output_queue.popleft()
            collected.append(chunk)
            total += len(chunk)
        if total < frames:
            outdata[:frames] = 0
            return
        buffer = np.concatenate(collected)
        outdata[:frames, 0] = buffer[:frames]
        if len(buffer) > frames:
            self.output_queue.appendleft(buffer[frames:])

# --- RUN ---
print("=" * 50)
print("🎧 SPECTRAL DJ — Fast Freq / Slow Amp")
print("=" * 50)

dj = SpectralDJ(FOLDER_PATH)
print("\nSPACE = Morph | Q = Quit")

try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False
    print("⚠️ pip install keyboard for hotkeys")

with sd.OutputStream(channels=1, callback=dj.audio_callback,
                     samplerate=SR, blocksize=HOP_SIZE, latency='low'):
    if HAS_KEYBOARD:
        keyboard.on_press_key('space', lambda _: dj.trigger_next())
        keyboard.wait('q')
    else:
        while True:
            cmd = input("> ").strip().lower()
            if cmd == 'm': dj.trigger_next()
            elif cmd == 'q': break

dj.running = False
dj.thread.join(timeout=1.0)
print("\n👋 Done!")