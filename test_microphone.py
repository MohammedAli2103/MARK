import sounddevice as sd
import numpy as np
import wave
from faster_whisper import WhisperModel

DURATION = 8
SAMPLE_RATE = 16000
MICROPHONE_DEVICE = 2
OUTPUT_FILE = "microphone_test.wav"

print("Loading Whisper base model...")
model = WhisperModel("base", device="cpu", compute_type="int8")

device_info = sd.query_devices(MICROPHONE_DEVICE)
print(f"Microphone: {device_info['name']}")
print(f"Recording for {DURATION} seconds.")
print("Say clearly: Hi, I am Mohammed Ali.")
input("Press Enter when you're ready...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    device=MICROPHONE_DEVICE,
)
sd.wait()

audio = np.squeeze(audio)
peak = float(np.max(np.abs(audio)))
rms = float(np.sqrt(np.mean(audio ** 2)))

print(f"Peak level: {peak:.4f}")
print(f"RMS level: {rms:.4f}")

# Save audio for replay and diagnosis.
pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

with wave.open(OUTPUT_FILE, "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(SAMPLE_RATE)
    wav.writeframes(pcm.tobytes())

print(f"Recording saved to: {OUTPUT_FILE}")
print("Transcribing...")

segments, info = model.transcribe(
    audio,
    language="en",
    beam_size=5,
    vad_filter=False,
    condition_on_previous_text=False,
)

transcript = " ".join(segment.text.strip() for segment in segments)

print("\nDetected language:", info.language)
print("Recognized speech:", transcript or "[No speech detected]")