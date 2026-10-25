import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

DEVICE = 2
RATE = 16000
DURATION = 8

print("Record your voice first...")
input("Press Enter to begin recording: ")

audio = sd.rec(
    int(DURATION * RATE),
    samplerate=RATE,
    channels=1,
    dtype="float32",
    device=DEVICE,
)
sd.wait()
audio = np.squeeze(audio)

print("\nLoading Whisper small model...")
model = WhisperModel("small", device="cpu", compute_type="int8")

segments, info = model.transcribe(
    audio,
    language="en",
    beam_size=5,
    vad_filter=False,
    condition_on_previous_text=False,
)

print("\nRecognized speech:")
print(" ".join(segment.text.strip() for segment in segments))
