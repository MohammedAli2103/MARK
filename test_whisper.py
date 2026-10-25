from faster_whisper import WhisperModel

print("Loading Whisper base model...")
model = WhisperModel("base", device="cpu", compute_type="int8")
print("Whisper base model loaded successfully.")
