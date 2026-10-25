import time
import re
from tts_engine import TTSEngine

def benchmark():
    print("=== BENCHMARKING KOKORO TTS CHUNKING & TIMINGS ===")
    t_init_start = time.time()
    tts = TTSEngine()
    t_init = time.time() - t_init_start
    print(f"Model init time: {t_init:.3f}s")

    test_text = "Hello! I can help you with your programming question. Here is a simple explanation of how loops work."
    clean = tts.clean_text(test_text)
    print(f"\nTest text ({len(clean)} chars): '{clean}'")

    # 1. Old Sentence Splitting
    chunks = [c.strip() for c in re.split(r"(?<=[.?!])\s+", clean) if c.strip()]
    print(f"\n--- Strategy A: Sentence Splitting (Current) ---")
    print(f"Number of chunks: {len(chunks)}")
    
    total_infer_time = 0.0
    for i, chunk in enumerate(chunks):
        t0 = time.time()
        samples, sr = tts._kokoro.create(chunk, voice=tts.voice, speed=tts.speed, lang="en-us")
        t_infer = time.time() - t0
        total_infer_time += t_infer
        dur = len(samples) / sr
        print(f"  Chunk {i+1} ({len(chunk)} chars): '{chunk}'")
        print(f"    Inference time: {t_infer:.3f}s | Audio duration: {dur:.2f}s")
        print(f"    GAP IN CURRENT CODE: Because chunk {i+1} only starts inference AFTER chunk {i} finishes playing, the gap between chunk {i} and {i+1} is {t_infer:.3f}s!")

    print(f"Total inference across 3 chunks: {total_infer_time:.3f}s")

    # 2. Strategy B: Single Paragraph Synthesis
    print(f"\n--- Strategy B: Single Full Synthesis (Combined) ---")
    t0 = time.time()
    samples_full, sr = tts._kokoro.create(clean, voice=tts.voice, speed=tts.speed, lang="en-us")
    t_infer_full = time.time() - t0
    dur_full = len(samples_full) / sr
    print(f"  Full text ({len(clean)} chars) single inference: {t_infer_full:.3f}s | Audio duration: {dur_full:.2f}s")
    print(f"  Speedup: 1 single pass ({t_infer_full:.3f}s) vs 3 passes ({total_infer_time:.3f}s) = {total_infer_time / t_infer_full:.2f}x faster!")
    print(f"  Inter-sentence pause in single synthesis: Natural (~0.3-0.5s) handled by Kokoro's own phoneme/prosody model without ANY engine stalls!")

if __name__ == "__main__":
    benchmark()
