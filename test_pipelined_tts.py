import time
import queue
import threading
import re
import numpy as np
import sounddevice as sd
from tts_engine import TTSEngine

class PipelinedTTS:
    def __init__(self, tts_engine: TTSEngine):
        self.tts = tts_engine

    def group_sentences(self, text, min_chars=120, max_chars=350):
        """Groups sentences into coherent conversational chunks (~120-350 chars)."""
        raw_sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]
        chunks = []
        current = ""
        for s in raw_sentences:
            if not current:
                current = s
            elif len(current) + len(s) + 1 <= max_chars:
                current += " " + s
            else:
                chunks.append(current)
                current = s
        if current:
            chunks.append(current)
        return chunks

    def test_pipeline(self, text):
        clean = self.tts.clean_text(text)
        chunks = self.group_sentences(clean)
        print(f"\nPipelined test on {len(chunks)} grouped chunks:")
        for i, c in enumerate(chunks):
            print(f"  Chunk {i+1} ({len(c)} chars): '{c}'")

        audio_queue = queue.Queue(maxsize=4)
        stop_event = threading.Event()
        timings = []

        # Producer thread: synthesizes chunk N+1 while chunk N is playing
        def producer():
            for i, chunk in enumerate(chunks):
                if stop_event.is_set():
                    break
                t0 = time.time()
                samples, sr = self.tts._kokoro.create(chunk, voice=self.tts.voice, speed=self.tts.speed, lang="en-us")
                t_inf = time.time() - t0
                print(f"[Producer] Chunk {i+1} synthesized in {t_inf:.3f}s (audio duration: {len(samples)/sr:.2f}s)")
                audio_queue.put((i, samples, sr, t_inf))
            audio_queue.put(None)  # Sentinel

        prod_thread = threading.Thread(target=producer, daemon=True)
        t_start = time.time()
        prod_thread.start()

        # Consumer / Playback (simulated timing check)
        prev_chunk_end = None
        while True:
            item = audio_queue.get()
            if item is None:
                break
            idx, samples, sr, t_inf = item
            playback_start = time.time()
            gap = (playback_start - prev_chunk_end) if prev_chunk_end else 0.0
            dur = len(samples) / sr
            print(f"[Consumer] Playing Chunk {idx+1}: gap={gap*1000:.1f}ms (target: <500ms)")
            timings.append((idx, gap, t_inf, dur))
            # Simulate playing audio (sleep duration)
            time.sleep(dur)
            prev_chunk_end = time.time()

        prod_thread.join()
        total_time = time.time() - t_start
        print(f"\nTotal pipelined playback time: {total_time:.2f}s")
        for idx, gap, t_inf, dur in timings:
            print(f"  Chunk {idx+1}: gap={gap*1000:.1f}ms, infer={t_inf:.2f}s, dur={dur:.2f}s")

if __name__ == "__main__":
    e = TTSEngine()
    pipe = PipelinedTTS(e)
    test_text = (
        "Here is a simple C program that prints the even numbers from 1 to 20. "
        "Note that we initialize the variable to 2 and increment by 2 in each iteration. "
        "The program includes the standard input output header file. "
        "Inside the main function, a for loop iterates through the range. "
        "Finally, the return zero statement indicates successful execution."
    )
    pipe.test_pipeline(test_text)
