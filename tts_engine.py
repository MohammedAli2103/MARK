"""
MARK TTS Engine — High-Fidelity Local Neural Speech (Kokoro-82M) with Pipelined Synthesis.
Eliminates inter-sentence pauses via natural chunking and background pre-synthesis.
"""

import os
import queue
import re
import threading
import time
from pathlib import Path
import numpy as np
import sounddevice as sd

class TTSEngine:
    def __init__(self, model_dir=None, voice="bm_george", speed=1.0):
        self.voice = voice
        self.speed = speed
        self._kokoro = None
        self._sapi_engine = None
        self._is_speaking = False
        self._lock = threading.Lock()
        self._job_counter = 0
        self._active_job_id = None
        self._active_cancel_event = None

        if model_dir is None:
            model_dir = Path("C:/Projects/Mark/models/kokoro")
        else:
            model_dir = Path(model_dir)

        self.model_path = model_dir / "kokoro-v1.0.onnx"
        self.voices_path = model_dir / "voices-v1.0.bin"

        self._init_kokoro()
        if not self._kokoro:
            self._init_sapi()

    def _get_lang_for_voice(self, voice: str) -> str:
        """Determines proper phonemizer language code for the selected Kokoro voice."""
        v = (voice or self.voice).lower()
        if v.startswith(("bm_", "bf_")):
            return "en-gb"
        if v.startswith(("am_", "af_")):
            return "en-us"
        if v.startswith("e"):
            return "es"
        if v.startswith("f"):
            return "fr-fr"
        if v.startswith("h"):
            return "hi"
        if v.startswith("i"):
            return "it"
        if v.startswith("j"):
            return "ja"
        if v.startswith("z"):
            return "zh"
        return "en-gb"

    def _init_kokoro(self):
        if self.model_path.is_file() and self.voices_path.is_file():
            try:
                import onnxruntime as rt
                from kokoro_onnx import Kokoro
                print(f"[TTSEngine] Loading Kokoro-82M with 8 CPU intra-threads (P-core hyperthread mapped)...")
                opts = rt.SessionOptions()
                opts.intra_op_num_threads = 8
                opts.inter_op_num_threads = 1
                opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
                opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
                sess = rt.InferenceSession(str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"])
                self._kokoro = Kokoro.from_session(sess, str(self.voices_path))
                print(f"[TTSEngine] Kokoro-82M loaded successfully (8 threads, default voice: {self.voice}, lang: {self._get_lang_for_voice(self.voice)}).")
            except Exception as exc:
                print(f"[TTSEngine] Note: Custom session fallback ({exc}), using default Kokoro loader...")
                try:
                    from kokoro_onnx import Kokoro
                    self._kokoro = Kokoro(str(self.model_path), str(self.voices_path))
                except Exception as e2:
                    print(f"[TTSEngine] Failed to load Kokoro-82M: {e2}")
                    self._kokoro = None
        else:
            print("[TTSEngine] Kokoro model files not ready yet.")

    def _init_sapi(self):
        try:
            import pyttsx3
            self._sapi_engine = pyttsx3.init()
            self._sapi_engine.setProperty("rate", 175)
            voices = self._sapi_engine.getProperty("voices")
            for v in voices:
                if "david" in v.name.lower() or "george" in v.name.lower() or "mark" in v.name.lower():
                    self._sapi_engine.setProperty("voice", v.id)
                    break
            print("[TTSEngine] Initialized fallback Windows SAPI engine.")
        except Exception as exc:
            print(f"[TTSEngine] SAPI init failed: {exc}")

    def clean_text(self, text: str) -> str:
        """Strip markdown syntax, code fences, and unwanted formatting for clean, natural speech."""
        if not text:
            return ""
        # Replace multiline code blocks with natural spoken summary
        text = re.sub(r"```[\w-]*\n[\s\S]*?```", " The complete implementation is displayed on your screen. ", text)
        text = re.sub(r"```[\s\S]*?```", " The complete implementation is displayed on your screen. ", text)
        # Inline code: speak naturally without backticks
        text = re.sub(r"`([^`]+)`", r"\1", text)
        # Remove markdown bold/italics
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)
        # Remove markdown links [text](url) -> text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Remove headers #
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Remove list dashes/bullets at start of lines
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
        # Remove numbered lists like 1. 
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
        # Collapse extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _segment_text(self, text: str, min_chars: int = 120, max_chars: int = 340) -> list[str]:
        """
        Groups sentences into natural, conversational chunks (~120-340 characters).
        Kokoro naturally handles intra-chunk sentences with perfect prosody (~300ms pause),
        while avoiding huge multi-thousand character monoliths.
        """
        raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not raw_sentences:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        for sentence in raw_sentences:
            if not current:
                current = sentence
            elif len(current) + len(sentence) + 1 <= max_chars:
                current += " " + sentence
            else:
                chunks.append(current)
                current = sentence
        if current:
            chunks.append(current)
        return chunks

    def stop(self):
        """Interrupt and cease current speech playback immediately."""
        self._active_job_id = None
        if self._active_cancel_event:
            self._active_cancel_event.set()
        try:
            sd.stop()
        except Exception:
            pass
        if self._sapi_engine:
            try:
                self._sapi_engine.stop()
            except Exception:
                pass
        self._is_speaking = False

    def is_speaking(self) -> bool:
        return self._is_speaking

    def synthesize_chunk(self, text: str, voice=None):
        """
        Synthesizes a single chunk or sentence without playing it.
        Returns (samples, sample_rate, duration_seconds, clean_text).
        """
        clean = self.clean_text(text)
        if not clean:
            return None, 16000, 0.0, ""

        if not self._kokoro and self.model_path.is_file() and self.voices_path.is_file():
            self._init_kokoro()

        if self._kokoro:
            chosen_voice = voice or self.voice
            samples, sr = self._kokoro.create(
                clean,
                voice=chosen_voice,
                speed=self.speed,
                lang=self._get_lang_for_voice(chosen_voice)
            )
            dur = len(samples) / sr
            return samples, sr, dur, clean
        return None, 16000, 0.0, clean

    def speak_stream(
        self,
        sentence_source,
        voice=None,
        job_id=None,
        cancel_token=None,
        on_playback_start=None,
        on_sentence_play=None,
        on_finish_callback=None
    ) -> bool:
        """
        Streams and speaks incoming sentences concurrently.
        sentence_source can be a queue.Queue or an iterable of strings.
        Playback starts as soon as the first chunk is synthesized while subsequent chunks
        synthesize in the background.
        """
        with self._lock:
            self._job_counter += 1
            current_job_id = str(job_id) if job_id is not None else f"tts_{self._job_counter}"
            self._active_job_id = current_job_id
            job_cancel = threading.Event()
            self._active_cancel_event = job_cancel

            def is_cancelled():
                if job_cancel.is_set():
                    return True
                if cancel_token and cancel_token.is_set():
                    return True
                if self._active_job_id != current_job_id:
                    return True
                return False

            if is_cancelled():
                return False

            self._is_speaking = True
            try:
                sd.stop()
            except Exception:
                pass

            t_total_start = time.time()
            completed = False

            try:
                if not self._kokoro and self.model_path.is_file() and self.voices_path.is_file():
                    self._init_kokoro()

                if self._kokoro:
                    audio_queue = queue.Queue(maxsize=3)
                    chosen_voice = voice or self.voice

                    def producer():
                        idx = 0
                        is_queue = hasattr(sentence_source, "get")

                        while not is_cancelled():
                            if is_queue:
                                try:
                                    sent = sentence_source.get(timeout=0.1)
                                except queue.Empty:
                                    continue
                                if sent is None:
                                    break
                            else:
                                try:
                                    sent = next(sentence_source)
                                except StopIteration:
                                    break

                            clean = self.clean_text(sent)
                            if not clean:
                                continue

                            t_inf_start = time.time()
                            try:
                                samples, sr, dur, clean_txt = self.synthesize_chunk(clean, voice=chosen_voice)
                                t_inf = time.time() - t_inf_start
                                if is_cancelled():
                                    break
                                while not is_cancelled():
                                    try:
                                        audio_queue.put((idx, samples, sr, dur, clean_txt, t_inf), timeout=0.1)
                                        idx += 1
                                        break
                                    except queue.Full:
                                        continue
                            except Exception as exc:
                                print(f"[TTSEngine] [{current_job_id}] Chunk {idx+1} synth error: {exc}")
                                break

                        audio_queue.put(None)

                    prod_thread = threading.Thread(target=producer, daemon=True)
                    prod_thread.start()

                    prev_chunk_finish = None
                    chunks_played = 0
                    first_played = False

                    while not is_cancelled():
                        try:
                            item = audio_queue.get(timeout=0.1)
                        except queue.Empty:
                            continue

                        if item is None or is_cancelled():
                            break

                        idx, samples, sr, audio_dur, clean_txt, t_inf = item
                        if samples is None:
                            continue

                        gap_ms = (time.time() - prev_chunk_finish) * 1000 if prev_chunk_finish else 0.0
                        print(f"[TTSEngine] [{current_job_id}] Playing chunk {idx+1}: gap={gap_ms:.1f}ms, infer={t_inf:.2f}s, dur={audio_dur:.2f}s")

                        if is_cancelled():
                            sd.stop()
                            break

                        if not first_played:
                            first_played = True
                            if on_playback_start:
                                try:
                                    on_playback_start()
                                except Exception:
                                    pass

                        if on_sentence_play:
                            try:
                                on_sentence_play(clean_txt)
                            except Exception:
                                pass

                        # Play current chunk with safety timeout
                        sd.play(samples, samplerate=sr)
                        t_chunk_start = time.time()
                        max_chunk_wait = audio_dur + 1.5

                        while sd.get_stream() and sd.get_stream().active and (time.time() - t_chunk_start < max_chunk_wait):
                            if is_cancelled():
                                sd.stop()
                                break
                            time.sleep(0.02)

                        if is_cancelled():
                            break

                        prev_chunk_finish = time.time()
                        chunks_played += 1

                        # Natural slight cadence pause between sentences (~25ms)
                        if not is_cancelled() and not audio_queue.empty():
                            time.sleep(0.025)

                    if is_cancelled():
                        try:
                            sd.stop()
                        except Exception:
                            pass

                    t_total = time.time() - t_total_start
                    completed = not is_cancelled()
                    if completed:
                        print(f"[TTSEngine] [{current_job_id}] Speech completed in {t_total:.2f}s ({chunks_played} chunks).")
                    else:
                        print(f"[TTSEngine] [{current_job_id}] Speech cancelled / interrupted.")

                elif self._sapi_engine:
                    if is_cancelled():
                        return False
                    # SAPI fallback: consume all sentences
                    full_text = []
                    if hasattr(sentence_source, "get"):
                        while True:
                            s = sentence_source.get()
                            if s is None:
                                break
                            full_text.append(s)
                    else:
                        full_text = list(sentence_source)

                    clean = self.clean_text(" ".join(full_text))
                    if on_playback_start:
                        on_playback_start()
                    self._sapi_engine.say(clean)
                    self._sapi_engine.runAndWait()
                    completed = not is_cancelled()
                else:
                    print("[TTSEngine] No TTS engine available!")
                    completed = False

            except Exception as exc:
                print(f"[TTSEngine] Speech playback error: {exc}")
                completed = False
            finally:
                self._is_speaking = False
                if self._active_job_id == current_job_id:
                    self._active_job_id = None
                if on_finish_callback and completed:
                    try:
                        on_finish_callback()
                    except Exception:
                        pass

            return completed

    def speak(self, text: str, voice=None, job_id=None, cancel_token=None, on_finish_callback=None) -> bool:
        """
        Synthesizes and speaks text out loud using pipelined background pre-synthesis.
        Guarantees strict cancellation check at every stage and immediate audio interruption.
        Returns True if completed, False if interrupted or failed.
        """
        clean = self.clean_text(text)
        if not clean:
            if on_finish_callback:
                on_finish_callback()
            return True

        chunks = self._segment_text(clean)
        return self.speak_stream(
            iter(chunks),
            voice=voice,
            job_id=job_id,
            cancel_token=cancel_token,
            on_finish_callback=on_finish_callback
        )

