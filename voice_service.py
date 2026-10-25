"""
MARK Voice Service — Continuous Hands-Free Conversation Session Manager.
Features: Strict request identity (session_id & turn_id), multi-layered cancellation,
acoustic isolation & anti-self-hearing refractory period, and clean lifecycle management.
"""

import os
import queue
import re
import sys
import threading
import time
import uuid
from pathlib import Path

# Configure local CUDA runtime for CTranslate2 / Whisper GPU acceleration
def _setup_cuda_env():
    cuda_candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "lib" / "ollama" / "cuda_v12",
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin"),
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin"),
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin"),
    ]
    for p in cuda_candidates:
        if p.is_dir():
            p_str = str(p)
            if p_str not in os.environ.get("PATH", ""):
                os.environ["PATH"] = p_str + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(p_str)
                except Exception:
                    pass

_setup_cuda_env()

import numpy as np
import sounddevice as sd

from tts_engine import TTSEngine


# Explicit Voice State Machine
class VoiceState:
    DISABLED = "DISABLED"
    STANDBY = "STANDBY"
    WAKE_WORD_LISTENING = "WAKE WORD LISTENING"  # Alias
    LISTENING = "LISTENING"
    LISTENING_FOR_COMMAND = "LISTENING FOR COMMAND"  # Alias
    PROCESSING = "PROCESSING"
    TRANSCRIBING = "TRANSCRIBING"  # Substate
    THINKING = "THINKING"  # Substate
    GREETING = "GREETING"
    SPEAKING = "SPEAKING"
    ERROR = "MICROPHONE ERROR"


EXIT_PATTERNS = [
    r"\bgoodbye mark\b",
    r"\bgoodbye\b",
    r"\bbye mark\b",
    r"\bend conversation\b",
    r"\bstop conversation\b",
    r"\bclose session\b",
    r"\bend session\b",
    r"\bthats all mark\b",
    r"\bthat is all mark\b",
    r"\bthats all\b",
    r"\bthat is all\b",
    r"\bstop listening\b",
    r"\bexit\b",
    r"\bquit\b",
]

# Robust wake phrase matcher requiring active greeting prefix (prevents accidental 'mark' triggers)
WAKE_WORD_PATTERN = re.compile(
    r"\b(?:hey|hi|hello|ok|okay)\s+(?:amrk|mark|marc|dark|mike|bark|park|jarvis)\b",
    re.IGNORECASE
)


class TurnMetrics:
    """Instruments a turn with all 13 latency checkpoints specified in Part 11."""
    def __init__(self, session_id: str, turn_id: int):
        self.session_id = session_id
        self.turn_id = turn_id
        self.t_speech_start = 0.0
        self.t_speech_endpoint = 0.0
        self.t_transcribe_start = 0.0
        self.t_transcribe_end = 0.0
        self.t_llm_start = 0.0
        self.t_first_token = 0.0
        self.t_first_sentence = 0.0
        self.t_tts_start = 0.0
        self.t_first_chunk_ready = 0.0
        self.t_first_playback = 0.0
        self.t_llm_end = 0.0
        self.t_final_chunk_ready = 0.0
        self.t_playback_end = 0.0
        self.t_relisten_active = 0.0
        self.cancelled = False
        self.cancel_reason = None

    def summary(self) -> dict:
        def delta(t1, t0):
            return round((t1 - t0) * 1000, 1) if (t1 and t0 and t1 >= t0) else None

        return {
            "turn_id": self.turn_id,
            "speech_duration_ms": delta(self.t_speech_endpoint, self.t_speech_start),
            "speech_end_to_transcript_ms": delta(self.t_transcribe_end, self.t_speech_endpoint),
            "speech_end_to_first_token_ms": delta(self.t_first_token, self.t_speech_endpoint),
            "speech_end_to_first_audible_ms": delta(self.t_first_playback, self.t_speech_endpoint),
            "transcription_duration_ms": delta(self.t_transcribe_end, self.t_transcribe_start),
            "llm_ttft_ms": delta(self.t_first_token, self.t_llm_start),
            "llm_total_gen_ms": delta(self.t_llm_end, self.t_llm_start),
            "tts_first_chunk_synth_ms": delta(self.t_first_playback, self.t_first_sentence) if self.t_first_playback and self.t_first_sentence else delta(self.t_first_playback, self.t_first_token),
            "total_playback_duration_ms": delta(self.t_playback_end, self.t_first_playback),
            "playback_end_to_relisten_ms": delta(self.t_relisten_active, self.t_playback_end),
            "cancelled": self.cancelled,
            "cancel_reason": self.cancel_reason,
        }


class SentenceSegmenter:
    """
    Streams tokens into natural sentences for immediate TTS synthesis.
    Ensures the first sentence is dispatched quickly without waiting for the full LLM answer.
    Handles abbreviations, decimal points, clause breaks, and code block suppression.
    """
    ABBREVIATIONS = {'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sr.', 'jr.', 'vs.', 'etc.', 'e.g.', 'i.e.', 'u.s.'}

    def __init__(self, min_first_chars=10, min_subsequent_chars=35):
        self.min_first_chars = min_first_chars
        self.min_subsequent_chars = min_subsequent_chars
        self.buffer = ""
        self.is_first_sentence = True
        self.in_code_block = False
        self.code_announced = False

    def add_token(self, token: str) -> list[str]:
        # Code fence tracking
        if "```" in token:
            if not self.in_code_block:
                self.in_code_block = True
                ready = []
                pre = self._extract_one()
                if pre:
                    ready.append(pre)
                if not self.code_announced:
                    self.code_announced = True
                    ready.append("The complete implementation is displayed on your screen.")
                return ready
            else:
                self.in_code_block = False
                self.buffer = ""
                return []

        if self.in_code_block:
            # Skip code tokens from spoken stream (full code remains in UI chat)
            return []

        self.buffer += token
        ready = []
        while True:
            sentence = self._extract_one()
            if sentence:
                ready.append(sentence)
                self.is_first_sentence = False
            else:
                break
        return ready

    def flush(self) -> list[str]:
        if self.in_code_block:
            self.buffer = ""
            return []
        clean = self.buffer.strip()
        self.buffer = ""
        return [clean] if clean else []

    def _extract_one(self) -> str:
        text = self.buffer
        min_chars = self.min_first_chars if self.is_first_sentence else self.min_subsequent_chars

        match = re.search(r'([.!?]+|\n+)\s+', text)
        if match:
            end_pos = match.end()
            candidate = text[:match.start() + len(match.group(1))].strip()
            last_word = candidate.split()[-1].lower() if candidate.split() else ''
            if last_word in self.ABBREVIATIONS:
                return None
            if re.search(r'\d\.\d', candidate[-4:]):
                return None

            if len(candidate) >= min_chars or self.is_first_sentence:
                self.buffer = text[end_pos:]
                return candidate

        # If buffer is getting long without punctuation, split at clause break (, ; :)
        if len(text) > 100:
            clause_match = re.search(r'([,;:]+)\s+', text[min_chars:])
            if clause_match:
                split_idx = min_chars + clause_match.end()
                candidate = text[:min_chars + clause_match.start() + len(clause_match.group(1))].strip()
                self.buffer = text[split_idx:]
                return candidate

        return None


class VoiceTurn:
    """Represents a single conversational turn with strict identity, metrics, and cancellation."""
    def __init__(self, session_id: str, turn_id: int):
        self.session_id = session_id
        self.turn_id = turn_id
        self.state = "CAPTURING"  # CAPTURING, TRANSCRIBING, THINKING, SPEAKING, COMPLETE, CANCELLED
        self.transcript = ""
        self.reply = ""
        self.partial_reply = ""
        self.cancel_event = threading.Event()
        self.created_at = time.time()
        self.metrics = TurnMetrics(session_id, turn_id)

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def cancel(self, reason: str = "user_cancel"):
        self.cancel_event.set()
        self.state = "CANCELLED"
        self.metrics.cancelled = True
        self.metrics.cancel_reason = reason


class VoiceService:
    def __init__(
        self,
        transcribe_fn=None,
        chat_fn=None,
        commit_history_fn=None,
        wake_transcribe_fn=None,
        microphone_device=None,
        sample_rate=16000,
        silence_threshold=1.2,
        session_timeout=60.0,
        max_duration=30.0,
        energy_threshold=0.008,
        greeting_text="Hi! How can I assist you today?",
        client_lease_timeout=25.0,
    ):
        self.transcribe_fn = transcribe_fn
        self.chat_fn = chat_fn
        self.commit_history_fn = commit_history_fn
        self.wake_transcribe_fn = wake_transcribe_fn
        self.microphone_device = microphone_device
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.session_timeout = session_timeout
        self.max_duration = max_duration
        self.energy_threshold = energy_threshold
        self.greeting_text = greeting_text
        self.client_lease_timeout = client_lease_timeout

        self.state = VoiceState.DISABLED
        self.enabled = False
        self.session_active = False
        self.session_id = None
        self.session_last_activity = 0.0
        self.error_message = None

        # Turn identity tracking
        self.turn_counter = 0
        self.current_turn = None
        self._turn_lock = threading.Lock()

        # Lifecycle & Thread safety
        self._worker_generation = 0
        self._audio_queue = queue.Queue()
        self._stream = None
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._muted = False

        # Client lease tracking (prevents orphan mic processing when tabs close)
        self._last_client_heartbeat = time.time()

        # Telemetry for HUD sync
        self.turn_id = 0
        self.last_transcript = ""
        self.last_reply = ""
        self.partial_reply = ""
        self.current_spoken_sentence = ""

        # Initialize TTS Engine (Kokoro neural with SAPI fallback)
        self.tts_engine = TTSEngine()

        # Low-latency wake word detectors: openWakeWord ONNX + Whisper tiny.en acoustic spotter
        self._oww_model = None
        self._wake_whisper_model = None
        self._init_wake_word_model()

        # Activation Chime (Futuristic two-tone sound)
        self._chime = self._create_chime()

    def _init_wake_word_model(self):
        # 1. Initialize openWakeWord ONNX engine (instantaneous ~50ms detection)
        models_to_load = []
        custom_paths = [
            Path(__file__).parent / "models" / "hey_mark.onnx",
            Path(__file__).parent / "hey_mark.onnx",
        ]
        for p in custom_paths:
            if p.is_file():
                models_to_load.append(str(p))

        # Always bundle hey_jarvis for native ONNX keyword spotting (~50ms)
        models_to_load.append("hey_jarvis")

        try:
            from openwakeword.model import Model
            print(f"[VoiceService] Loading openWakeWord ONNX engine with {models_to_load}...")
            self._oww_model = Model(wakeword_models=models_to_load, inference_framework="onnx")
            print(f"[VoiceService] openWakeWord ONNX loaded ({list(self._oww_model.models.keys())}).")
        except Exception as exc:
            print(f"[VoiceService] Note: openWakeWord init fallback: {exc}")
            self._oww_model = None

        # 2. Initialize Whisper tiny.en for ultra-fast acoustic wake spotting (~85ms on CUDA)
        if self.wake_transcribe_fn is None:
            try:
                from faster_whisper import WhisperModel
                cuda_avail = False
                try:
                    import ctranslate2
                    cuda_avail = (ctranslate2.get_cuda_device_count() > 0)
                except Exception:
                    pass
                device = "cuda" if cuda_avail else "cpu"
                compute_type = "float16" if cuda_avail else "int8"
                print(f"[VoiceService] Loading Whisper tiny.en ({device}, {compute_type}) for low-latency acoustic wake spotting...")
                try:
                    self._wake_whisper_model = WhisperModel("tiny.en", device=device, compute_type=compute_type)
                except Exception as exc:
                    if device != "cpu":
                        print(f"[VoiceService] Whisper tiny.en CUDA failed ({exc}); falling back to CPU...")
                        self._wake_whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
                    else:
                        raise
                print(f"[VoiceService] Whisper tiny.en acoustic wake spotter ready ({self._wake_whisper_model.model.device}).")
            except Exception as exc:
                print(f"[VoiceService] Note: Whisper tiny.en spotter fallback: {exc}")
                self._wake_whisper_model = None
        else:
            print("[VoiceService] Using custom wake_transcribe_fn for acoustic wake spotting.")

    def _create_chime(self):
        t1 = np.linspace(0, 0.08, int(self.sample_rate * 0.08), False)
        t2 = np.linspace(0, 0.10, int(self.sample_rate * 0.10), False)
        tone1 = 0.22 * np.sin(2 * np.pi * 587.33 * t1) * np.linspace(1, 0.2, len(t1))
        tone2 = 0.22 * np.sin(2 * np.pi * 880.00 * t2) * np.linspace(1, 0.0, len(t2))
        return np.concatenate([tone1, tone2]).astype(np.float32)

    def heartbeat(self):
        """Refreshes client lease heartbeat."""
        self._last_client_heartbeat = time.time()

    def _log_turn(self, turn: VoiceTurn, event: str, detail: str = ""):
        elapsed = time.time() - turn.created_at
        msg = f"[VoiceTurn] session={turn.session_id} turn={turn.turn_id} state={turn.state} event={event} elapsed={elapsed:.2f}s"
        if detail:
            msg += f" {detail}"
        print(msg)

    def _audio_callback(self, indata, frames, time_info, status):
        if not self._muted and self.enabled and not self._stop_event.is_set():
            self._audio_queue.put(indata.copy())

    def _drain_queue(self):
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def _refractory_drain(self, duration: float = 0.6):
        """Mutes and drains queue to prevent assistant speech echo/reverb from triggering input."""
        self._muted = True
        t_end = time.time() + duration
        while time.time() < t_end:
            self._drain_queue()
            time.sleep(0.03)
        self._drain_queue()
        self._muted = False

    def cancel_active_turn(self, reason: str = "user_cancel"):
        """Authoritatively cancels active voice turn and invalidates ongoing processing."""
        with self._turn_lock:
            if self.current_turn and not self.current_turn.is_cancelled():
                self._log_turn(self.current_turn, "CANCELLED", f"reason={reason}")
                self.current_turn.cancel(reason)
        self.tts_engine.stop()
        self._drain_queue()

    def stop_speaking(self):
        """Interrupts current spoken response immediately and safely resumes listening."""
        print("[VoiceService] Stop speaking requested.")
        self.tts_engine.stop()
        with self._turn_lock:
            if self.current_turn and self.current_turn.state == "SPEAKING":
                self._log_turn(self.current_turn, "STOP_SPEAKING")
        self._refractory_drain(duration=0.3)
        if self.state == VoiceState.SPEAKING:
            if self.session_active and self.enabled:
                self.state = VoiceState.LISTENING_FOR_COMMAND
            elif self.enabled:
                self.state = VoiceState.WAKE_WORD_LISTENING
            else:
                self.state = VoiceState.DISABLED

    def end_session(self, speak_goodbye: bool = True):
        """Ends the active conversation session and returns to standby wake-word listening."""
        if not self.session_active:
            return

        print("[VoiceService] Ending conversation session.")
        self.cancel_active_turn("end_session")
        self.session_active = False

        if speak_goodbye and self.enabled and not self._stop_event.is_set():
            self.state = VoiceState.SPEAKING
            self._speak_text("Goodbye! Feel free to say Hey Mark whenever you need me.")

        if self.enabled:
            self.state = VoiceState.WAKE_WORD_LISTENING
            print("[VoiceService] Returned to WAKE WORD LISTENING standby.")
        else:
            self.state = VoiceState.DISABLED

    def stop(self, reason: str = "user_stop"):
        """
        Authoritative emergency shutdown.
        Stops playback, invalidates workers via generation counter, releases stream safely.
        """
        print(f"[VoiceService] Authoritative stop initiated ({reason})...")
        self._stop_event.set()
        self._worker_generation += 1  # Invalidate any running worker immediately

        self.enabled = False
        self.session_active = False

        # Cancel turn and stop playback
        self.cancel_active_turn(reason)
        self.tts_engine.stop()

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.5)
        self._worker_thread = None

        self._drain_queue()
        self.state = VoiceState.DISABLED
        print(f"[VoiceService] Service stopped cleanly ({reason}).")

    def start(self):
        """Starts hands-free listener with fresh session identity and generation guard."""
        if self._worker_thread and self._worker_thread.is_alive() and self.enabled:
            print("[VoiceService] Worker already running.")
            return

        self._stop_event.clear()
        self._worker_generation += 1
        current_gen = self._worker_generation
        self.session_id = f"s_{uuid.uuid4().hex[:8]}"
        self.enabled = True
        self.session_active = False
        self.error_message = None
        self.state = VoiceState.WAKE_WORD_LISTENING
        self._last_client_heartbeat = time.time()

        self._drain_queue()

        try:
            self._stream = sd.InputStream(
                device=self.microphone_device,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=1024,
                callback=self._audio_callback,
            )
            self._stream.start()
            print(f"[VoiceService] Audio stream started on device {self.microphone_device or 'default'}.")
        except Exception as exc:
            self.state = VoiceState.ERROR
            self.error_message = f"Microphone stream failed: {exc}"
            self.enabled = False
            print(f"[VoiceService] Error starting audio stream: {exc}")
            return

        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(current_gen,),
            daemon=True
        )
        self._worker_thread.start()
        print(f"[VoiceService] Hands-free listener started (session={self.session_id}, gen={current_gen}).")

    def set_enabled(self, val: bool):
        if val and not self.enabled:
            self.start()
        elif not val and self.enabled:
            self.stop(reason="user_toggle_disabled")

    def get_status(self):
        self.heartbeat()
        turn_status = "IDLE"
        turn_metrics = None
        with self._turn_lock:
            if self.current_turn:
                turn_status = self.current_turn.state
                turn_metrics = self.current_turn.metrics.summary()

        # Authoritative state truth-check: if audio playback ended, state cannot remain SPEAKING
        effective_state = self.state
        if effective_state == VoiceState.SPEAKING and not self.tts_engine.is_speaking():
            effective_state = VoiceState.LISTENING_FOR_COMMAND if (self.session_active and self.enabled) else VoiceState.WAKE_WORD_LISTENING
            self.state = effective_state

        return {
            "state": self.state,
            "hands_free": self.enabled,
            "session_active": self.session_active,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "turn_status": turn_status,
            "last_transcript": self.last_transcript,
            "last_reply": self.last_reply,
            "partial_reply": getattr(self, "partial_reply", self.last_reply),
            "current_spoken_sentence": getattr(self, "current_spoken_sentence", ""),
            "metrics": turn_metrics,
            "error": self.error_message,
        }

    def _play_activation_chime(self):
        try:
            sd.play(self._chime, samplerate=self.sample_rate)
        except Exception as exc:
            print(f"[VoiceService] Chime playback warning: {exc}")

    def _speak_text(self, text: str, cancel_token=None, job_id=None):
        if not text:
            return

        self.state = VoiceState.SPEAKING
        self._muted = True
        try:
            self.tts_engine.speak(text, job_id=job_id, cancel_token=cancel_token)
        except Exception as exc:
            print(f"[VoiceService] Spoken output error: {exc}")
        finally:
            self._refractory_drain(duration=0.5)

    def _transcribe(self, audio, beam_size=2):
        if not self.transcribe_fn:
            return ""
        try:
            if isinstance(audio, np.ndarray) and audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            return self.transcribe_fn(audio, beam_size=beam_size)
        except TypeError:
            return self.transcribe_fn(audio)
        except Exception as exc:
            print(f"[VoiceService] Transcription call error: {exc}")
            return ""

    def _wake_transcribe(self, audio: np.ndarray) -> str:
        """High-speed low-latency transcription for wake-word acoustic spotting."""
        if self.wake_transcribe_fn is not None:
            try:
                if isinstance(audio, np.ndarray) and audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                return self.wake_transcribe_fn(audio)
            except Exception as exc:
                print(f"[VoiceService] Custom wake transcribe error: {exc}")

        if self._wake_whisper_model is not None:
            try:
                if isinstance(audio, np.ndarray) and audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                if len(audio) == 0 or np.max(np.abs(audio)) < 1e-4:
                    return ""
                segments, _ = self._wake_whisper_model.transcribe(
                    audio,
                    language="en",
                    beam_size=1,
                    vad_filter=False,
                    condition_on_previous_text=False,
                )
                return " ".join(s.text.strip() for s in segments).strip()
            except Exception as exc:
                print(f"[VoiceService] Fast wake transcribe error: {exc}")

        return self._transcribe(audio, beam_size=1)

    def _worker_loop(self, gen: int):
        print(f"[VoiceService] Worker loop active (gen={gen}).")
        oww_pcm_buffer = np.array([], dtype=np.int16)
        noise_floor = self.energy_threshold
        wake_pre_roll = []
        wake_chunks = []
        wake_speech_started = False
        wake_silence_start_time = None
        wake_speech_start_time = None
        last_interim_check_time = 0.0

        while not self._stop_event.is_set() and self.enabled and self._worker_generation == gen:
            # Client Lease Check: If no client has contacted in lease timeout, pause listening
            if time.time() - self._last_client_heartbeat > self.client_lease_timeout:
                time.sleep(0.2)
                wake_pre_roll.clear()
                wake_chunks.clear()
                oww_pcm_buffer = np.array([], dtype=np.int16)
                wake_speech_started = False
                continue

            # 1. STANDBY: Listen for "Hey Mark" / "Hey Jarvis"
            if self.state == VoiceState.WAKE_WORD_LISTENING:
                try:
                    chunk = self._audio_queue.get(timeout=0.08)
                except queue.Empty:
                    continue

                if self._worker_generation != gen or self._stop_event.is_set():
                    break

                flat = chunk.flatten()
                rms = float(np.sqrt(np.mean(flat ** 2)))

                # Stream to openWakeWord (1280 samples = 80ms chunks, ~1.4ms ONNX inference)
                pcm_data = (flat * 32767.0).clip(-32768, 32767).astype(np.int16)
                oww_pcm_buffer = np.concatenate([oww_pcm_buffer, pcm_data])
                oww_triggered = False
                oww_model_hit = ""

                while len(oww_pcm_buffer) >= 1280:
                    oww_frame = oww_pcm_buffer[:1280]
                    oww_pcm_buffer = oww_pcm_buffer[1280:]
                    if self._oww_model:
                        try:
                            self._oww_model.predict(oww_frame)
                            for m_name, scores in self._oww_model.prediction_buffer.items():
                                if scores and scores[-1] >= 0.45:
                                    oww_triggered = True
                                    oww_model_hit = m_name
                                    break
                        except Exception:
                            pass
                    if oww_triggered:
                        break

                if oww_triggered and self._worker_generation == gen and not self._stop_event.is_set():
                    print(f"[VoiceService] openWakeWord instantaneous trigger: '{oww_model_hit}'!")
                    wake_pre_roll.clear()
                    oww_pcm_buffer = np.array([], dtype=np.int16)

                    # Preserve trailing audio if continuous command was spoken
                    trailing_audio = np.concatenate(wake_chunks) if (wake_chunks and len(wake_chunks) > 4) else None
                    wake_chunks.clear()
                    wake_speech_started = False
                    wake_silence_start_time = None
                    wake_speech_start_time = None

                    self.session_active = True
                    self.session_last_activity = time.time()
                    self.state = VoiceState.LISTENING_FOR_COMMAND

                    if trailing_audio is not None and len(trailing_audio) > int(self.sample_rate * 0.3):
                        self.barge_in_audio = trailing_audio

                    threading.Thread(target=self._play_activation_chime, daemon=True).start()
                    continue

                # Streaming acoustic spotter for "Hey Mark" / "Hi Mark"
                if not wake_speech_started:
                    noise_floor = 0.96 * noise_floor + 0.04 * rms
                    speech_thresh = max(self.energy_threshold, noise_floor * 2.2)
                    if rms > speech_thresh:
                        wake_speech_started = True
                        wake_chunks = list(wake_pre_roll)
                        wake_chunks.append(flat)
                        wake_speech_start_time = time.time()
                        wake_silence_start_time = None
                        last_interim_check_time = time.time()
                    else:
                        wake_pre_roll.append(flat)
                        if len(wake_pre_roll) > 8:
                            wake_pre_roll.pop(0)
                else:
                    wake_chunks.append(flat)
                    silence_thresh = max(self.energy_threshold * 0.65, noise_floor * 1.35)
                    if rms < silence_thresh:
                        if wake_silence_start_time is None:
                            wake_silence_start_time = time.time()
                    else:
                        wake_silence_start_time = None

                    now = time.time()
                    speech_dur = now - wake_speech_start_time
                    silence_dur = (now - wake_silence_start_time) if wake_silence_start_time else 0.0

                    check_wake_candidate = False
                    is_interim = False

                    # Trigger condition A: Silence endpoint after speech (pause after "Hey Mark", 140ms endpoint)
                    if silence_dur >= 0.14 and speech_dur >= 0.25:
                        check_wake_candidate = True

                    # Trigger condition B: Early interim check on continuous utterance (user speaking sentence without pause)
                    elif speech_dur >= 0.65 and (now - last_interim_check_time >= 0.35):
                        check_wake_candidate = True
                        is_interim = True
                        last_interim_check_time = now

                    # Trigger condition C: Max window cap for wake phrase (tight 1.8s bound, avoids 15s hang)
                    elif speech_dur >= 1.80:
                        check_wake_candidate = True

                    if check_wake_candidate:
                        total_audio = np.concatenate(wake_chunks) if wake_chunks else np.array([], dtype=np.float32)

                        if len(total_audio) >= int(self.sample_rate * 0.35):
                            # Slice first 1.2s if interim to spot the wake phrase immediately
                            eval_audio = total_audio[:int(self.sample_rate * 1.2)] if is_interim else total_audio
                            detected, extra_command = self._check_wake_word_utterance(eval_audio, gen)

                            if detected and self._worker_generation == gen and not self._stop_event.is_set():
                                print(f"[VoiceService] Wake word 'Hey Mark' detected! (interim={is_interim}, command: '{extra_command}')")
                                wake_pre_roll.clear()
                                oww_pcm_buffer = np.array([], dtype=np.int16)
                                self.session_active = True
                                self.session_last_activity = time.time()

                                # Preserve any command audio trailing after the wake phrase (wake phrase is ~0.65s)
                                trailing_audio = total_audio[int(self.sample_rate * 0.65):] if is_interim else None
                                wake_chunks.clear()
                                wake_speech_started = False
                                wake_silence_start_time = None
                                wake_speech_start_time = None

                                if extra_command and len(extra_command.strip()) > 3:
                                    print(f"[VoiceService] Inline command detected: '{extra_command}'")
                                    threading.Thread(target=self._play_activation_chime, daemon=True).start()
                                    self.turn_counter += 1
                                    turn = VoiceTurn(self.session_id, self.turn_counter)
                                    turn.transcript = extra_command
                                    now = time.time()
                                    turn.metrics.t_speech_start = wake_speech_start_time or (now - speech_dur)
                                    turn.metrics.t_speech_endpoint = wake_silence_start_time or (now - 0.20)
                                    turn.metrics.t_transcribe_start = now
                                    turn.metrics.t_transcribe_end = now
                                    with self._turn_lock:
                                        self.current_turn = turn
                                    self._process_command(turn, gen)
                                else:
                                    self.state = VoiceState.LISTENING_FOR_COMMAND
                                    if trailing_audio is not None and len(trailing_audio) > int(self.sample_rate * 0.25):
                                        self.barge_in_audio = trailing_audio
                                    threading.Thread(target=self._play_activation_chime, daemon=True).start()
                                    self.session_last_activity = time.time()
                                    print("[VoiceService] Wake word active. Now LISTENING FOR COMMAND.")
                                continue

                        if not is_interim:
                            # Speech ended or max duration reached without wake word detection: reset
                            wake_speech_started = False
                            wake_silence_start_time = None
                            wake_speech_start_time = None
                            wake_chunks.clear()

            # 2. ACTIVE SESSION: Listen for commands
            elif self.state == VoiceState.LISTENING_FOR_COMMAND:
                wake_speech_started = False
                wake_chunks = []
                wake_pre_roll = []
                self._handle_active_command_turn(gen)

            else:
                time.sleep(0.04)

        print(f"[VoiceService] Worker loop ended (gen={gen}).")

    def _check_wake_word_utterance(self, audio: np.ndarray, gen: int):
        if len(audio) == 0 or self._worker_generation != gen or self._stop_event.is_set():
            return False, ""

        if isinstance(audio, np.ndarray) and audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # 1. Custom openWakeWord if available
        if self._oww_model:
            try:
                pcm = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)
                for i in range(0, len(pcm) - 1280, 1280):
                    self._oww_model.predict(pcm[i:i + 1280])
                    for name, score in self._oww_model.prediction_buffer.items():
                        if score and score[-1] >= 0.45:
                            return True, ""
            except Exception:
                pass

        # 2. High-speed acoustic wake-word spotting
        transcript = self._wake_transcribe(audio)
        if not transcript or self._worker_generation != gen:
            return False, ""

        clean = re.sub(r"[^\w\s]", "", transcript.lower()).strip()
        print(f"[VoiceService] Standby speech recognized: '{transcript}'")

        # Check for explicit session exit commands first (e.g. "goodbye mark" should never activate)
        for pattern in EXIT_PATTERNS:
            if re.search(pattern, clean):
                print(f"[VoiceService] Standby exit phrase detected: '{clean}'")
                return False, ""

        wake_match = WAKE_WORD_PATTERN.search(clean)
        if wake_match:
            parts = clean[wake_match.end():].strip()
            return True, parts

        return False, ""

    def _handle_active_command_turn(self, gen: int):
        """Captures a single speech turn within an active multi-turn session."""
        self._drain_queue()
        pre_roll_chunks = []
        speech_chunks = []
        speech_started = False
        silence_start_time = None
        turn_start_time = time.time()

        # Check if pre-roll audio was captured from barge-in interruption of previous turn
        barge_in = getattr(self, "barge_in_audio", None)
        if barge_in is not None and len(barge_in) > 0:
            speech_chunks = [barge_in]
            self.barge_in_audio = None
            speech_started = True
            turn_start_time = time.time()
            self.session_last_activity = time.time()
            self.turn_counter += 1
            turn = VoiceTurn(self.session_id, self.turn_counter)
            turn.state = "CAPTURING"
            turn.metrics.t_speech_start = time.time()
            self.turn_id = turn.turn_id
            with self._turn_lock:
                self.current_turn = turn
            self.last_transcript = ""
            self.partial_reply = ""
            self.last_reply = ""
            self.current_spoken_sentence = ""

        while not self._stop_event.is_set() and self.enabled and self.session_active and self._worker_generation == gen:
            # Check session inactivity timeout (default 60s)
            if not speech_started:
                silence_duration = time.time() - self.session_last_activity
                if silence_duration >= self.session_timeout:
                    print(f"[VoiceService] Inactivity timeout ({self.session_timeout}s). Ending session.")
                    self.end_session(speak_goodbye=False)
                    return

            try:
                chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._worker_generation != gen or self._stop_event.is_set():
                return

            flat = chunk.flatten()
            rms = np.sqrt(np.mean(flat ** 2))

            if not speech_started:
                if rms > self.energy_threshold:
                    print("[VoiceService] User speech started.")
                    speech_started = True
                    speech_chunks = list(pre_roll_chunks)
                    speech_chunks.append(flat)
                    self.session_last_activity = time.time()
                    silence_start_time = None

                    # Allocate authoritative turn immediately when speech starts
                    self.turn_counter += 1
                    turn = VoiceTurn(self.session_id, self.turn_counter)
                    turn.state = "CAPTURING"
                    turn.metrics.t_speech_start = time.time()
                    self.turn_id = turn.turn_id
                    with self._turn_lock:
                        self.current_turn = turn
                    self.last_transcript = ""
                    self.partial_reply = ""
                    self.last_reply = ""
                    self.current_spoken_sentence = ""
                else:
                    pre_roll_chunks.append(flat)
                    if len(pre_roll_chunks) > 8:
                        pre_roll_chunks.pop(0)
            else:
                speech_chunks.append(flat)
                if rms > self.energy_threshold:
                    self.session_last_activity = time.time()
                    silence_start_time = None
                else:
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    elif time.time() - silence_start_time >= self.silence_threshold:
                        print(f"[VoiceService] Silence detected ({self.silence_threshold}s). End of utterance.")
                        break

            # Safeguard max turn duration
            if time.time() - turn_start_time > self.max_duration:
                print("[VoiceService] Max duration reached for turn.")
                break

        if not speech_chunks or not speech_started or not self.session_active or self._worker_generation != gen:
            return

        with self._turn_lock:
            turn = self.current_turn

        if not turn:
            self.turn_counter += 1
            turn = VoiceTurn(self.session_id, self.turn_counter)
            with self._turn_lock:
                self.current_turn = turn

        turn.metrics.t_speech_endpoint = silence_start_time if silence_start_time is not None else time.time()
        self._log_turn(turn, "CAPTURE_COMPLETE", f"chunks={len(speech_chunks)}")

        # 1. PROCESSING (TRANSCRIBING)
        self.state = VoiceState.PROCESSING
        turn.state = "TRANSCRIBING"
        turn.metrics.t_transcribe_start = time.time()
        self._drain_queue()
        audio_data = np.concatenate(speech_chunks)

        if turn.is_cancelled() or not self.enabled or self._worker_generation != gen:
            self._log_turn(turn, "DISCARDED_BEFORE_TRANSCRIBE")
            return

        self._log_turn(turn, "TRANSCRIBE_START")
        transcript = self._transcribe(audio_data, beam_size=1).strip()
        turn.metrics.t_transcribe_end = time.time()
        self._log_turn(turn, "TRANSCRIBE_COMPLETE", f"chars={len(transcript)}")

        if turn.is_cancelled() or not self.enabled or self._worker_generation != gen:
            self._log_turn(turn, "DISCARDED_AFTER_TRANSCRIBE")
            return

        clean_transcript = re.sub(r"[^\w\s]", "", transcript.lower()).strip()
        if len(clean_transcript) < 2:
            print(f"[VoiceService] Empty or noise transcript ('{transcript}'). Discarded.")
            turn.cancel("empty_transcript")
            self.state = VoiceState.LISTENING_FOR_COMMAND
            return

        print(f"[VoiceService] Recognized: '{transcript}'")

        # Check for explicit session exit commands
        for pattern in EXIT_PATTERNS:
            if re.search(pattern, clean_transcript):
                self._log_turn(turn, "EXIT_PHRASE")
                self.end_session(speak_goodbye=True)
                return

        # Strip repeated wake phrase if user said "Hey Mark, <command>" in active session
        wake_match = WAKE_WORD_PATTERN.search(clean_transcript)
        if wake_match and wake_match.start() == 0:
            stripped = clean_transcript[wake_match.end():].strip()
            if stripped:
                transcript = stripped

        turn.transcript = transcript
        self.last_transcript = transcript
        self.turn_id = turn.turn_id
        self._process_command(turn, gen)

    def _process_command(self, turn: VoiceTurn, gen: int):
        if turn.is_cancelled() or not self.enabled or self._worker_generation != gen:
            self._log_turn(turn, "DISCARDED_BEFORE_THINKING")
            return

        turn.state = "THINKING"
        self.state = VoiceState.PROCESSING
        turn.metrics.t_llm_start = time.time()
        self.session_last_activity = time.time()
        self._log_turn(turn, "THINKING_START")

        # Telemetry updates: transcript is available immediately
        self.last_transcript = turn.transcript
        self.turn_id = turn.turn_id
        self.last_reply = ""
        self.partial_reply = ""
        self.current_spoken_sentence = ""

        # Sentence buffer & queues for true concurrent streaming
        sentence_queue = queue.Queue()
        segmenter = SentenceSegmenter()
        streamed_pieces = []

        def on_playback_started():
            if not turn.is_cancelled() and self._worker_generation == gen:
                if turn.metrics.t_first_playback == 0.0:
                    turn.metrics.t_first_playback = time.time()
                self.state = VoiceState.SPEAKING
                turn.state = "SPEAKING"
                self._log_turn(turn, "SPEAKING_START")

        def on_sentence_playing(sentence_text):
            if not turn.is_cancelled() and self._worker_generation == gen:
                self.current_spoken_sentence = sentence_text

        # Start concurrent TTS streaming player thread
        tts_finished = threading.Event()

        def tts_player_worker():
            try:
                turn.metrics.t_tts_start = time.time()
                if hasattr(self.tts_engine, "speak_stream"):
                    self.tts_engine.speak_stream(
                        sentence_queue,
                        cancel_token=turn.cancel_event,
                        job_id=f"turn_{turn.turn_id}",
                        on_playback_start=on_playback_started,
                        on_sentence_play=on_sentence_playing
                    )
                else:
                    collected_sents = []
                    while True:
                        s = sentence_queue.get()
                        if s is None:
                            break
                        collected_sents.append(s)
                    full_text = " ".join(collected_sents)
                    on_playback_started()
                    self.tts_engine.speak(
                        full_text,
                        cancel_token=turn.cancel_event,
                        job_id=f"turn_{turn.turn_id}"
                    )
            except Exception as e:
                print(f"[VoiceService] TTS worker error: {e}")
            finally:
                tts_finished.set()

        tts_thread = threading.Thread(target=tts_player_worker, daemon=True)
        tts_thread.start()

        # Token streaming callback from LLM
        def on_token(token):
            if turn.is_cancelled() or self._stop_event.is_set():
                return
            if turn.metrics.t_first_token == 0.0:
                turn.metrics.t_first_token = time.time()
            streamed_pieces.append(token)
            accumulated = "".join(streamed_pieces)
            turn.reply = accumulated
            self.last_reply = accumulated
            self.partial_reply = accumulated

            ready_sentences = segmenter.add_token(token)
            for s in ready_sentences:
                if not turn.is_cancelled():
                    if turn.metrics.t_first_sentence == 0.0:
                        turn.metrics.t_first_sentence = time.time()
                    sentence_queue.put(s)

        # Run chat inference
        reply = ""
        if self.chat_fn:
            try:
                try:
                    reply = self.chat_fn(turn.transcript, stream_callback=on_token)
                except TypeError:
                    reply = self.chat_fn(turn.transcript)
            except Exception as exc:
                print(f"[VoiceService] Chat error: {exc}")
                reply = "I apologize, I encountered an error while processing that."

        turn.metrics.t_llm_end = time.time()

        if turn.is_cancelled() or not self.enabled or self._worker_generation != gen:
            self._log_turn(turn, "DISCARDED_AFTER_THINKING")
            sentence_queue.put(None)
            self.last_reply = ""
            self.partial_reply = ""
            if tts_thread and tts_thread.is_alive():
                tts_thread.join(timeout=0.2)
            return

        # If chat_fn returned bulk without streaming tokens
        if not streamed_pieces and reply:
            streamed_pieces.append(reply)
            for s in segmenter.add_token(reply):
                sentence_queue.put(s)

        # Flush any remaining tokens in buffer
        for remaining in segmenter.flush():
            if turn.metrics.t_first_sentence == 0.0:
                turn.metrics.t_first_sentence = time.time()
            sentence_queue.put(remaining)
        sentence_queue.put(None)  # Sentinel to end TTS

        full_reply = "".join(streamed_pieces).strip() or reply or "I didn't receive a response."
        turn.reply = full_reply
        self.last_reply = full_reply
        self.partial_reply = full_reply

        self._log_turn(turn, "THINKING_COMPLETE", f"reply_chars={len(full_reply)}")

        # Active Barge-In and Interruption Supervisor Loop
        barge_in_chunks = []
        barge_in_consecutive = 0
        barge_in_thresh = max(0.025, self.energy_threshold * 2.5)

        while tts_thread.is_alive() and not turn.is_cancelled() and not self._stop_event.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.04)
                flat = chunk.flatten()
                rms = np.sqrt(np.mean(flat ** 2))
                if self.tts_engine.is_speaking():
                    if rms > barge_in_thresh:
                        barge_in_consecutive += 1
                        barge_in_chunks.append(flat)
                        if barge_in_consecutive >= 3:
                            print(f"[VoiceService] Barge-in detected (rms={rms:.4f})! Halting speech playback.")
                            turn.cancel("barge_in")
                            self.tts_engine.stop()
                            try:
                                sd.stop()
                            except Exception:
                                pass
                            self.barge_in_audio = np.concatenate(barge_in_chunks) if barge_in_chunks else None
                            break
                    else:
                        barge_in_consecutive = max(0, barge_in_consecutive - 1)
                        if len(barge_in_chunks) > 4:
                            barge_in_chunks.pop(0)
            except queue.Empty:
                continue

        # Wait for TTS audio playback to complete or be cancelled
        if tts_thread and tts_thread.is_alive():
            tts_thread.join(timeout=0.6)

        turn.metrics.t_playback_end = time.time()
        self._log_turn(turn, "SPEAKING_COMPLETE")

        if turn.is_cancelled() or not self.enabled or self._worker_generation != gen:
            self._log_turn(turn, "DISCARDED_AFTER_SPEAKING")
            self.state = VoiceState.LISTENING_FOR_COMMAND if (self.enabled and self.session_active) else VoiceState.WAKE_WORD_LISTENING
            return

        turn.state = "COMPLETE"
        self._log_turn(turn, "COMPLETE")

        # Commit to shared history only for authoritative, valid turn
        if self.commit_history_fn:
            try:
                self.commit_history_fn(turn.transcript, full_reply)
            except Exception as exc:
                print(f"[VoiceService] Commit history warning: {exc}")

        # Quick Refractory Drain: Eliminate speaker echo before reopening mic
        self._refractory_drain(duration=0.2)
        turn.metrics.t_relisten_active = time.time()

        # Log comprehensive turn metrics for Section 11
        m = turn.metrics.summary()
        print(f"[MarkMetrics] turn={turn.turn_id} "
              f"stt={m.get('transcription_duration_ms')}ms "
              f"ttft={m.get('llm_ttft_ms')}ms "
              f"first_audible_lead={m.get('speech_end_to_first_audible_ms')}ms "
              f"total_gen={m.get('llm_total_gen_ms')}ms "
              f"play_dur={m.get('total_playback_duration_ms')}ms")

        self.current_spoken_sentence = ""
        self.session_last_activity = time.time()
        if self.enabled and self.session_active and self._worker_generation == gen:
            self.state = VoiceState.LISTENING_FOR_COMMAND
            print("[VoiceService] Turn complete. Ready for next command (Session ACTIVE).")
        elif self.enabled and self._worker_generation == gen:
            self.state = VoiceState.WAKE_WORD_LISTENING
            print("[VoiceService] Session ended. In WAKE WORD LISTENING standby.")
        else:
            self.state = VoiceState.DISABLED
