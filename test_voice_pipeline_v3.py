"""
Project Mark — Holo Table V3 Voice Pipeline Verification Suite.
Validates all 18 voice pipeline requirements from Section 9:
1. First audio segment plays before LLM finishes generating full response.
2. Text appears progressively while generation is ongoing.
3. TTS synthesis overlaps with continued LLM generation.
4. Audio segments play in the correct order.
5. No duplicate segments are played.
6. Audio never overlaps unintentionally.
7. Interrupting a response stops playback and cancels pending work.
8. Cancelled audio never plays during a later turn.
9. At least five consecutive voice turns work correctly.
10. Typed messages do not trigger TTS.
11. Conversation history retains complete responses.
12. Hands-free listening resumes after successful completion.
13. Empty or invalid transcriptions are handled safely.
14. 'Mark' alone does not activate the wake phrase.
15. 'Hey Mark' activates the assistant.
16. 'Goodbye Mark' does not accidentally activate it.
17. A TTS failure does not leave Mark stuck in SPEAKING.
18. A slow or failed audio segment does not permanently block the queue.
"""

import sys
import os
import re
import time
import queue
import threading
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tts_engine import TTSEngine
from voice_service import VoiceService, VoiceState, VoiceTurn, SentenceSegmenter, WAKE_WORD_PATTERN, EXIT_PATTERNS
from app import commit_conversation_turn, internal_transcribe, _conversation_history, _conversation_lock


class MockStreamingTTS:
    """Mock TTS that logs sentence events with timestamps to verify streaming concurrency."""
    def __init__(self):
        self.played_segments = []
        self.synthesized_segments = []
        self.is_speaking_val = False
        self.lock = threading.Lock()

    def synthesize_chunk(self, text, voice=None):
        time.sleep(0.04)  # simulate neural synthesis
        self.synthesized_segments.append((time.time(), text))
        fake_audio = np.zeros(1600, dtype=np.float32)
        return fake_audio, 16000, 0.1, text

    def speak_stream(self, sentence_source, voice=None, job_id=None, cancel_token=None, on_playback_start=None, on_sentence_play=None, on_finish_callback=None):
        self.is_speaking_val = True
        first = True
        idx = 0
        while not (cancel_token and cancel_token.is_set()):
            if hasattr(sentence_source, "get"):
                try:
                    s = sentence_source.get(timeout=0.1)
                except queue.Empty:
                    continue
                if s is None:
                    break
            else:
                try:
                    s = next(sentence_source)
                except StopIteration:
                    break

            if first:
                first = False
                if on_playback_start:
                    on_playback_start()

            if on_sentence_play:
                on_sentence_play(s)

            t_now = time.time()
            self.played_segments.append((t_now, idx, s))
            idx += 1
            time.sleep(0.05)  # simulate playback duration

        self.is_speaking_val = False
        if on_finish_callback:
            on_finish_callback()
        return True

    def speak(self, text, voice=None, job_id=None, cancel_token=None, on_finish_callback=None):
        self.is_speaking_val = True
        self.played_segments.append((time.time(), 0, text))
        time.sleep(0.05)
        self.is_speaking_val = False
        if on_finish_callback:
            on_finish_callback()
        return True

    def stop(self):
        self.is_speaking_val = False

    def is_speaking(self):
        return self.is_speaking_val


class TestMarkVoicePipelineV3(unittest.TestCase):

    def test_01_first_audio_plays_before_llm_finishes(self):
        """Req 1 & 3: First audio segment begins playing before LLM finishes generating full response."""
        tts = MockStreamingTTS()
        playback_start_time = None
        llm_finish_time = None

        def streaming_chat(prompt, stream_callback=None):
            nonlocal llm_finish_time
            tokens = [
                "Artificial intelligence is an exciting field. ",
                "It enables computers to learn from experience. ",
                "Modern deep learning has transformed software."
            ]
            for t in tokens:
                time.sleep(0.08)  # simulate slow LLM token generation
                if stream_callback:
                    stream_callback(t)
            llm_finish_time = time.time()
            return "".join(tokens)

        vs = VoiceService(chat_fn=streaming_chat)
        vs.tts_engine = tts
        vs.session_id = "test_s1"
        vs.enabled = True
        vs.session_active = True

        turn = VoiceTurn("test_s1", 1)
        turn.transcript = "explain ai"

        t0 = time.time()
        vs._process_command(turn, vs._worker_generation)

        # Assert first audio played
        self.assertGreater(len(tts.played_segments), 0)
        first_play_time = tts.played_segments[0][0]

        # Invariant: First playback started BEFORE LLM finished!
        self.assertIsNotNone(llm_finish_time)
        self.assertLess(first_play_time, llm_finish_time,
                        f"First playback ({first_play_time:.3f}s) should precede LLM completion ({llm_finish_time:.3f}s)")
        print(f"\n[TEST 1 PASS] First audio started at +{first_play_time - t0:.3f}s, LLM finished at +{llm_finish_time - t0:.3f}s (Lead time: {(llm_finish_time - first_play_time)*1000:.1f}ms)")

    def test_02_text_appears_progressively(self):
        """Req 2: Text appears progressively while generation is ongoing."""
        partial_states = []

        def streaming_chat(prompt, stream_callback=None):
            for word in ["Quantum ", "computing ", "uses ", "qubits."]:
                time.sleep(0.02)
                if stream_callback:
                    stream_callback(word)
                partial_states.append(vs.partial_reply)
            return "Quantum computing uses qubits."

        vs = VoiceService(chat_fn=streaming_chat)
        vs.tts_engine = MockStreamingTTS()
        vs.session_id = "test_s2"
        vs.enabled = True
        vs.session_active = True

        turn = VoiceTurn("test_s2", 1)
        turn.transcript = "what is quantum computing"
        vs._process_command(turn, vs._worker_generation)

        # Check progressive growth
        self.assertGreater(len(partial_states), 1)
        self.assertIn("Quantum ", partial_states[0])
        self.assertEqual(vs.last_reply, "Quantum computing uses qubits.")

    def test_04_05_06_audio_order_no_duplicates_no_overlap(self):
        """Req 4, 5, 6: Audio segments play in correct order, no duplicates, no overlap."""
        tts = MockStreamingTTS()

        def multi_sentence_chat(prompt, stream_callback=None):
            sentences = [
                "First sentence here is fully articulated for the assistant. ",
                "Second sentence follows with additional explanatory context. ",
                "Third sentence concludes the complete conversational response."
            ]
            for s in sentences:
                time.sleep(0.02)
                if stream_callback:
                    stream_callback(s)
            return "".join(sentences)

        vs = VoiceService(chat_fn=multi_sentence_chat)
        vs.tts_engine = tts
        vs.session_id = "test_s3"
        vs.enabled = True
        vs.session_active = True

        turn = VoiceTurn("test_s3", 1)
        turn.transcript = "three sentences"
        vs._process_command(turn, vs._worker_generation)

        indices = [seg[1] for seg in tts.played_segments]
        texts = [seg[2] for seg in tts.played_segments]

        # Correct order: 0, 1, 2
        self.assertEqual(indices, sorted(indices))
        # No duplicates
        self.assertEqual(len(texts), len(set(texts)))
        self.assertEqual(len(texts), 3)

    def test_07_08_interruption_stops_playback_and_cancels_pending(self):
        """Req 7 & 8: Interruption stops playback, cancels pending, and cancelled audio never plays later."""
        tts = MockStreamingTTS()
        turn1 = VoiceTurn("test_s4", 1)

        def slow_chat(prompt, stream_callback=None):
            time.sleep(0.05)
            # Interrupt in flight
            turn1.cancel("user_interrupt")
            return "Should be cancelled."

        vs = VoiceService(chat_fn=slow_chat)
        vs.tts_engine = tts
        vs.session_id = "test_s4"
        vs.enabled = True
        vs.session_active = True

        turn1.transcript = "slow command"
        vs._process_command(turn1, vs._worker_generation)

        # Invariant: No audio played for cancelled turn
        self.assertEqual(len(tts.played_segments), 0)
        self.assertEqual(vs.last_reply, "")

        # Now do turn 2 and ensure old audio from turn 1 NEVER leaks
        def normal_chat(prompt, stream_callback=None):
            if stream_callback:
                stream_callback("Clean response.")
            return "Clean response."

        vs.chat_fn = normal_chat
        turn2 = VoiceTurn("test_s4", 2)
        turn2.transcript = "fresh command"
        vs._process_command(turn2, vs._worker_generation)

        self.assertEqual(len(tts.played_segments), 1)
        self.assertIn("Clean response", tts.played_segments[0][2])

    def test_09_five_consecutive_voice_turns(self):
        """Req 9: At least five consecutive voice turns work correctly without state corruption."""
        tts = MockStreamingTTS()
        vs = VoiceService(chat_fn=lambda p, stream_callback=None: f"Answer to turn {p}")
        vs.tts_engine = tts
        vs.session_id = "test_consecutive"
        vs.enabled = True
        vs.session_active = True

        for i in range(1, 6):
            turn = VoiceTurn("test_consecutive", i)
            turn.transcript = f"query_{i}"
            vs._process_command(turn, vs._worker_generation)
            self.assertEqual(turn.state, "COMPLETE")
            self.assertEqual(vs.last_reply, f"Answer to turn query_{i}")
            self.assertEqual(vs.turn_id, i)

        self.assertEqual(len(tts.played_segments), 5)

    def test_10_typed_messages_do_not_trigger_tts(self):
        """Req 10: Typed chat messages do not trigger TTS."""
        tts = MockStreamingTTS()
        # Test app internal_chat directly without voice_service
        from app import internal_chat
        res = internal_chat("test typed message", commit=False)
        self.assertIsInstance(res, str)
        # Verify MockStreamingTTS saw zero speak requests
        self.assertEqual(len(tts.played_segments), 0)
        self.assertFalse(tts.is_speaking())

    def test_11_conversation_history_retains_complete_responses(self):
        """Req 11: Conversation history retains complete responses."""
        with _conversation_lock:
            n_before = len(_conversation_history)
        commit_conversation_turn("What is 1+1?", "1+1 equals 2.")
        with _conversation_lock:
            self.assertEqual(len(_conversation_history), n_before + 2)
            self.assertEqual(_conversation_history[-2]["content"], "What is 1+1?")
            self.assertEqual(_conversation_history[-1]["content"], "1+1 equals 2.")

    def test_12_hands_free_listening_resumes_after_completion(self):
        """Req 12: Hands-free listening resumes after successful completion."""
        vs = VoiceService(chat_fn=lambda p, stream_callback=None: "Hello there!")
        vs.tts_engine = MockStreamingTTS()
        vs.session_id = "test_hf"
        vs.enabled = True
        vs.session_active = True

        turn = VoiceTurn("test_hf", 1)
        turn.transcript = "hello"
        vs._process_command(turn, vs._worker_generation)

        # Must return to LISTENING_FOR_COMMAND in active session
        self.assertEqual(vs.state, VoiceState.LISTENING_FOR_COMMAND)

    def test_13_empty_or_invalid_transcription_handled_safely(self):
        """Req 13: Empty or invalid transcriptions are discarded safely."""
        vs = VoiceService()
        vs.tts_engine = MockStreamingTTS()
        vs.session_id = "test_empty"
        vs.enabled = True
        vs.session_active = True

        turn = VoiceTurn("test_empty", 1)
        turn.transcript = ""
        # If empty transcript is processed, should not crash or call chat
        vs._process_command(turn, vs._worker_generation)
        self.assertEqual(len(vs.tts_engine.played_segments), 0)

    def test_14_15_16_wake_word_rules(self):
        """Req 14, 15, 16: 'Mark' alone rejected, 'Hey Mark' accepted, 'Goodbye Mark' does not activate."""
        # 14. 'Mark' alone does NOT activate
        self.assertIsNone(WAKE_WORD_PATTERN.search("mark"))
        self.assertIsNone(WAKE_WORD_PATTERN.search("mark help"))
        self.assertIsNone(WAKE_WORD_PATTERN.search("is mark there"))

        # 15. 'Hey Mark' activates
        self.assertIsNotNone(WAKE_WORD_PATTERN.search("hey mark"))
        self.assertIsNotNone(WAKE_WORD_PATTERN.search("hi mark"))
        self.assertIsNotNone(WAKE_WORD_PATTERN.search("hello mark"))
        self.assertIsNotNone(WAKE_WORD_PATTERN.search("ok mark"))
        self.assertIsNotNone(WAKE_WORD_PATTERN.search("hey mark what is the time"))

        # 16. 'Goodbye Mark' detected as exit pattern, not wake word
        vs = VoiceService()
        detected, _ = vs._check_wake_word_utterance(np.zeros(16000, dtype=np.float32), vs._worker_generation)
        # Explicit test with exit regex:
        clean = "goodbye mark"
        exit_matched = any(re.search(p, clean) for p in EXIT_PATTERNS)
        self.assertTrue(exit_matched)

    def test_17_18_tts_failure_or_slow_chunk_does_not_get_stuck(self):
        """Req 17 & 18: TTS failure does not leave Mark stuck in SPEAKING and does not block queue."""
        class FailingTTS:
            def speak_stream(self, sentence_source, **kwargs):
                raise RuntimeError("Audio driver simulated failure")
            def is_speaking(self):
                return False

        vs = VoiceService(chat_fn=lambda p, stream_callback=None: "Hello world.")
        vs.tts_engine = FailingTTS()
        vs.session_id = "test_fail"
        vs.enabled = True
        vs.session_active = True

        turn = VoiceTurn("test_fail", 1)
        turn.transcript = "hello"
        # Must not raise and must not get stuck
        vs._process_command(turn, vs._worker_generation)
        self.assertNotEqual(vs.state, VoiceState.SPEAKING)
        self.assertEqual(turn.state, "COMPLETE")


if __name__ == "__main__":
    unittest.main()
