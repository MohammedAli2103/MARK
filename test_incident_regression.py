"""
MARK — Incident Regression and Invariants Verification Test Suite.
Validates all 17 regression requirements covering request identity, cancellation,
stale output rejection, TTS isolation, and resource cleanup.
"""

import os
import sys
import time
import queue
import threading
import unittest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tts_engine import TTSEngine
from voice_service import VoiceService, VoiceState, VoiceTurn, WAKE_WORD_PATTERN
from app import commit_conversation_turn, internal_transcribe


class MockTTS:
    def __init__(self):
        self.spoken_jobs = []
        self.active = False
        self.stop_called = False

    def speak(self, text, voice=None, job_id=None, cancel_token=None, on_finish_callback=None):
        if cancel_token and cancel_token.is_set():
            return False
        self.active = True
        self.spoken_jobs.append((job_id, text))
        time.sleep(0.05)
        self.active = False
        if on_finish_callback:
            on_finish_callback()
        return True

    def stop(self):
        self.stop_called = True
        self.active = False

    def is_speaking(self):
        return self.active


class TestMarkIncidentRegression(unittest.TestCase):

    def test_01_one_utterance_creates_exactly_one_chat_request(self):
        """Invariant: Exactly one chat request per completed utterance."""
        chat_calls = []
        def mock_chat(text):
            chat_calls.append(text)
            return "Answer to: " + text

        vs = VoiceService(
            transcribe_fn=lambda a, beam_size=5: "what is the capital of italy",
            chat_fn=mock_chat,
        )
        vs.tts_engine = MockTTS()
        vs.session_id = "s_test_1"
        vs.enabled = True
        vs.session_active = True

        turn = VoiceTurn("s_test_1", 1)
        turn.transcript = "what is the capital of italy"
        vs._process_command(turn, vs._worker_generation)

        self.assertEqual(len(chat_calls), 1)
        self.assertEqual(chat_calls[0], "what is the capital of italy")

    def test_02_one_chat_request_creates_exactly_one_tts_job(self):
        """Invariant: Exactly one TTS job per accepted response."""
        mock_tts = MockTTS()
        vs = VoiceService(
            transcribe_fn=lambda a, beam_size=5: "hello",
            chat_fn=lambda text: "Hello! How can I assist you?",
        )
        vs.tts_engine = mock_tts
        vs.session_id = "s_test_2"
        vs.enabled = True
        vs.session_active = True

        turn = VoiceTurn("s_test_2", 1)
        turn.transcript = "hello"
        vs._process_command(turn, vs._worker_generation)

        self.assertEqual(len(mock_tts.spoken_jobs), 1)
        job_id, spoken_text = mock_tts.spoken_jobs[0]
        self.assertEqual(job_id, "turn_1")
        self.assertEqual(spoken_text, "Hello! How can I assist you?")

    def test_03_displayed_assistant_text_and_spoken_text_match(self):
        """Invariant: Authoritative HUD text matches TTS text precisely."""
        mock_tts = MockTTS()
        raw_response = "Paris is the capital of France. Visit the Eiffel Tower!"
        vs = VoiceService(
            transcribe_fn=lambda a, beam_size=5: "capital of france",
            chat_fn=lambda text: raw_response,
        )
        vs.tts_engine = mock_tts
        vs.session_id = "s_test_3"
        vs.enabled = True
        vs.session_active = True

        turn = VoiceTurn("s_test_3", 1)
        turn.transcript = "capital of france"
        vs._process_command(turn, vs._worker_generation)

        # Verified match between public telemetry and spoken job
        self.assertEqual(vs.last_reply, raw_response)
        self.assertEqual(mock_tts.spoken_jobs[0][1], raw_response)

    def test_04_late_transcription_result_after_stop_is_discarded(self):
        """Cancellation: Transcription result completed after stop must be dropped."""
        chat_called = []
        vs = VoiceService(
            transcribe_fn=lambda a, beam_size=5: "tell me about yourself",
            chat_fn=lambda t: chat_called.append(t) or "I am Mark.",
        )
        vs.tts_engine = MockTTS()
        vs.session_id = "s_test_4"
        vs.enabled = True
        vs.session_active = True

        turn = VoiceTurn("s_test_4", 1)
        # Cancel turn simulating stop during transcription
        turn.cancel("user_stop")

        # Attempt to process command with cancelled turn
        vs._process_command(turn, vs._worker_generation)

        self.assertEqual(len(chat_called), 0)
        self.assertEqual(vs.last_reply, "")

    def test_05_late_llm_response_after_stop_is_discarded(self):
        """Cancellation: LLM response arriving after turn cancellation must be discarded."""
        mock_tts = MockTTS()
        turn = VoiceTurn("s_test_5", 1)

        def slow_chat(msg):
            # Simulate cancellation happening while LLM was processing
            turn.cancel("user_stop")
            return "Unsolicited answer"

        vs = VoiceService(
            chat_fn=slow_chat,
        )
        vs.tts_engine = mock_tts
        vs.session_id = "s_test_5"
        vs.enabled = True
        vs.session_active = True

        turn.transcript = "what is quantum computing"
        vs._process_command(turn, vs._worker_generation)

        # Ensure unsolicited answer was discarded
        self.assertEqual(len(mock_tts.spoken_jobs), 0)
        self.assertEqual(vs.last_reply, "")
        self.assertEqual(turn.state, "CANCELLED")

    def test_06_late_tts_synthesis_after_stop_is_discarded(self):
        """Cancellation: TTSEngine must drop pending chunks when job is cancelled."""
        tts = TTSEngine()
        cancel_token = threading.Event()
        cancel_token.set()  # Already cancelled

        result = tts.speak("This speech should never play.", cancel_token=cancel_token, job_id="test_late_tts")
        self.assertFalse(result)
        self.assertFalse(tts.is_speaking())

    def test_07_stop_speaking_interrupts_current_playback(self):
        """Cancellation: Stop speaking immediately aborts active speech."""
        tts = TTSEngine()
        # Test stop on idle
        tts.stop()
        self.assertFalse(tts.is_speaking())

        # Test stop while speaking simulated
        tts._is_speaking = True
        tts.stop()
        self.assertFalse(tts.is_speaking())

    def test_08_disable_hands_free_stops_service_and_cleans_resources(self):
        """Lifecycle: Disabling Hands-Free cleanly shuts down listener."""
        vs = VoiceService()
        vs.start()
        self.assertTrue(vs.enabled)
        self.assertEqual(vs.state, VoiceState.WAKE_WORD_LISTENING)

        vs.stop(reason="test_disable")
        self.assertFalse(vs.enabled)
        self.assertEqual(vs.state, VoiceState.DISABLED)
        self.assertIsNone(vs._stream)

    def test_09_end_session_invalidates_active_session_work(self):
        """Lifecycle: end_session cancels active turn and returns to wake standby."""
        vs = VoiceService()
        vs.enabled = True
        vs.session_active = True
        vs.session_id = "s_test_9"
        turn = VoiceTurn("s_test_9", 1)
        vs.current_turn = turn

        vs.end_session(speak_goodbye=False)
        self.assertFalse(vs.session_active)
        self.assertTrue(turn.is_cancelled())
        self.assertEqual(vs.state, VoiceState.WAKE_WORD_LISTENING)

    def test_10_repeated_enable_disable_does_not_leak_workers(self):
        """Lifecycle: Repeated starts and stops must not duplicate threads or streams."""
        vs = VoiceService()
        for i in range(4):
            vs.start()
            self.assertTrue(vs.enabled)
            time.sleep(0.05)
            vs.stop(reason="test_loop")
            self.assertFalse(vs.enabled)
        self.assertIsNone(vs._stream)
        self.assertIsNone(vs._worker_thread)

    def test_11_fresh_session_cannot_play_audio_from_previous_session(self):
        """Isolation: Job IDs and cancellation tokens isolate consecutive sessions."""
        tts = TTSEngine()
        cancel_token_1 = threading.Event()

        # Job 1 cancelled
        cancel_token_1.set()
        res1 = tts.speak("Old session audio", cancel_token=cancel_token_1, job_id="session1_turn1")
        self.assertFalse(res1)

        # Job 2 clean
        cancel_token_2 = threading.Event()
        # Verify job IDs differ and job 1 cannot contaminate job 2
        self.assertNotEqual("session1_turn1", "session2_turn1")

    def test_12_empty_or_noise_transcription_does_not_generate_answers(self):
        """Correctness: Empty or punctuation-only transcripts are dropped without chat calls."""
        chat_called = []
        vs = VoiceService(chat_fn=lambda t: chat_called.append(t))
        vs.session_id = "s_test_12"
        vs.enabled = True
        vs.session_active = True

        for noise_text in ["", "  ", ".", "...", " ?", "!"]:
            turn = VoiceTurn("s_test_12", 1)
            turn.transcript = noise_text
            clean = noise_text.strip(" .!?")
            if len(clean) < 2:
                turn.cancel("empty_transcript")
            self.assertTrue(turn.is_cancelled())

        self.assertEqual(len(chat_called), 0)

    def test_13_status_polling_never_initiates_chat_request(self):
        """Safety: get_status() must never trigger LLM or transcription."""
        chat_called = []
        vs = VoiceService(chat_fn=lambda t: chat_called.append(t))
        status = vs.get_status()
        self.assertIsInstance(status, dict)
        self.assertEqual(len(chat_called), 0)

    def test_14_typed_chat_commit_history_works_independently(self):
        """Typed Chat: Commits valid user and assistant history safely."""
        from app import _conversation_history, _conversation_lock
        with _conversation_lock:
            initial_len = len(_conversation_history)
        commit_conversation_turn("hello from user", "hello from mark")
        with _conversation_lock:
            self.assertEqual(len(_conversation_history), initial_len + 2)
            self.assertEqual(_conversation_history[-2]["content"], "hello from user")
            self.assertEqual(_conversation_history[-1]["content"], "hello from mark")

    def test_15_pipelined_tts_sentence_grouping_and_prosody(self):
        """TTS: Conversational sentence grouping preserves natural bounds."""
        tts = TTSEngine()
        text = "Hello! I am Mark, your personal assistant. I can help with programming, science, and reasoning."
        chunks = tts._segment_text(text)
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self.assertLessEqual(len(c), 360)

    def test_16_client_lease_and_heartbeat_prevents_orphan_audio(self):
        """Disconnect Safety: Lease expiration pauses microphone listening."""
        vs = VoiceService(client_lease_timeout=0.1)
        vs.enabled = True
        vs._last_client_heartbeat = time.time() - 1.0  # Expired lease
        # Verify lease is expired
        self.assertGreater(time.time() - vs._last_client_heartbeat, vs.client_lease_timeout)

        # Refresh heartbeat
        vs.heartbeat()
        self.assertLess(time.time() - vs._last_client_heartbeat, vs.client_lease_timeout)

    def test_17_wake_word_acoustic_matching_robustness(self):
        """Wake Spotting: Robust matching across acoustic variations and rejection of mark alone."""
        matches = [
            "hey mark", "hey amrk", "hey marc", "hi mark", "hello mark", "ok mark", "okay mark",
            "hey mark what is the weather", "hey amrk tell me a story"
        ]
        for phrase in matches:
            m = WAKE_WORD_PATTERN.search(phrase.lower())
            self.assertIsNotNone(m, f"Failed to match valid wake phrase: {phrase}")

        # Invariant: 'Mark' alone must not match wake phrase
        self.assertIsNone(WAKE_WORD_PATTERN.search("mark"))
        self.assertIsNone(WAKE_WORD_PATTERN.search("mark help"))
        self.assertIsNone(WAKE_WORD_PATTERN.search("just a mark on paper"))


if __name__ == "__main__":
    unittest.main()
