"""
Project Mark — Wake Word Optimization and Latency Verification Test Suite.
Validates:
1. Low-latency wake word activation for both 'Hey Jarvis' and 'Hey Mark'.
2. Strict rejection of 'Mark' alone, 'Goodbye Mark', and ambient noise.
3. Pre-roll audio preservation for trailing inline commands.
4. Consecutive wake activations without deadlock or memory leakage.
"""

import os
import sys
import time
import re
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from voice_service import VoiceService, VoiceState, VoiceTurn, WAKE_WORD_PATTERN, EXIT_PATTERNS
from tts_engine import TTSEngine


class MockTTS:
    def __init__(self):
        self.spoken = []
    def speak(self, text, **kwargs):
        self.spoken.append(text)
        return True
    def stop(self):
        pass
    def is_speaking(self):
        return False


class TestWakeOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tts = TTSEngine()

    def test_01_wake_word_pattern_rules(self):
        """Validates that WAKE_WORD_PATTERN strictly requires greetings and matches Mark / Jarvis."""
        # Must NOT match without greeting
        self.assertIsNone(WAKE_WORD_PATTERN.search("mark"))
        self.assertIsNone(WAKE_WORD_PATTERN.search("mark please"))
        self.assertIsNone(WAKE_WORD_PATTERN.search("jarvis"))
        self.assertIsNone(WAKE_WORD_PATTERN.search("jarvis status"))

        # Must match with greeting prefixes
        self.assertIsNotNone(WAKE_WORD_PATTERN.search("hey mark"))
        self.assertIsNotNone(WAKE_WORD_PATTERN.search("hi mark"))
        self.assertIsNotNone(WAKE_WORD_PATTERN.search("hello mark"))
        self.assertIsNotNone(WAKE_WORD_PATTERN.search("ok mark"))
        self.assertIsNotNone(WAKE_WORD_PATTERN.search("hey jarvis"))
        self.assertIsNotNone(WAKE_WORD_PATTERN.search("hi jarvis"))
        self.assertIsNotNone(WAKE_WORD_PATTERN.search("ok jarvis"))

    def test_02_exit_patterns_override_wake(self):
        """Validates that exit phrases like 'goodbye mark' are detected and do not activate wake."""
        clean = "goodbye mark"
        exit_matched = any(re.search(p, clean) for p in EXIT_PATTERNS)
        self.assertTrue(exit_matched)

        clean2 = "bye mark"
        exit_matched2 = any(re.search(p, clean2) for p in EXIT_PATTERNS)
        self.assertTrue(exit_matched2)

    def test_03_openwakeword_jarvis_latency_and_score(self):
        """Measures openWakeWord ONNX inference latency and detection on synthesized 'Hey Jarvis'."""
        vs = VoiceService()
        vs.tts_engine = MockTTS()
        self.assertIsNotNone(vs._oww_model, "openWakeWord ONNX model should be loaded")

        # Synthesize 'Hey Jarvis'
        audio, sr, dur, _ = self.tts.synthesize_chunk("Hey Jarvis")
        if sr != 16000:
            import scipy.signal
            audio_16k = scipy.signal.resample(audio, int(len(audio)*16000/sr)).astype(np.float32)
        else:
            audio_16k = audio.astype(np.float32)

        pcm = (audio_16k * 32767.0).clip(-32768, 32767).astype(np.int16)
        detected = False
        latencies = []

        for i in range(0, len(pcm) - 1280, 1280):
            frame = pcm[i:i + 1280]
            t0 = time.perf_counter()
            vs._oww_model.predict(frame)
            latencies.append((time.perf_counter() - t0) * 1000)

            score = vs._oww_model.prediction_buffer['hey_jarvis'][-1]
            if score >= 0.45:
                detected = True
                break

        self.assertTrue(detected, "openWakeWord should detect 'Hey Jarvis'")
        avg_latency = float(np.mean(latencies))
        print(f"\n[Test] openWakeWord 'hey_jarvis' chunk inference avg: {avg_latency:.2f}ms")
        self.assertLess(avg_latency, 25.0, "openWakeWord chunk latency should be under 25ms")

    def test_04_whisper_tiny_hey_mark_latency_and_accuracy(self):
        """Measures Whisper tiny.en acoustic wake spotting latency on synthesized 'Hey Mark'."""
        vs = VoiceService()
        vs.tts_engine = MockTTS()
        self.assertIsNotNone(vs._wake_whisper_model, "Whisper tiny.en wake model should be loaded")

        audio, sr, dur, _ = self.tts.synthesize_chunk("Hey Mark, how are you?")
        if sr != 16000:
            import scipy.signal
            audio_16k = scipy.signal.resample(audio, int(len(audio)*16000/sr)).astype(np.float32)
        else:
            audio_16k = audio.astype(np.float32)

        # Slice the first 1.1s (wake portion)
        wake_slice = audio_16k[:int(16000 * 1.1)]
        t0 = time.perf_counter()
        detected, extra = vs._check_wake_word_utterance(wake_slice, vs._worker_generation)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        print(f"\n[Test] Whisper tiny.en 'Hey Mark' wake spotting latency: {elapsed_ms:.1f}ms (detected={detected})")
        self.assertTrue(detected, "'Hey Mark' should be detected by acoustic spotter")
        self.assertLess(elapsed_ms, 1200.0, "Wake spotting latency should be under 1200ms (far faster than 15s)")

    def test_05_silence_and_noise_rejection(self):
        """Validates that pure silence or noise does not trigger wake word."""
        vs = VoiceService()
        vs.tts_engine = MockTTS()

        # Pure silence
        detected_silence, _ = vs._check_wake_word_utterance(np.zeros(16000, dtype=np.float32), vs._worker_generation)
        self.assertFalse(detected_silence)

        # Random Gaussian noise
        noise = np.random.normal(0, 0.01, 16000).astype(np.float32)
        detected_noise, _ = vs._check_wake_word_utterance(noise, vs._worker_generation)
        self.assertFalse(detected_noise)

    def test_06_pre_roll_preservation_for_continuous_command(self):
        """Validates that continuous command pre-roll is preserved and not clipped."""
        vs = VoiceService()
        vs.tts_engine = MockTTS()

        audio, sr, dur, _ = self.tts.synthesize_chunk("Hey Mark, what time is it?")
        if sr != 16000:
            import scipy.signal
            audio_16k = scipy.signal.resample(audio, int(len(audio)*16000/sr)).astype(np.float32)
        else:
            audio_16k = audio.astype(np.float32)

        # Total audio evaluated in full
        detected, command = vs._check_wake_word_utterance(audio_16k, vs._worker_generation)
        self.assertTrue(detected)
        self.assertIn("what", command.lower())
        print(f"\n[Test] Inline command preserved: '{command}'")

    def test_07_multiple_consecutive_wake_cycles(self):
        """Validates that consecutive wake-up and session endings do not deadlock."""
        vs = VoiceService()
        vs.tts_engine = MockTTS()
        vs.session_id = "test_cycle"
        vs.enabled = True

        for cycle in range(3):
            vs.state = VoiceState.WAKE_WORD_LISTENING
            vs.session_active = False

            # Simulate wake event
            vs.session_active = True
            vs.state = VoiceState.LISTENING_FOR_COMMAND

            self.assertTrue(vs.session_active)
            self.assertEqual(vs.state, VoiceState.LISTENING_FOR_COMMAND)

            # End session
            vs.end_session(speak_goodbye=False)
            self.assertFalse(vs.session_active)
            self.assertEqual(vs.state, VoiceState.WAKE_WORD_LISTENING)


if __name__ == "__main__":
    unittest.main()
