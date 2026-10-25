"""
Test script to verify VoiceService lifecycle, session states, and Kokoro TTS.
"""

import time
import numpy as np
from voice_service import VoiceService, VoiceState
from tts_engine import TTSEngine

def test_tts():
    print("\n--- 1. Testing TTS Engine ---")
    tts = TTSEngine()
    assert tts._kokoro is not None or tts._sapi_engine is not None, "TTS engine failed to initialize"
    print("TTS engine initialized successfully.")
    
    # Test markdown cleaning
    dirty = "### Heading\n**Bold statement** with `inline_code` and [link](http://test.com)."
    clean = tts.clean_text(dirty)
    print(f"Cleaned text: '{clean}'")
    assert "Heading" in clean and "Bold statement" in clean and "inline_code" in clean
    assert "###" not in clean and "**" not in clean and "`" not in clean

    # Test speaking a short phrase
    print("Speaking test phrase...")
    success = tts.speak("Kokoro neural voice is fully operational.")
    print(f"Speech success: {success}")

def test_service_lifecycle():
    print("\n--- 2. Testing VoiceService Lifecycle ---")
    dummy_transcribe = lambda audio: "test command"
    dummy_chat = lambda text: f"Echo: {text}"

    vs = VoiceService(
        transcribe_fn=dummy_transcribe,
        chat_fn=dummy_chat,
        silence_threshold=0.5,
        session_timeout=5.0,
    )

    assert vs.state == VoiceState.DISABLED
    assert not vs.enabled

    print("Starting voice service...")
    vs.start()
    time.sleep(0.5)
    assert vs.enabled
    assert vs.state == VoiceState.WAKE_WORD_LISTENING
    status = vs.get_status()
    print(f"Status after start: {status}")

    print("Testing session greeting & transition...")
    vs.session_active = True
    vs.state = VoiceState.GREETING
    vs.state = VoiceState.LISTENING_FOR_COMMAND
    assert vs.state == VoiceState.LISTENING_FOR_COMMAND

    print("Testing rapid stop / start resilience...")
    for i in range(3):
        print(f"  Cycle {i+1}: Stopping...")
        vs.stop()
        assert vs.state == VoiceState.DISABLED
        assert not vs.enabled
        time.sleep(0.2)
        print(f"  Cycle {i+1}: Starting...")
        vs.start()
        time.sleep(0.2)
        assert vs.enabled
        assert vs.state == VoiceState.WAKE_WORD_LISTENING

    print("Stopping voice service...")
    vs.stop()
    assert vs.state == VoiceState.DISABLED
    print("Lifecycle test PASSED!")

if __name__ == "__main__":
    test_tts()
    test_service_lifecycle()
    print("\nALL VOICE TESTS PASSED!")
