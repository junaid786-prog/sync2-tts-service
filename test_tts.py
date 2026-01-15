"""
Simple test script for Kokoro TTS Service
Tests the TTS without needing Asterisk

Usage:
  python test_tts.py "Hello, how can I help you today?"
"""

import asyncio
import sys
import wave
import requests

# Fix encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

# Configuration
TTS_URL = "http://localhost:8765"


def test_health():
    """Test if TTS service is running"""
    try:
        response = requests.get(f"{TTS_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] TTS Service is running")
            print(f"     Model: {data.get('model')}")
            print(f"     Device: {data.get('device')}")
            print(f"     GPU: {data.get('gpu_name', 'N/A')}")
            return True
        else:
            print(f"[FAIL] Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"[FAIL] Cannot connect to TTS service at {TTS_URL}")
        print(f"       Make sure the service is running: docker-compose up")
        return False


def test_voices():
    """List available voices"""
    try:
        response = requests.get(f"{TTS_URL}/voices", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"\n[VOICES] Available voices:")
            for voice_id, description in data.get('voices', {}).items():
                print(f"   - {voice_id}: {description}")
            print(f"   Default: {data.get('default')}")
            return True
    except Exception as e:
        print(f"[FAIL] Failed to get voices: {e}")
        return False


def test_synthesize(text, output_file="test_output.wav"):
    """Test TTS synthesis"""
    print(f"\n[SYNTH] Synthesizing: \"{text}\"")

    try:
        response = requests.post(
            f"{TTS_URL}/tts/synthesize",
            json={
                "text": text,
                "voice": "af_sarah",
                "output_format": "pcm16"
            },
            timeout=30
        )

        if response.status_code == 200:
            # Get audio data
            audio_data = response.content
            duration_ms = response.headers.get('X-Duration-Ms', '?')

            # Save as WAV file
            with wave.open(output_file, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(8000)  # 8kHz
                wav_file.writeframes(audio_data)

            print(f"[OK] Synthesis complete in {duration_ms}ms")
            print(f"     Output saved to: {output_file}")
            print(f"     Audio size: {len(audio_data)} bytes")
            print(f"\n[PLAY] Open the file: {output_file}")
            return True
        else:
            print(f"[FAIL] Synthesis failed: {response.status_code}")
            print(f"       {response.text}")
            return False

    except Exception as e:
        print(f"[FAIL] Synthesis error: {e}")
        return False


def main():
    print("=" * 50)
    print("  Kokoro TTS Test Script")
    print("=" * 50)

    # Get text from command line or use default
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = "Hello! I am Kokoro, an open source text to speech model. How can I help you today?"

    # Run tests
    if not test_health():
        print("\n[WARN] Start the TTS service first:")
        print("       cd sync2-tts-service")
        print("       docker-compose up")
        return

    test_voices()
    test_synthesize(text)


if __name__ == "__main__":
    main()
