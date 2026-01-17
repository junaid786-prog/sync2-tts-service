"""
Test script for Kokoro TTS Service

Usage:
  python test_tts.py "Hello, how can I help you today?"
"""

import sys
import requests

# Fix encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

# Configuration
TTS_URL = "http://localhost:8765"


def test_health():
    """Test if TTS service is running"""
    try:
        response = requests.get(f"{TTS_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] TTS Service is running")
            print(f"     Model: {data.get('model')}")
            print(f"     Device: {data.get('device')}")
            print(f"     Ready: {data.get('ready')}")
            print(f"     Voices: {data.get('total_voices')}")
            return True
        else:
            print(f"[FAIL] Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"[FAIL] Cannot connect to TTS service at {TTS_URL}")
        print(f"       Make sure the service is running")
        return False


def test_voices():
    """List available voices"""
    try:
        response = requests.get(f"{TTS_URL}/voices", timeout=5)
        if response.status_code == 200:
            data = response.json()
            voices = data.get('voices', {})
            print(f"\n[VOICES] Available: {len(voices)} voices")
            print(f"   Default: {data.get('default')}")
            # Show first 5 voices as sample
            sample = list(voices.keys())[:5]
            print(f"   Sample: {', '.join(sample)}...")
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
                "output_format": "wav"
            },
            timeout=60  # Longer timeout for synthesis
        )

        if response.status_code == 200:
            audio_data = response.content
            duration_ms = response.headers.get('X-Duration-Ms', 'N/A')
            sample_rate = response.headers.get('X-Sample-Rate', 'N/A')

            # Save as WAV file
            with open(output_file, 'wb') as f:
                f.write(audio_data)

            print(f"[OK] Synthesis complete in {duration_ms}ms")
            print(f"     Output saved to: {output_file}")
            print(f"     Audio size: {len(audio_data)} bytes")
            print(f"     Sample rate: {sample_rate}Hz")
            print(f"\n[PLAY] Open the file: {output_file}")
            return True
        else:
            print(f"[FAIL] Synthesis failed: {response.status_code}")
            print(f"       {response.text}")
            return False

    except requests.exceptions.Timeout:
        print(f"[FAIL] Synthesis timed out (model may still be loading)")
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
        print("       python -m src.server")
        return

    test_voices()
    test_synthesize(text)


if __name__ == "__main__":
    main()
