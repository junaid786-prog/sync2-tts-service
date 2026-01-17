"""
Test Full Production Flow (without Asterisk)

Simulates the complete voice AI flow:
1. User says something (simulated STT input)
2. Claude AI generates response (simulated)
3. Kokoro TTS converts to speech
4. Audio plays to user

This is exactly what happens in a real call!
"""

import asyncio
import sys
import json
import wave
import os

sys.stdout.reconfigure(encoding='utf-8')

# Simulated conversation
SAMPLE_CONVERSATIONS = [
    {
        "user": "Hi, I'd like to book an appointment",
        "ai_response": "Hello! I'd be happy to help you book an appointment. What date works best for you?"
    },
    {
        "user": "How about next Tuesday?",
        "ai_response": "Next Tuesday sounds great. I have openings at 9 AM, 11 AM, and 2 PM. Which time would you prefer?"
    },
    {
        "user": "2 PM works for me",
        "ai_response": "Perfect! I've scheduled your appointment for next Tuesday at 2 PM. You'll receive a confirmation text shortly. Is there anything else I can help you with?"
    },
    {
        "user": "No that's all, thank you",
        "ai_response": "You're welcome! Have a great day. Goodbye!"
    }
]


async def synthesize_response(text, output_file):
    """Synthesize AI response using Kokoro TTS via WebSocket"""
    import websockets

    ws_url = "ws://localhost:8765/tts/stream"

    async with websockets.connect(ws_url) as ws:
        request = {
            "text": text,
            "voice": "af_sarah",
            "format": "pcm16",
            "speed": 1.0
        }

        await ws.send(json.dumps(request))

        audio_chunks = []
        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                audio_chunks.append(message)
            else:
                data = json.loads(message)
                if data.get("done"):
                    break

        # Save audio
        all_audio = b''.join(audio_chunks)
        with wave.open(output_file, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(8000)
            wav.writeframes(all_audio)

        return len(all_audio)


async def run_conversation():
    """Run a simulated phone conversation"""

    print("=" * 70)
    print("  SIMULATED PHONE CALL - Full Production Flow Test")
    print("=" * 70)
    print("\n  This simulates exactly what happens during a real call:")
    print("  1. Caller speaks -> STT -> Text")
    print("  2. Text -> Claude AI -> Response")
    print("  3. Response -> Kokoro TTS -> Audio")
    print("  4. Audio -> Asterisk -> Caller hears it")
    print("\n" + "=" * 70)

    # Create output directory
    os.makedirs("call_audio", exist_ok=True)

    print("\n[CALL STARTED]\n")

    for i, turn in enumerate(SAMPLE_CONVERSATIONS, 1):
        # Simulate user speaking (STT would convert this)
        print(f"  CALLER: \"{turn['user']}\"")
        print(f"     [STT converts speech to text]")

        # Simulate Claude AI response
        print(f"\n  AI THINKING...")
        print(f"     [Claude generates response]")

        ai_response = turn['ai_response']
        print(f"\n  AI: \"{ai_response}\"")

        # Generate TTS audio
        output_file = f"call_audio/turn_{i}.wav"
        print(f"     [Kokoro TTS generating audio...]", end=" ")

        try:
            audio_size = await synthesize_response(ai_response, output_file)
            print(f"OK ({audio_size} bytes)")
            print(f"     [Audio saved: {output_file}]")
        except Exception as e:
            print(f"FAILED: {e}")

        print(f"\n  {'─' * 60}\n")

        # Small delay between turns
        await asyncio.sleep(0.5)

    print("[CALL ENDED]\n")
    print("=" * 70)
    print("  Audio files saved in: call_audio/")
    print("  Play them in order to hear the AI side of the conversation!")
    print("=" * 70)

    # List generated files
    print("\n  Generated audio files:")
    for i in range(1, len(SAMPLE_CONVERSATIONS) + 1):
        filepath = f"call_audio/turn_{i}.wav"
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"    {filepath} ({size} bytes)")


def main():
    try:
        import websockets
    except ImportError:
        print("Install websockets: pip install websockets")
        return

    asyncio.run(run_conversation())


if __name__ == "__main__":
    main()
