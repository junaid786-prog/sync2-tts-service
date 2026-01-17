"""
Test WebSocket streaming - simulates exactly what ARI Bridge does in production

This shows the real-time streaming flow:
1. Connect to TTS service via WebSocket
2. Send text to synthesize
3. Receive audio chunks as they're generated
4. Save to file for playback

Usage:
  python test_websocket.py "Hello, how can I help you today?"
"""

import asyncio
import sys
import json
import wave

# Fix encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

async def test_websocket_stream(text, output_file="test_stream.wav"):
    """Test WebSocket streaming - exactly like ARI Bridge does"""

    try:
        import websockets
    except ImportError:
        print("[FAIL] websockets not installed. Run: pip install websockets")
        return

    ws_url = "ws://localhost:8765/tts/stream"

    print("=" * 60)
    print("  WebSocket TTS Stream Test (Production Flow Simulation)")
    print("=" * 60)
    print(f"\n[1] Connecting to: {ws_url}")

    try:
        async with websockets.connect(ws_url) as ws:
            print("[OK] Connected to TTS service")

            # This is exactly what ARI Bridge sends
            request = {
                "text": text,
                "voice": "af_sarah",
                "format": "pcm16",  # or "ulaw" for Asterisk
                "speed": 1.0
            }

            print(f"\n[2] Sending TTS request:")
            print(f"    Text: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
            print(f"    Voice: {request['voice']}")
            print(f"    Format: {request['format']}")

            await ws.send(json.dumps(request))

            print(f"\n[3] Receiving audio chunks (streaming)...")

            audio_chunks = []
            chunk_count = 0

            while True:
                message = await ws.recv()

                if isinstance(message, bytes):
                    # Binary audio chunk
                    chunk_count += 1
                    audio_chunks.append(message)
                    # Show progress
                    total_bytes = sum(len(c) for c in audio_chunks)
                    print(f"    Chunk {chunk_count}: {len(message)} bytes (total: {total_bytes})", end='\r')
                else:
                    # JSON message (completion or error)
                    data = json.loads(message)
                    print()  # New line after progress

                    if data.get("done"):
                        print(f"\n[OK] Stream complete!")
                        print(f"    Total chunks: {chunk_count}")
                        print(f"    Total bytes: {data.get('total_bytes', 'N/A')}")
                        print(f"    Duration: {data.get('duration_ms', 'N/A'):.0f}ms")
                        break
                    elif data.get("error"):
                        print(f"\n[FAIL] Error: {data['error']}")
                        return

            # Save audio to WAV file
            if audio_chunks:
                print(f"\n[4] Saving to: {output_file}")

                all_audio = b''.join(audio_chunks)

                with wave.open(output_file, 'wb') as wav:
                    wav.setnchannels(1)       # Mono
                    wav.setsampwidth(2)       # 16-bit
                    wav.setframerate(8000)    # 8kHz (Asterisk format)
                    wav.writeframes(all_audio)

                print(f"[OK] Saved {len(all_audio)} bytes")
                print(f"\n[PLAY] Open the file to hear: {output_file}")

            print("\n" + "=" * 60)
            print("  This is exactly how audio flows in production!")
            print("  ARI Bridge receives these chunks and plays to caller")
            print("=" * 60)

    except ConnectionRefusedError:
        print(f"[FAIL] Cannot connect to {ws_url}")
        print("       Make sure the TTS server is running:")
        print("       > .\\run_server.bat")
    except Exception as e:
        print(f"[FAIL] Error: {e}")


def main():
    # Get text from command line or use default
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = "Hello! Thank you for calling. How can I help you with your appointment today?"

    asyncio.run(test_websocket_stream(text))


if __name__ == "__main__":
    main()
