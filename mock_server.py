"""
Mock TTS Server for Testing
Runs without GPU - returns silent audio for testing the pipeline

Usage:
  python mock_server.py
"""

import json
import asyncio
import sys
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
import uvicorn

# Fix encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI(title="Mock TTS Server")

# Generate silence (for testing pipeline without actual TTS)
def generate_silent_audio(duration_ms=1000, sample_rate=8000):
    """Generate silent audio for testing"""
    num_samples = int(sample_rate * duration_ms / 1000)
    # Generate very quiet noise instead of pure silence (sounds more natural)
    audio = np.random.randn(num_samples) * 0.001
    return (audio * 32767).astype(np.int16).tobytes()


def generate_tone_audio(text, sample_rate=8000):
    """Generate a simple tone based on text length (for testing)"""
    # Duration based on text length (roughly 100ms per word)
    words = len(text.split())
    duration_ms = max(500, words * 150)
    num_samples = int(sample_rate * duration_ms / 1000)

    # Generate a simple sine wave tone
    t = np.linspace(0, duration_ms / 1000, num_samples)
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t) * 0.3

    # Add envelope (fade in/out)
    fade_samples = int(sample_rate * 0.05)  # 50ms fade
    audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
    audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)

    return (audio * 32767).astype(np.int16).tobytes()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "mock-tts",
        "device": "cpu",
        "gpu_available": False,
        "gpu_name": None,
        "ready": True
    }


@app.get("/voices")
async def voices():
    return {
        "voices": {
            "af_sarah": "Mock Female - Sarah",
            "am_adam": "Mock Male - Adam"
        },
        "default": "af_sarah"
    }


@app.post("/tts/synthesize")
async def synthesize(request: dict):
    text = request.get("text", "")
    output_format = request.get("output_format", "pcm16")

    print(f"🎙️ [MOCK] Synthesizing: '{text[:50]}...'")

    # Generate test audio
    audio_data = generate_tone_audio(text)

    content_type = "audio/basic" if output_format == "ulaw" else "audio/L16"

    return Response(
        content=audio_data,
        media_type=content_type,
        headers={
            "X-Duration-Ms": "100",
            "X-Audio-Length": str(len(audio_data)),
            "X-Sample-Rate": "8000",
            "X-Format": output_format
        }
    )


@app.websocket("/tts/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    print("🔌 [MOCK] WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_json()
            text = data.get("text", "")
            output_format = data.get("format", "pcm16")

            print(f"🎙️ [MOCK] Streaming: '{text[:50]}...'")

            # Generate audio
            audio_data = generate_tone_audio(text)

            # Send in chunks (simulate streaming)
            chunk_size = 640  # 40ms at 8kHz, 16-bit
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                await websocket.send_bytes(chunk)
                await asyncio.sleep(0.01)  # Small delay between chunks

            # Send completion
            await websocket.send_json({
                "done": True,
                "duration_ms": 100,
                "total_bytes": len(audio_data)
            })

    except WebSocketDisconnect:
        print("🔌 [MOCK] WebSocket client disconnected")


if __name__ == "__main__":
    print("=" * 50)
    print("  Mock TTS Server (for testing without GPU)")
    print("=" * 50)
    print("  Endpoints:")
    print("    GET  http://localhost:8765/health")
    print("    GET  http://localhost:8765/voices")
    print("    POST http://localhost:8765/tts/synthesize")
    print("    WS   ws://localhost:8765/tts/stream")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8765)
