"""
FastAPI + WebSocket Server for Piper TTS Service
Drop-in replacement for Kokoro TTS with same interface
"""
import asyncio
import logging
import time
import sys
import os
import io
import subprocess
import wave
from typing import Optional
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
SAMPLE_RATE = 22050  # Piper native sample rate
OUTPUT_SAMPLE_RATE = 8000  # Telephony output

# Piper model/voice configuration
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "/app/models/en_US-amy-medium.onnx")
PIPER_CONFIG_PATH = os.getenv("PIPER_CONFIG_PATH", "/app/models/en_US-amy-medium.onnx.json")
PIPER_BINARY = os.getenv("PIPER_BINARY", "/app/piper/piper")

# Available voices (map to model files)
VOICES = {
    "amy": {
        "model": "en_US-amy-medium.onnx",
        "description": "US English Female (Amy) - Medium quality"
    },
    "lessac": {
        "model": "en_US-lessac-medium.onnx",
        "description": "US English Male (Lessac) - Medium quality"
    },
    "libritts": {
        "model": "en_US-libritts_r-medium.onnx",
        "description": "US English Multi-speaker - Medium quality"
    }
}

# Default voice
DEFAULT_VOICE = "amy"
ALL_VOICES = list(VOICES.keys())


def check_piper_installation():
    """Verify Piper is installed and models exist"""
    if not os.path.exists(PIPER_BINARY):
        raise RuntimeError(f"Piper binary not found at {PIPER_BINARY}")

    if not os.path.exists(PIPER_MODEL_PATH):
        raise RuntimeError(f"Piper model not found at {PIPER_MODEL_PATH}")

    if not os.path.exists(PIPER_CONFIG_PATH):
        raise RuntimeError(f"Piper config not found at {PIPER_CONFIG_PATH}")

    logger.info(f"Piper binary: {PIPER_BINARY}")
    logger.info(f"Piper model: {PIPER_MODEL_PATH}")
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - verify Piper on startup"""
    logger.info("=" * 50)
    logger.info("  Sync2 Piper TTS Service - Starting")
    logger.info("=" * 50)

    try:
        check_piper_installation()
        logger.info("Piper TTS ready")
        logger.info(f"Available voices: {len(ALL_VOICES)}")
    except Exception as e:
        logger.error(f"Failed to initialize Piper: {e}")
        raise

    yield

    logger.info("Shutting down Piper TTS Service...")


# Create FastAPI app
app = FastAPI(
    title="Sync2 Piper TTS Service",
    description="Fast open-source TTS service using Piper for Sync2.ai Voice AI Platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request models
class TTSRequest(BaseModel):
    text: str
    voice: str = "amy"
    speed: float = 1.0
    output_format: str = "wav"  # "wav", "pcm16", "ulaw"


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate"""
    if orig_sr == target_sr:
        return audio

    duration = len(audio) / orig_sr
    target_length = int(duration * target_sr)
    indices = np.linspace(0, len(audio) - 1, target_length)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def audio_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float audio to 16-bit PCM bytes"""
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    return pcm.tobytes()


def audio_to_ulaw(audio: np.ndarray) -> bytes:
    """Convert float audio to u-law encoded bytes"""
    import audioop
    pcm = audio_to_pcm16(audio)
    return audioop.lin2ulaw(pcm, 2)


def synthesize_with_piper(text: str, voice: str = "amy", speed: float = 1.0) -> np.ndarray:
    """
    Synthesize speech using Piper CLI
    Returns numpy array of audio samples
    """
    # Get model path for voice
    voice_info = VOICES.get(voice, VOICES[DEFAULT_VOICE])
    model_path = f"/app/models/{voice_info['model']}"
    config_path = f"{model_path}.json"

    # Use default if voice model doesn't exist
    if not os.path.exists(model_path):
        model_path = PIPER_MODEL_PATH
        config_path = PIPER_CONFIG_PATH

    # Build Piper command
    cmd = [
        PIPER_BINARY,
        "--model", model_path,
        "--config", config_path,
        "--output-raw",
        "--length_scale", str(1.0 / speed)  # Piper uses length_scale (inverse of speed)
    ]

    try:
        # Run Piper
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Send text and get audio
        stdout, stderr = process.communicate(input=text.encode('utf-8'), timeout=30)

        if process.returncode != 0:
            logger.error(f"Piper error: {stderr.decode()}")
            raise RuntimeError(f"Piper failed: {stderr.decode()}")

        # Convert raw PCM to numpy array (Piper outputs 16-bit PCM)
        audio = np.frombuffer(stdout, dtype=np.int16).astype(np.float32) / 32767.0

        return audio

    except subprocess.TimeoutExpired:
        process.kill()
        raise RuntimeError("Piper synthesis timeout")
    except Exception as e:
        logger.error(f"Piper synthesis error: {e}")
        raise


# REST Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    piper_ready = os.path.exists(PIPER_BINARY) and os.path.exists(PIPER_MODEL_PATH)
    return {
        "status": "ok" if piper_ready else "error",
        "model": "Piper",
        "model_loaded": piper_ready,
        "device": "cpu",
        "gpu_available": False,
        "gpu_name": None,
        "ready": piper_ready,
        "sample_rate": SAMPLE_RATE,
        "output_sample_rate": OUTPUT_SAMPLE_RATE,
        "total_voices": len(ALL_VOICES)
    }


@app.get("/voices")
async def list_voices():
    """List available voices - flat format for ARI Bridge"""
    flat_voices = {v: v for v in ALL_VOICES}
    return {
        "voices": flat_voices,
        "default": DEFAULT_VOICE
    }


@app.get("/api/voices")
async def list_voices_detailed():
    """List available voices with details"""
    return {
        "voices": VOICES,
        "all_voices": ALL_VOICES,
        "default": DEFAULT_VOICE
    }


@app.post("/tts/synthesize")
async def synthesize(request: TTSRequest):
    """
    Synthesize speech from text
    Returns audio in requested format (wav, pcm16, or ulaw)
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    start = time.time()

    try:
        # Generate audio with Piper
        full_audio = synthesize_with_piper(request.text, request.voice, request.speed)
        gen_time = time.time() - start

        logger.info(f"[TTS] Generated {len(request.text)} chars in {gen_time*1000:.0f}ms")

        # Process based on output format
        if request.output_format == "wav":
            # Return full WAV file at original sample rate
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(audio_to_pcm16(full_audio))
            buffer.seek(0)

            return StreamingResponse(
                buffer,
                media_type="audio/wav",
                headers={
                    "X-Duration-Ms": str(gen_time * 1000),
                    "X-Sample-Rate": str(SAMPLE_RATE),
                    "X-Format": "wav"
                }
            )

        else:
            # Resample to 8kHz for telephony
            resampled = resample_audio(full_audio, SAMPLE_RATE, OUTPUT_SAMPLE_RATE)

            if request.output_format == "ulaw":
                audio_bytes = audio_to_ulaw(resampled)
                content_type = "audio/basic"
            else:  # pcm16
                audio_bytes = audio_to_pcm16(resampled)
                content_type = "audio/L16"

            return Response(
                content=audio_bytes,
                media_type=content_type,
                headers={
                    "X-Duration-Ms": str(gen_time * 1000),
                    "X-Audio-Length": str(len(audio_bytes)),
                    "X-Sample-Rate": str(OUTPUT_SAMPLE_RATE),
                    "X-Format": request.output_format
                }
            )

    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tts/generate")
async def generate_speech(request: TTSRequest):
    """Generate speech - returns WAV file"""
    request.output_format = "wav"
    return await synthesize(request)


# WebSocket Endpoint for Streaming
@app.websocket("/tts/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming TTS

    Protocol:
    1. Client connects
    2. Client sends JSON: {"text": "Hello", "voice": "amy", "format": "ulaw"}
    3. Server streams binary audio chunks
    4. Server sends JSON: {"done": true, "duration_ms": 123}
    """
    await websocket.accept()
    logger.info("[WS] Client connected")

    try:
        while True:
            # Receive synthesis request
            data = await websocket.receive_json()

            text = data.get("text", "")
            voice = data.get("voice", DEFAULT_VOICE)
            output_format = data.get("format", "ulaw")
            speed = data.get("speed", 1.0)

            if not text:
                await websocket.send_json({"error": "Text is required"})
                continue

            logger.info(f"[WS] Streaming: '{text[:50]}...' voice={voice}")

            start_time = time.time()
            total_bytes = 0

            try:
                # Generate audio with Piper
                audio = synthesize_with_piper(text, voice, speed)

                # Resample to 8kHz for telephony
                resampled = resample_audio(audio, SAMPLE_RATE, OUTPUT_SAMPLE_RATE)

                # Convert to requested format
                if output_format == "ulaw":
                    audio_bytes = audio_to_ulaw(resampled)
                else:
                    audio_bytes = audio_to_pcm16(resampled)

                # Stream in chunks (4KB chunks for smooth playback)
                chunk_size = 4000
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i:i + chunk_size]
                    await websocket.send_bytes(chunk)
                    total_bytes += len(chunk)
                    await asyncio.sleep(0.001)

                duration_ms = (time.time() - start_time) * 1000

                # Send completion message
                await websocket.send_json({
                    "done": True,
                    "duration_ms": duration_ms,
                    "total_bytes": total_bytes
                })

                logger.info(f"[WS] Streamed {total_bytes} bytes in {duration_ms:.0f}ms")

            except Exception as e:
                logger.error(f"[WS] Streaming error: {e}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        logger.info("[WS] Client disconnected")
    except Exception as e:
        logger.error(f"[WS] Error: {e}")


if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("  Sync2 Piper TTS Service")
    print("=" * 50)
    print("  Endpoints:")
    print("    GET  http://localhost:8765/health")
    print("    GET  http://localhost:8765/voices")
    print("    POST http://localhost:8765/tts/synthesize")
    print("    WS   ws://localhost:8765/tts/stream")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8765)
