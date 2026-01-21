"""
FastAPI + WebSocket Server for Chatterbox-Turbo TTS Service
Sub-200ms latency voice synthesis with GPU acceleration
Drop-in replacement for Kokoro/XTTS with same interface
"""
import asyncio
import logging
import time
import sys
import io
import wave
import torch
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
tts_model = None
SAMPLE_RATE = 24000  # Chatterbox native sample rate
OUTPUT_SAMPLE_RATE = 8000  # Telephony output

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Available voices (reference audio files for voice cloning)
VOICES = {
    "default": {
        "description": "Chatterbox Default Voice",
        "language": "en"
    },
    "sarah": {
        "description": "US English Female - Warm and friendly",
        "language": "en"
    },
    "emma": {
        "description": "US English Female - Professional",
        "language": "en"
    },
    "james": {
        "description": "US English Male - Deep and calm",
        "language": "en"
    }
}

DEFAULT_VOICE = "default"
ALL_VOICES = list(VOICES.keys())


def load_chatterbox_model():
    """Load Chatterbox-Turbo model"""
    global tts_model

    logger.info(f"Loading Chatterbox-Turbo model on {DEVICE}...")
    start = time.time()

    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        # Load Chatterbox-Turbo model
        tts_model = ChatterboxTurboTTS.from_pretrained(device=DEVICE)

        load_time = time.time() - start
        logger.info(f"Chatterbox-Turbo model loaded in {load_time:.1f}s on {DEVICE}")

        return True
    except Exception as e:
        logger.error(f"Failed to load Chatterbox-Turbo model: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - load model on startup"""
    logger.info("=" * 50)
    logger.info("  Sync2 Chatterbox-Turbo TTS Service - Starting")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 50)

    try:
        load_chatterbox_model()
        logger.info("Chatterbox-Turbo TTS ready")
        logger.info(f"Available voices: {len(ALL_VOICES)}")
    except Exception as e:
        logger.error(f"Failed to initialize Chatterbox-Turbo: {e}")
        raise

    yield

    logger.info("Shutting down Chatterbox-Turbo TTS Service...")


# Create FastAPI app
app = FastAPI(
    title="Sync2 Chatterbox-Turbo TTS Service",
    description="Sub-200ms latency TTS service using Chatterbox-Turbo for Sync2.ai Voice AI Platform",
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
    voice: str = "default"
    speed: float = 1.0
    output_format: str = "wav"  # "wav", "pcm16", "ulaw"
    language: str = "en"


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


def synthesize_with_chatterbox(text: str, voice: str = "default", speed: float = 1.0) -> np.ndarray:
    """
    Synthesize speech using Chatterbox-Turbo
    Returns numpy array of audio samples
    """
    global tts_model

    if tts_model is None:
        raise RuntimeError("Chatterbox-Turbo model not loaded")

    try:
        # Get speaker reference wav path if custom voice
        audio_prompt_path = None
        voice_dir = Path("/app/voices")

        if voice != "default" and voice_dir.exists():
            voice_file = voice_dir / f"{voice}.wav"
            if voice_file.exists():
                audio_prompt_path = str(voice_file)

        # Generate audio with Chatterbox-Turbo
        if audio_prompt_path:
            # Use voice cloning with reference audio
            wav = tts_model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path
            )
        else:
            # Use default voice (no reference audio)
            wav = tts_model.generate(text=text)

        # Convert to numpy array
        if torch.is_tensor(wav):
            audio = wav.cpu().numpy()
        else:
            audio = np.array(wav, dtype=np.float32)

        # Ensure 1D array
        if audio.ndim > 1:
            audio = audio.squeeze()

        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95

        return audio.astype(np.float32)

    except Exception as e:
        logger.error(f"Chatterbox-Turbo synthesis error: {e}")
        raise


# REST Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

    return {
        "status": "ok" if tts_model is not None else "error",
        "model": "Chatterbox-Turbo",
        "model_loaded": tts_model is not None,
        "device": DEVICE,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "ready": tts_model is not None,
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
        # Generate audio with Chatterbox-Turbo
        full_audio = synthesize_with_chatterbox(
            request.text,
            request.voice,
            request.speed
        )
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
    2. Client sends JSON: {"text": "Hello", "voice": "default", "format": "ulaw"}
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
                # Generate audio with Chatterbox-Turbo
                audio = synthesize_with_chatterbox(text, voice, speed)

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
    print("  Sync2 Chatterbox-Turbo TTS Service")
    print(f"  Device: {DEVICE}")
    print(f"  GPU: {torch.cuda.is_available()}")
    print("=" * 50)
    print("  Endpoints:")
    print("    GET  http://localhost:8765/health")
    print("    GET  http://localhost:8765/voices")
    print("    POST http://localhost:8765/tts/synthesize")
    print("    WS   ws://localhost:8765/tts/stream")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8765)
