"""
FastAPI + WebSocket Server for Orpheus TTS Service
Human-like voice synthesis with emotional control
Drop-in replacement for Kokoro/XTTS/Chatterbox with same interface
"""
import asyncio
import logging
import time
import sys
import os
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

# Setup HuggingFace token from environment
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    os.environ["HF_TOKEN"] = HF_TOKEN
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
    except Exception:
        pass

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
SAMPLE_RATE = 24000  # Orpheus native sample rate
OUTPUT_SAMPLE_RATE = 8000  # Telephony output

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Available voices in Orpheus
VOICES = {
    "tara": {
        "description": "US English Female - Natural and warm",
        "language": "en"
    },
    "leah": {
        "description": "US English Female - Clear and professional",
        "language": "en"
    },
    "jess": {
        "description": "US English Female - Friendly",
        "language": "en"
    },
    "mia": {
        "description": "US English Female - Soft and gentle",
        "language": "en"
    },
    "zoe": {
        "description": "US English Female - Energetic",
        "language": "en"
    },
    "leo": {
        "description": "US English Male - Deep and calm",
        "language": "en"
    },
    "dan": {
        "description": "US English Male - Professional",
        "language": "en"
    },
    "zac": {
        "description": "US English Male - Friendly",
        "language": "en"
    }
}

DEFAULT_VOICE = "tara"
ALL_VOICES = list(VOICES.keys())


def load_orpheus_model():
    """Load Orpheus TTS model"""
    global tts_model

    logger.info(f"Loading Orpheus TTS model on {DEVICE}...")
    start = time.time()

    try:
        from orpheus_tts import OrpheusModel

        # Load Orpheus model (simple initialization)
        tts_model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")

        load_time = time.time() - start
        logger.info(f"Orpheus TTS model loaded in {load_time:.1f}s on {DEVICE}")

        return True
    except Exception as e:
        logger.error(f"Failed to load Orpheus TTS model: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - load model on startup"""
    logger.info("=" * 50)
    logger.info("  Sync2 Orpheus TTS Service - Starting")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 50)

    try:
        load_orpheus_model()
        logger.info("Orpheus TTS ready")
        logger.info(f"Available voices: {len(ALL_VOICES)}")
    except Exception as e:
        logger.error(f"Failed to initialize Orpheus TTS: {e}")
        raise

    yield

    logger.info("Shutting down Orpheus TTS Service...")


# Create FastAPI app
app = FastAPI(
    title="Sync2 Orpheus TTS Service",
    description="Human-like TTS service using Orpheus for Sync2.ai Voice AI Platform",
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
    voice: str = "tara"
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


def synthesize_with_orpheus(text: str, voice: str = "tara", speed: float = 1.0) -> np.ndarray:
    """
    Synthesize speech using Orpheus TTS
    Returns numpy array of audio samples
    """
    global tts_model

    if tts_model is None:
        raise RuntimeError("Orpheus TTS model not loaded")

    try:
        # Validate voice
        if voice not in ALL_VOICES:
            voice = DEFAULT_VOICE

        # Generate audio with Orpheus (streaming output)
        syn_tokens = tts_model.generate_speech(
            prompt=text,
            voice=voice
        )

        # Collect all audio chunks
        audio_chunks = []
        for audio_chunk in syn_tokens:
            audio_chunks.append(audio_chunk)

        # Combine chunks into single audio array
        if audio_chunks:
            audio_bytes = b''.join(audio_chunks)
            # Convert bytes to numpy array (16-bit PCM)
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        else:
            audio = np.array([], dtype=np.float32)

        # Normalize
        if len(audio) > 0 and np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95

        return audio.astype(np.float32)

    except Exception as e:
        logger.error(f"Orpheus TTS synthesis error: {e}")
        raise


# REST Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

    return {
        "status": "ok" if tts_model is not None else "error",
        "model": "Orpheus-TTS",
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
        # Generate audio with Orpheus TTS
        full_audio = synthesize_with_orpheus(
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
    2. Client sends JSON: {"text": "Hello", "voice": "tara", "format": "ulaw"}
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
                # Generate audio with Orpheus TTS
                audio = synthesize_with_orpheus(text, voice, speed)

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
    print("  Sync2 Orpheus TTS Service")
    print(f"  Device: {DEVICE}")
    print(f"  GPU: {torch.cuda.is_available()}")
    print("=" * 50)
    print("  Endpoints:")
    print("    GET  http://localhost:8765/health")
    print("    GET  http://localhost:8765/voices")
    print("    POST http://localhost:8765/tts/synthesize")
    print("    WS   ws://localhost:8765/tts/stream")
    print("=" * 50)
    print("  Voices: tara, leah, jess, mia, zoe, leo, dan, zac")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8765)
