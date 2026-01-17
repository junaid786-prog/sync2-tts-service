"""
FastAPI + WebSocket Server for TTS Service
Uses existing Kokoro TTS installation from D:\Live Projects\kokoro-tts
"""
import asyncio
import logging
import time
import sys
import os
import io
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Fix encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global pipeline
pipeline = None
SAMPLE_RATE = 24000
OUTPUT_SAMPLE_RATE = 8000

# Available voices (from your existing kokoro-tts setup)
VOICES = {
    "American English": {
        "female": ["af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"],
        "male": ["am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa"]
    },
    "British English": {
        "female": ["bf_alice", "bf_emma", "bf_isabella", "bf_lily"],
        "male": ["bm_daniel", "bm_fable", "bm_george", "bm_lewis"]
    },
    "Japanese": {
        "female": ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro"],
        "male": ["jm_kumo"]
    },
    "Mandarin Chinese": {
        "female": ["zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi"],
        "male": ["zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"]
    },
    "Spanish": {
        "female": ["ef_dora"],
        "male": ["em_alex", "em_santa"]
    },
    "French": {
        "female": ["ff_siwis"],
        "male": []
    },
    "Hindi": {
        "female": ["hf_alpha", "hf_beta"],
        "male": ["hm_omega", "hm_psi"]
    },
    "Italian": {
        "female": ["if_sara"],
        "male": ["im_nicola"]
    },
    "Brazilian Portuguese": {
        "female": ["pf_dora"],
        "male": ["pm_alex", "pm_santa"]
    }
}

# Flatten voices for quick lookup
ALL_VOICES = []
for lang, genders in VOICES.items():
    ALL_VOICES.extend(genders.get("female", []))
    ALL_VOICES.extend(genders.get("male", []))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - load model on startup"""
    global pipeline

    logger.info("=" * 50)
    logger.info("  Sync2 TTS Service - Starting")
    logger.info("=" * 50)

    try:
        from kokoro import KPipeline

        logger.info("Loading Kokoro TTS pipeline...")
        start = time.time()
        pipeline = KPipeline(lang_code='a')  # 'a' for auto language detection
        logger.info(f"Pipeline loaded in {time.time() - start:.1f}s")
        logger.info(f"Available voices: {len(ALL_VOICES)}")
    except ImportError as e:
        logger.error(f"Kokoro not installed: {e}")
        logger.error("Install with: pip install kokoro")
        raise
    except Exception as e:
        logger.error(f"Failed to load Kokoro: {e}")
        raise

    yield

    logger.info("Shutting down TTS Service...")


# Create FastAPI app
app = FastAPI(
    title="Sync2 TTS Service",
    description="Open-source TTS service using Kokoro for Sync2.ai Voice AI Platform",
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
    voice: str = "af_sarah"
    speed: float = 1.0
    output_format: str = "wav"  # "wav", "pcm16", "ulaw"


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate"""
    if orig_sr == target_sr:
        return audio

    # Simple resampling using numpy interpolation
    duration = len(audio) / orig_sr
    target_length = int(duration * target_sr)
    indices = np.linspace(0, len(audio) - 1, target_length)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def audio_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float audio to 16-bit PCM bytes"""
    # Normalize to [-1, 1]
    audio = np.clip(audio, -1.0, 1.0)
    # Convert to int16
    pcm = (audio * 32767).astype(np.int16)
    return pcm.tobytes()


def audio_to_ulaw(audio: np.ndarray) -> bytes:
    """Convert float audio to u-law encoded bytes"""
    import audioop
    pcm = audio_to_pcm16(audio)
    return audioop.lin2ulaw(pcm, 2)


# REST Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model": "Kokoro-82M",
        "model_loaded": pipeline is not None,
        "device": "cpu",
        "gpu_available": False,
        "gpu_name": None,
        "ready": pipeline is not None,
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
        "default": "af_sarah"
    }


@app.get("/api/voices")
async def list_voices_detailed():
    """List available voices organized by language"""
    return {
        "voices": VOICES,
        "all_voices": ALL_VOICES,
        "default": "af_sarah"
    }


@app.post("/tts/synthesize")
async def synthesize(request: TTSRequest):
    """
    Synthesize speech from text
    Returns audio in requested format (wav, pcm16, or ulaw)
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.voice not in ALL_VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice: {request.voice}. Use /voices to see available voices."
        )

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    start = time.time()

    try:
        # Generate audio
        generator = pipeline(request.text, voice=request.voice, speed=request.speed)

        # Collect all audio chunks
        audio_chunks = []
        for gs, ps, audio in generator:
            audio_chunks.append(audio)

        if not audio_chunks:
            raise HTTPException(status_code=500, detail="No audio generated")

        # Concatenate all chunks
        full_audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
        gen_time = time.time() - start

        logger.info(f"[TTS] Generated {len(request.text)} chars in {gen_time*1000:.0f}ms")

        # Process based on output format
        if request.output_format == "wav":
            # Return full WAV file at original sample rate
            import soundfile as sf
            buffer = io.BytesIO()
            sf.write(buffer, full_audio, SAMPLE_RATE, format='WAV')
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
    """Generate speech - returns WAV file (compatible with kokoro-tts API)"""
    request.output_format = "wav"
    return await synthesize(request)


# WebSocket Endpoint for Streaming
@app.websocket("/tts/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming TTS

    Protocol:
    1. Client connects
    2. Client sends JSON: {"text": "Hello", "voice": "af_sarah", "format": "ulaw"}
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
            voice = data.get("voice", "af_sarah")
            output_format = data.get("format", "ulaw")
            speed = data.get("speed", 1.0)

            if not text:
                await websocket.send_json({"error": "Text is required"})
                continue

            if pipeline is None:
                await websocket.send_json({"error": "Model not loaded"})
                continue

            logger.info(f"[WS] Streaming: '{text[:50]}...' voice={voice}")

            start_time = time.time()
            total_bytes = 0

            try:
                # Generate audio
                generator = pipeline(text, voice=voice, speed=speed)

                # Stream chunks
                for gs, ps, audio_chunk in generator:
                    # Resample to 8kHz
                    resampled = resample_audio(audio_chunk, SAMPLE_RATE, OUTPUT_SAMPLE_RATE)

                    # Convert to requested format
                    if output_format == "ulaw":
                        chunk_bytes = audio_to_ulaw(resampled)
                    else:
                        chunk_bytes = audio_to_pcm16(resampled)

                    await websocket.send_bytes(chunk_bytes)
                    total_bytes += len(chunk_bytes)
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
    print("  Sync2 TTS Service (Kokoro)")
    print("=" * 50)
    print("  Endpoints:")
    print("    GET  http://localhost:8765/health")
    print("    GET  http://localhost:8765/voices")
    print("    POST http://localhost:8765/tts/synthesize")
    print("    WS   ws://localhost:8765/tts/stream")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8765)
