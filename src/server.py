"""
FastAPI + WebSocket Server for TTS Service
Provides REST and WebSocket endpoints for text-to-speech synthesis
"""
import asyncio
import logging
import time
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

from .config import config, AVAILABLE_VOICES
from .tts_engine import KokoroTTSEngine
from .audio_processor import AudioProcessor

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
tts_engine: Optional[KokoroTTSEngine] = None
audio_processor: Optional[AudioProcessor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - load model on startup"""
    global tts_engine, audio_processor

    logger.info("Starting TTS Service...")

    # Initialize TTS engine
    try:
        tts_engine = KokoroTTSEngine(use_gpu=config.use_gpu)
        logger.info("TTS Engine initialized")
    except Exception as e:
        logger.error(f"Failed to initialize TTS Engine: {e}")
        raise

    # Initialize audio processor
    audio_processor = AudioProcessor(
        input_sample_rate=config.input_sample_rate,
        output_sample_rate=config.output_sample_rate,
        chunk_duration_ms=config.chunk_duration_ms,
        fade_duration_ms=config.fade_duration_ms,
        normalize_db=config.normalize_db
    )
    logger.info("Audio Processor initialized")

    logger.info(f"TTS Service ready on {config.host}:{config.port}")

    yield

    # Cleanup
    logger.info("Shutting down TTS Service...")


# Create FastAPI app
app = FastAPI(
    title="Sync2 TTS Service",
    description="Open-source Text-to-Speech service for Sync2.ai Voice AI Platform",
    version="1.0.0",
    lifespan=lifespan
)


# Request/Response models
class SynthesizeRequest(BaseModel):
    text: str
    voice: str = "af_sarah"
    output_format: str = "ulaw"  # "ulaw" or "pcm16"


class SynthesizeResponse(BaseModel):
    success: bool
    duration_ms: float
    audio_length_bytes: int
    sample_rate: int
    format: str


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    gpu_available: bool
    gpu_name: Optional[str]
    ready: bool


# REST Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancer"""
    if not tts_engine or not tts_engine.is_ready():
        raise HTTPException(status_code=503, detail="TTS Engine not ready")

    info = tts_engine.get_info()
    return HealthResponse(
        status="ok",
        model=info["model"],
        device=info["device"],
        gpu_available=info["gpu_available"],
        gpu_name=info["gpu_name"],
        ready=info["ready"]
    )


@app.get("/voices")
async def list_voices():
    """List available voices"""
    return {
        "voices": AVAILABLE_VOICES,
        "default": config.default_voice
    }


@app.post("/tts/synthesize")
async def synthesize(request: SynthesizeRequest):
    """
    Synthesize speech from text (batch mode)
    Returns complete audio file
    """
    if not tts_engine or not audio_processor:
        raise HTTPException(status_code=503, detail="Service not ready")

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    if request.voice not in AVAILABLE_VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid voice. Available: {list(AVAILABLE_VOICES.keys())}"
        )

    try:
        start_time = time.time()

        # Synthesize audio
        audio = tts_engine.synthesize(request.text, request.voice)

        # Process audio for Asterisk
        processed = audio_processor.process(audio, request.output_format)

        duration_ms = (time.time() - start_time) * 1000

        logger.info(f"Synthesized {len(request.text)} chars in {duration_ms:.0f}ms")

        # Return audio as binary response
        content_type = "audio/basic" if request.output_format == "ulaw" else "audio/L16"

        return Response(
            content=processed,
            media_type=content_type,
            headers={
                "X-Duration-Ms": str(duration_ms),
                "X-Audio-Length": str(len(processed)),
                "X-Sample-Rate": str(config.output_sample_rate),
                "X-Format": request.output_format
            }
        )

    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    logger.info("WebSocket client connected")

    try:
        while True:
            # Receive synthesis request
            data = await websocket.receive_json()

            text = data.get("text", "")
            voice = data.get("voice", config.default_voice)
            output_format = data.get("format", "ulaw")

            if not text:
                await websocket.send_json({"error": "Text is required"})
                continue

            logger.info(f"Streaming TTS: '{text[:50]}...' voice={voice}")

            start_time = time.time()
            total_bytes = 0

            try:
                # Stream synthesis
                audio_generator = tts_engine.synthesize_stream(text, voice)

                # Process and stream chunks
                for chunk in audio_processor.process_stream(audio_generator, output_format):
                    await websocket.send_bytes(chunk)
                    total_bytes += len(chunk)
                    # Small delay to prevent overwhelming the client
                    await asyncio.sleep(0.001)

                duration_ms = (time.time() - start_time) * 1000

                # Send completion message
                await websocket.send_json({
                    "done": True,
                    "duration_ms": duration_ms,
                    "total_bytes": total_bytes
                })

                logger.info(f"Streamed {total_bytes} bytes in {duration_ms:.0f}ms")

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


def start_server():
    """Start the TTS service"""
    import uvicorn

    uvicorn.run(
        "src.server:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        reload=False
    )


if __name__ == "__main__":
    start_server()
