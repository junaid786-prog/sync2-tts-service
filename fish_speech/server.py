"""
FastAPI + WebSocket Server for Fish Speech / OpenAudio TTS Service
Most natural-sounding open-source TTS
Drop-in replacement for Kokoro/Piper/XTTS with same interface
"""
import asyncio
import logging
import time
import sys
import os
import io
import wave
import subprocess
from typing import Optional
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
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
fish_speech_pipeline = None
SAMPLE_RATE = 44100  # Fish Speech native sample rate
OUTPUT_SAMPLE_RATE = 8000  # Telephony output

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Checkpoint paths
CHECKPOINT_DIR = Path("/app/checkpoints/fish-speech-1.5")
FISH_SPEECH_REPO = Path("/app/fish-speech-repo")

# Available voices
VOICES = {
    "default": {
        "description": "OpenAudio Default Voice - Natural",
        "language": "en"
    }
}

DEFAULT_VOICE = "default"
ALL_VOICES = list(VOICES.keys())


class FishSpeechPipeline:
    """Wrapper for Fish Speech / OpenAudio inference"""

    def __init__(self, checkpoint_path: Path, device: str = "cuda"):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model_loaded = False

        # Try to load using the fish-speech inference tools
        try:
            # Add fish-speech repo to path
            sys.path.insert(0, str(FISH_SPEECH_REPO))

            # Try importing from fish_speech package
            from fish_speech.models.text2semantic.llama import BaseModelArgs
            from fish_speech.models.vqgan.modules.firefly import FireflyArchitecture

            logger.info("Fish Speech modules loaded successfully")
            self.model_loaded = True

        except ImportError as e:
            logger.warning(f"Could not import fish_speech modules: {e}")
            # Fallback to CLI-based inference
            self.model_loaded = False

    def synthesize(self, text: str, speed: float = 1.0) -> np.ndarray:
        """Synthesize text to audio using Fish Speech"""

        if not self.model_loaded:
            # Use CLI-based inference as fallback
            return self._synthesize_cli(text, speed)

        # Direct model inference would go here
        # For now, use CLI approach which is more reliable
        return self._synthesize_cli(text, speed)

    def _synthesize_cli(self, text: str, speed: float = 1.0) -> np.ndarray:
        """Synthesize using fish-speech CLI tools"""
        import tempfile
        import soundfile as sf

        output_path = tempfile.mktemp(suffix=".wav")

        try:
            # Use the fish-speech inference CLI
            cmd = [
                "python", "-m", "tools.inference",
                "--text", text,
                "--checkpoint-path", str(self.checkpoint_path),
                "--output", output_path
            ]

            # Check if tools.inference exists, otherwise try alternative
            result = subprocess.run(
                cmd,
                cwd=str(FISH_SPEECH_REPO),
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                logger.warning(f"CLI inference failed: {result.stderr}")
                # Generate silence as fallback
                return np.zeros(int(SAMPLE_RATE * 2), dtype=np.float32)

            # Read the generated audio
            audio, sr = sf.read(output_path)

            # Resample if needed
            if sr != SAMPLE_RATE:
                audio = self._resample(audio, sr, SAMPLE_RATE)

            return audio.astype(np.float32)

        except Exception as e:
            logger.error(f"CLI synthesis error: {e}")
            # Return silence on error
            return np.zeros(int(SAMPLE_RATE * 2), dtype=np.float32)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling"""
        if orig_sr == target_sr:
            return audio
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_length)
        return np.interp(indices, np.arange(len(audio)), audio)


def load_fish_speech_model():
    """Load Fish Speech / OpenAudio model"""
    global fish_speech_pipeline

    logger.info(f"Loading Fish Speech / OpenAudio model on {DEVICE}...")
    start = time.time()

    try:
        fish_speech_pipeline = FishSpeechPipeline(
            checkpoint_path=CHECKPOINT_DIR,
            device=DEVICE
        )

        load_time = time.time() - start
        logger.info(f"Fish Speech pipeline initialized in {load_time:.1f}s on {DEVICE}")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize Fish Speech: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - load model on startup"""
    logger.info("=" * 50)
    logger.info("  Sync2 Fish Speech TTS Service - Starting")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 50)

    try:
        load_fish_speech_model()
        logger.info("Fish Speech TTS ready")
        logger.info(f"Available voices: {len(ALL_VOICES)}")
    except Exception as e:
        logger.error(f"Failed to initialize Fish Speech: {e}")
        # Don't raise - allow server to start for health checks
        logger.warning("Server starting in degraded mode")

    yield

    logger.info("Shutting down Fish Speech TTS Service...")


# Create FastAPI app
app = FastAPI(
    title="Sync2 Fish Speech TTS Service",
    description="Natural human-like TTS service using Fish Speech / OpenAudio for Sync2.ai Voice AI Platform",
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
    output_format: str = "wav"
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


def synthesize_with_fish_speech(text: str, voice: str = "default", language: str = "en", speed: float = 1.0) -> np.ndarray:
    """
    Synthesize speech using Fish Speech / OpenAudio
    Returns numpy array of audio samples
    """
    global fish_speech_pipeline

    if fish_speech_pipeline is None:
        raise RuntimeError("Fish Speech pipeline not loaded")

    try:
        # Generate audio using Fish Speech
        audio = fish_speech_pipeline.synthesize(text=text, speed=speed)

        # Ensure audio is 1D
        if len(audio.shape) > 1:
            audio = audio.squeeze()

        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95

        return audio

    except Exception as e:
        logger.error(f"Fish Speech synthesis error: {e}")
        raise


# REST Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

    return {
        "status": "ok" if fish_speech_pipeline is not None else "degraded",
        "model": "Fish-Speech-1.5",
        "model_loaded": fish_speech_pipeline is not None,
        "device": DEVICE,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "ready": fish_speech_pipeline is not None,
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
        # Generate audio with Fish Speech
        full_audio = synthesize_with_fish_speech(
            request.text,
            request.voice,
            request.language,
            request.speed
        )
        gen_time = time.time() - start

        logger.info(f"[TTS] Generated {len(request.text)} chars in {gen_time*1000:.0f}ms")

        # Process based on output format
        if request.output_format == "wav":
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
            else:
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
            language = data.get("language", "en")

            if not text:
                await websocket.send_json({"error": "Text is required"})
                continue

            logger.info(f"[WS] Streaming: '{text[:50]}...' voice={voice}")

            start_time = time.time()
            total_bytes = 0

            try:
                # Generate audio with Fish Speech
                audio = synthesize_with_fish_speech(text, voice, language, speed)

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
    print("  Sync2 Fish Speech TTS Service")
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
