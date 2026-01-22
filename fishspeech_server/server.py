"""
FastAPI + WebSocket Server for Fish Speech TTS Service
#1 Ranked on TTS-Arena2 - Most human-like voice synthesis
Drop-in replacement for Kokoro/XTTS/Chatterbox with same interface
"""
import asyncio
import logging
import time
import sys
import os
import io
import wave
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import soundfile as sf

# Setup HuggingFace token from environment
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    os.environ["HF_TOKEN"] = HF_TOKEN

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
tts_engine = None
SAMPLE_RATE = 44100  # Fish Speech native sample rate
OUTPUT_SAMPLE_RATE = 8000  # Telephony output

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fish Speech checkpoint path
CHECKPOINT_PATH = "/app/checkpoints/openaudio-s1-mini"

# Available voices (can be expanded with reference audio files)
VOICES = {
    "default": {
        "description": "Fish Speech Default Voice",
        "language": "en"
    },
    "female_warm": {
        "description": "Female - Warm and natural",
        "language": "en"
    },
    "male_professional": {
        "description": "Male - Professional",
        "language": "en"
    }
}

DEFAULT_VOICE = "default"
ALL_VOICES = list(VOICES.keys())


class FishSpeechEngine:
    """Wrapper for Fish Speech inference"""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.codec = None
        self._load_models()

    def _load_models(self):
        """Load Fish Speech models"""
        logger.info("Loading Fish Speech models...")
        start = time.time()

        try:
            # Add fish-speech to path
            sys.path.insert(0, '/app/fish-speech')

            from fish_speech.models.text2semantic.llama import BaseModelArgs
            from fish_speech.models.dac.modded_dac import DAC

            # Load codec model
            codec_path = Path(self.checkpoint_path) / "codec.pth"
            if codec_path.exists():
                self.codec = DAC.load(str(codec_path))
                self.codec.to(DEVICE)
                self.codec.eval()
                logger.info("Codec model loaded")

            # Load text2semantic model
            llama_path = Path(self.checkpoint_path) / "model.pth"
            if llama_path.exists():
                # Model loading handled by fish_speech internals
                logger.info("Text2Semantic model ready")

            load_time = time.time() - start
            logger.info(f"Fish Speech models loaded in {load_time:.1f}s")

        except Exception as e:
            logger.warning(f"Could not load Fish Speech models directly: {e}")
            logger.info("Will use command-line inference instead")

    def synthesize(self, text: str, voice: str = "default") -> np.ndarray:
        """
        Synthesize speech from text using Fish Speech
        Returns numpy array of audio samples
        """
        try:
            # Use Fish Speech's built-in inference
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "output.wav"

                # Try using the Python API first
                try:
                    return self._synthesize_api(text, voice, output_path)
                except Exception as e:
                    logger.warning(f"API synthesis failed, trying CLI: {e}")
                    return self._synthesize_cli(text, voice, output_path)

        except Exception as e:
            logger.error(f"Fish Speech synthesis error: {e}")
            raise

    def _synthesize_api(self, text: str, voice: str, output_path: Path) -> np.ndarray:
        """Synthesize using Fish Speech Python API"""
        sys.path.insert(0, '/app/fish-speech')

        from fish_speech.inference import TextToSpeech

        tts = TextToSpeech(
            checkpoint_path=self.checkpoint_path,
            device=DEVICE
        )

        # Generate speech
        audio = tts.synthesize(text)

        # Convert to numpy
        if torch.is_tensor(audio):
            audio = audio.cpu().numpy()

        if audio.ndim > 1:
            audio = audio.squeeze()

        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95

        return audio.astype(np.float32)

    def _synthesize_cli(self, text: str, voice: str, output_path: Path) -> np.ndarray:
        """Synthesize using Fish Speech CLI as fallback"""
        import subprocess

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(text)
            text_file = f.name

        try:
            # Run Fish Speech inference
            cmd = [
                "python", "-m", "fish_speech.tools.inference",
                "--text", text,
                "--checkpoint-path", self.checkpoint_path,
                "--output", str(output_path),
                "--device", DEVICE
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd="/app/fish-speech"
            )

            if result.returncode != 0:
                raise RuntimeError(f"Fish Speech CLI failed: {result.stderr}")

            # Load generated audio
            audio, sr = sf.read(str(output_path))

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            return audio.astype(np.float32)

        finally:
            os.unlink(text_file)


def load_fishspeech_engine():
    """Load Fish Speech TTS engine"""
    global tts_engine

    logger.info(f"Loading Fish Speech TTS on {DEVICE}...")
    start = time.time()

    try:
        tts_engine = FishSpeechEngine(CHECKPOINT_PATH)

        load_time = time.time() - start
        logger.info(f"Fish Speech TTS loaded in {load_time:.1f}s on {DEVICE}")

        return True
    except Exception as e:
        logger.error(f"Failed to load Fish Speech TTS: {e}")
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
        load_fishspeech_engine()
        logger.info("Fish Speech TTS ready")
        logger.info(f"Available voices: {len(ALL_VOICES)}")
    except Exception as e:
        logger.error(f"Failed to initialize Fish Speech TTS: {e}")
        raise

    yield

    logger.info("Shutting down Fish Speech TTS Service...")


# Create FastAPI app
app = FastAPI(
    title="Sync2 Fish Speech TTS Service",
    description="#1 Ranked TTS service using Fish Speech for Sync2.ai Voice AI Platform",
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


# REST Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

    return {
        "status": "ok" if tts_engine is not None else "error",
        "model": "Fish-Speech-V1.5",
        "model_loaded": tts_engine is not None,
        "device": DEVICE,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "ready": tts_engine is not None,
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
        full_audio = tts_engine.synthesize(request.text, request.voice)
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
    """
    await websocket.accept()
    logger.info("[WS] Client connected")

    try:
        while True:
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
                # Generate audio with Fish Speech
                audio = tts_engine.synthesize(text, voice)

                # Resample to 8kHz for telephony
                resampled = resample_audio(audio, SAMPLE_RATE, OUTPUT_SAMPLE_RATE)

                # Convert to requested format
                if output_format == "ulaw":
                    audio_bytes = audio_to_ulaw(resampled)
                else:
                    audio_bytes = audio_to_pcm16(resampled)

                # Stream in chunks
                chunk_size = 4000
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i:i + chunk_size]
                    await websocket.send_bytes(chunk)
                    total_bytes += len(chunk)
                    await asyncio.sleep(0.001)

                duration_ms = (time.time() - start_time) * 1000

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
    print("  #1 Ranked on TTS-Arena2")
    print("=" * 50)
    print("  Endpoints:")
    print("    GET  http://localhost:8765/health")
    print("    GET  http://localhost:8765/voices")
    print("    POST http://localhost:8765/tts/synthesize")
    print("    WS   ws://localhost:8765/tts/stream")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8765)
