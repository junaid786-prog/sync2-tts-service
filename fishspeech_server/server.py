"""
FastAPI + WebSocket Server for Fish Speech TTS Service
#1 Ranked on TTS-Arena2 - Most human-like voice synthesis
Drop-in replacement for Kokoro/XTTS/Chatterbox with same interface

Uses Fish Speech OpenAudio S1 Mini model via subprocess CLI
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
import shutil
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

# Fish Speech paths
FISH_SPEECH_PATH = "/app/fish-speech"
CHECKPOINT_PATH = "/app/checkpoints/openaudio-s1-mini"

# Available voices
VOICES = {
    "default": {
        "description": "Fish Speech Default Voice",
        "language": "en"
    }
}

DEFAULT_VOICE = "default"
ALL_VOICES = list(VOICES.keys())


class FishSpeechEngine:
    """Wrapper for Fish Speech inference using subprocess CLI"""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.fish_speech_path = Path(FISH_SPEECH_PATH)
        self.ready = False
        self._verify_installation()

    def _verify_installation(self):
        """Verify Fish Speech is properly installed"""
        logger.info("Verifying Fish Speech installation...")

        # Check if fish-speech directory exists
        if not self.fish_speech_path.exists():
            raise RuntimeError(f"Fish Speech not found at {self.fish_speech_path}")

        # Check for tools directory
        tools_path = self.fish_speech_path / "tools"
        if not tools_path.exists():
            raise RuntimeError(f"Fish Speech tools not found at {tools_path}")

        # List available tools
        logger.info(f"Fish Speech tools found: {list(tools_path.iterdir())}")

        # Check checkpoint
        if not self.checkpoint_path.exists():
            logger.warning(f"Checkpoint not found at {self.checkpoint_path}, will download on first use")

        # Check for required model files
        model_files = list(self.checkpoint_path.glob("*.pth")) if self.checkpoint_path.exists() else []
        logger.info(f"Model files found: {[f.name for f in model_files]}")

        self.ready = True
        logger.info("Fish Speech verification complete")

    def synthesize(self, text: str, voice: str = "default") -> np.ndarray:
        """
        Synthesize speech from text using Fish Speech CLI
        Returns numpy array of audio samples at 44100 Hz
        """
        if not self.ready:
            raise RuntimeError("Fish Speech engine not ready")

        logger.info(f"Synthesizing: '{text[:50]}...'")
        start = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.wav"
            codes_path = Path(tmpdir) / "codes.npy"

            try:
                # Method 1: Try using the inference module directly
                env = os.environ.copy()
                env["PYTHONPATH"] = str(self.fish_speech_path)

                # Step 1: Generate semantic codes from text
                logger.info("Generating semantic codes...")
                cmd_generate = [
                    sys.executable, "-m", "tools.llama.generate",
                    "--text", text,
                    "--checkpoint-path", str(self.checkpoint_path),
                    "--output-path", str(codes_path),
                    "--device", DEVICE,
                    "--compile", "False",
                    "--num-samples", "1",
                    "--max-new-tokens", "0",  # Auto determine
                    "--top-p", "0.7",
                    "--temperature", "0.7",
                    "--repetition-penalty", "1.2"
                ]

                result = subprocess.run(
                    cmd_generate,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(self.fish_speech_path),
                    env=env
                )

                if result.returncode != 0:
                    logger.warning(f"Generate stderr: {result.stderr}")
                    # Try alternative: direct webui inference
                    return self._synthesize_webui(text, output_path, env)

                if not codes_path.exists():
                    logger.warning("Codes file not created, trying webui method")
                    return self._synthesize_webui(text, output_path, env)

                # Step 2: Decode codes to audio using VQGAN
                logger.info("Decoding to audio...")

                # Find the decoder model
                decoder_path = self.checkpoint_path / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
                if not decoder_path.exists():
                    # Try alternative name
                    decoder_files = list(self.checkpoint_path.glob("*generator*.pth"))
                    if decoder_files:
                        decoder_path = decoder_files[0]
                    else:
                        decoder_path = self.checkpoint_path / "firefly_gan_vq"

                cmd_decode = [
                    sys.executable, "-m", "tools.vqgan.inference",
                    "--input-path", str(codes_path),
                    "--output-path", str(output_path),
                    "--checkpoint-path", str(decoder_path),
                    "--device", DEVICE
                ]

                result = subprocess.run(
                    cmd_decode,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=str(self.fish_speech_path),
                    env=env
                )

                if result.returncode != 0:
                    logger.warning(f"Decode stderr: {result.stderr}")
                    return self._synthesize_webui(text, output_path, env)

                if not output_path.exists():
                    return self._synthesize_webui(text, output_path, env)

                # Load and return audio
                audio, sr = sf.read(str(output_path))
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)

                elapsed = time.time() - start
                logger.info(f"Generated {len(audio)} samples in {elapsed:.2f}s")

                # Normalize
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio)) * 0.95

                return audio.astype(np.float32)

            except subprocess.TimeoutExpired:
                logger.error("Fish Speech inference timed out")
                raise RuntimeError("TTS generation timed out")
            except Exception as e:
                logger.error(f"Fish Speech synthesis error: {e}")
                raise

    def _synthesize_webui(self, text: str, output_path: Path, env: dict) -> np.ndarray:
        """Fallback: Try using the webui inference API"""
        logger.info("Trying webui inference method...")

        try:
            # Escape text for use in the script
            escaped_text = text.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")

            # Create a simple inference script
            script = f'''
import sys
sys.path.insert(0, "{self.fish_speech_path}")
import os
os.chdir("{self.fish_speech_path}")

try:
    from tools.inference_engine import TTSInferenceEngine

    engine = TTSInferenceEngine(
        llama_checkpoint_path="{self.checkpoint_path}",
        decoder_checkpoint_path="{self.checkpoint_path}",
        device="{DEVICE}"
    )

    result = engine.inference(text="{escaped_text}")

    import soundfile as sf
    sf.write("{output_path}", result["audio"], result["sr"])
    print("SUCCESS")

except ImportError as ie:
    print(f"IMPORT_ERROR: {{ie}}")
except Exception as e:
    print(f"ERROR: {{e}}")
'''
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=120,
                env=env
            )

            if "SUCCESS" in result.stdout and output_path.exists():
                audio, sr = sf.read(str(output_path))
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio)) * 0.95
                return audio.astype(np.float32)

            logger.warning(f"Webui inference output: {result.stdout}")
            logger.warning(f"Webui inference error: {result.stderr}")

            # Last resort: try the API webui
            return self._synthesize_api_server(text, output_path, env)

        except Exception as e:
            logger.error(f"Webui synthesis failed: {e}")
            raise

    def _synthesize_api_server(self, text: str, output_path: Path, env: dict) -> np.ndarray:
        """Last resort: Start Fish Speech API server and call it"""
        logger.info("Trying API server method...")

        # For now, raise an error - we'll need to implement a proper fallback
        raise RuntimeError(
            "Fish Speech inference failed. The model may need different configuration. "
            "Please check /app/fish-speech for available inference methods."
        )


def load_fishspeech_engine():
    """Load Fish Speech TTS engine"""
    global tts_engine

    logger.info(f"Loading Fish Speech TTS on {DEVICE}...")
    start = time.time()

    try:
        tts_engine = FishSpeechEngine(CHECKPOINT_PATH)

        load_time = time.time() - start
        logger.info(f"Fish Speech TTS engine ready in {load_time:.1f}s")

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
    logger.info(f"  Fish Speech Path: {FISH_SPEECH_PATH}")
    logger.info(f"  Checkpoint: {CHECKPOINT_PATH}")
    logger.info("=" * 50)

    try:
        load_fishspeech_engine()
        logger.info("Fish Speech TTS ready")
    except Exception as e:
        logger.error(f"Failed to initialize Fish Speech TTS: {e}")
        # Don't raise - let the server start but report unhealthy
        logger.warning("Server starting in degraded mode")

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
        "status": "ok" if tts_engine is not None and tts_engine.ready else "error",
        "model": "Fish-Speech-OpenAudio-S1-Mini",
        "model_loaded": tts_engine is not None and tts_engine.ready,
        "device": DEVICE,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "ready": tts_engine is not None and tts_engine.ready,
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

    if tts_engine is None or not tts_engine.ready:
        raise HTTPException(status_code=503, detail="TTS engine not ready")

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
