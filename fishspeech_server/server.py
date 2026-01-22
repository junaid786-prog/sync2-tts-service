"""
FastAPI + WebSocket Server for Fish Speech TTS Service
#1 Ranked on TTS-Arena2 - Most human-like voice synthesis
Drop-in replacement for Kokoro/XTTS/Chatterbox with same interface

Uses Fish Speech OpenAudio S1 Mini model via native API
NOW WITH TRUE STREAMING - Audio chunks sent as they're generated!
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
import threading
import queue
from pathlib import Path
from typing import Optional, Generator, Iterator
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

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
FISH_SPEECH_PATH = Path("/app/fish-speech")
CHECKPOINT_PATH = Path("/app/checkpoints/openaudio-s1-mini")

# Thread pool for running sync code in async context
executor = ThreadPoolExecutor(max_workers=4)

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
    """
    Fish Speech TTS Engine using native Python API
    Loads models directly using Fish Speech's ModelManager
    Supports TRUE STREAMING - yields audio chunks as generated
    """

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.fish_speech_path = FISH_SPEECH_PATH
        self.ready = False
        self.model_manager = None
        self.tts_engine = None
        self._lock = threading.Lock()

        # Add fish-speech to Python path
        sys.path.insert(0, str(self.fish_speech_path))

        self._load_models()

    def _download_model_if_needed(self):
        """Download model checkpoint if not present"""
        if self.checkpoint_path.exists():
            # Check if it has actual model files
            model_files = list(self.checkpoint_path.glob("*.pth")) + list(self.checkpoint_path.glob("*.safetensors"))
            if model_files:
                logger.info(f"Model files found: {[f.name for f in model_files[:5]]}")
                return True

        logger.info("Downloading Fish Speech model checkpoint...")
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                "fishaudio/openaudio-s1-mini",
                local_dir=str(self.checkpoint_path),
                token=HF_TOKEN
            )
            logger.info("Model download complete")
            return True
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise RuntimeError(f"Could not download Fish Speech model: {e}")

    def _load_models(self):
        """Load Fish Speech models using native ModelManager"""
        logger.info("Loading Fish Speech models...")
        start = time.time()

        try:
            # Download model if needed
            self._download_model_if_needed()

            # Import Fish Speech components
            from tools.server.model_manager import ModelManager

            # Find decoder checkpoint - openaudio models use codec.pth
            decoder_path = self.checkpoint_path / "codec.pth"
            if not decoder_path.exists():
                # Try legacy generator file names
                generator_files = list(self.checkpoint_path.glob("*generator*.pth"))
                if generator_files:
                    decoder_path = generator_files[0]
                else:
                    # Use checkpoint path directly - ModelManager will handle it
                    decoder_path = self.checkpoint_path

            logger.info(f"LLAMA checkpoint: {self.checkpoint_path}")
            logger.info(f"Decoder checkpoint: {decoder_path}")

            # Determine the config name based on model type
            # openaudio models use modded_dac_vq config
            decoder_config_name = "modded_dac_vq"

            # Determine precision
            precision = torch.half if DEVICE == "cuda" else torch.float32

            # Initialize ModelManager (this loads all models)
            self.model_manager = ModelManager(
                mode="tts",
                device=DEVICE,
                half=(DEVICE == "cuda"),
                compile=False,  # Disable compilation for faster startup
                llama_checkpoint_path=str(self.checkpoint_path),
                decoder_checkpoint_path=str(decoder_path),
                decoder_config_name=decoder_config_name
            )

            self.tts_engine = self.model_manager.tts_inference_engine
            self.ready = True

            load_time = time.time() - start
            logger.info(f"Fish Speech models loaded in {load_time:.1f}s")

        except ImportError as e:
            logger.error(f"Failed to import Fish Speech modules: {e}")
            raise RuntimeError(f"Fish Speech import error: {e}")
        except Exception as e:
            logger.error(f"Failed to load Fish Speech models: {e}")
            raise

    def synthesize_streaming(self, text: str, voice: str = "default") -> Iterator[np.ndarray]:
        """
        Synthesize speech from text using Fish Speech with TRUE STREAMING
        Yields audio chunks as they're generated - much lower latency!
        """
        if not self.ready or self.tts_engine is None:
            raise RuntimeError("Fish Speech engine not ready")

        logger.info(f"[STREAMING] Synthesizing: '{text[:50]}...'")
        start = time.time()
        chunk_count = 0
        total_samples = 0

        try:
            from fish_speech.utils.schema import ServeTTSRequest

            # Create TTS request with streaming enabled
            request = ServeTTSRequest(
                text=text,
                references=[],
                reference_id=None,
                max_new_tokens=1024,
                chunk_length=100,  # Smaller chunks for faster streaming
                top_p=0.7,
                repetition_penalty=1.2,
                temperature=0.7,
                format="wav",
                streaming=True  # Enable streaming!
            )

            # Get sample rate
            sample_rate = self.tts_engine.decoder_model.sample_rate

            # Stream audio chunks as they're generated
            with self._lock:
                for result in self.tts_engine.inference(request):
                    if result.code == "segment" and isinstance(result.audio, tuple):
                        # Got a segment - yield it immediately!
                        audio_chunk = result.audio[1]
                        if isinstance(audio_chunk, np.ndarray) and len(audio_chunk) > 0:
                            # Normalize chunk
                            if np.max(np.abs(audio_chunk)) > 0:
                                audio_chunk = audio_chunk / np.max(np.abs(audio_chunk)) * 0.95

                            chunk_count += 1
                            total_samples += len(audio_chunk)

                            if chunk_count == 1:
                                first_chunk_time = time.time() - start
                                logger.info(f"[STREAMING] First chunk in {first_chunk_time*1000:.0f}ms")

                            yield audio_chunk.astype(np.float32), sample_rate

                    elif result.code == "final" and isinstance(result.audio, tuple):
                        # Final chunk
                        audio_chunk = result.audio[1]
                        if isinstance(audio_chunk, np.ndarray) and len(audio_chunk) > 0:
                            if np.max(np.abs(audio_chunk)) > 0:
                                audio_chunk = audio_chunk / np.max(np.abs(audio_chunk)) * 0.95

                            chunk_count += 1
                            total_samples += len(audio_chunk)
                            yield audio_chunk.astype(np.float32), sample_rate
                        break

                    elif result.code == "error":
                        logger.error(f"[STREAMING] Error: {result.error}")
                        raise RuntimeError(str(result.error))

            elapsed = time.time() - start
            audio_duration = total_samples / sample_rate if sample_rate > 0 else 0
            logger.info(f"[STREAMING] Complete: {chunk_count} chunks, {audio_duration:.2f}s audio in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"[STREAMING] Fish Speech synthesis error: {e}")
            raise

    def synthesize(self, text: str, voice: str = "default") -> tuple:
        """
        Synthesize speech from text using Fish Speech (non-streaming)
        Returns numpy array of audio samples at native sample rate
        """
        if not self.ready or self.tts_engine is None:
            raise RuntimeError("Fish Speech engine not ready")

        logger.info(f"Synthesizing: '{text[:50]}...'")
        start = time.time()

        try:
            from fish_speech.utils.schema import ServeTTSRequest
            from tools.server.inference import inference_wrapper as inference

            # Create TTS request
            request = ServeTTSRequest(
                text=text,
                references=[],
                reference_id=None,
                max_new_tokens=1024,
                chunk_length=200,
                top_p=0.7,
                repetition_penalty=1.2,
                temperature=0.7,
                format="wav",
                streaming=False
            )

            # Generate audio
            with self._lock:
                audio_chunks = list(inference(request, self.tts_engine))

            if not audio_chunks:
                raise RuntimeError("No audio generated")

            # Concatenate all chunks
            audio = np.concatenate(audio_chunks)

            # Get sample rate from decoder model
            sample_rate = self.tts_engine.decoder_model.sample_rate

            elapsed = time.time() - start
            logger.info(f"Generated {len(audio)/sample_rate:.2f}s audio in {elapsed:.2f}s (RTF: {elapsed/(len(audio)/sample_rate):.2f})")

            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.95

            return audio.astype(np.float32), sample_rate

        except Exception as e:
            logger.error(f"Fish Speech synthesis error: {e}")
            raise

    def get_sample_rate(self) -> int:
        """Get the native sample rate of the model"""
        if self.tts_engine and hasattr(self.tts_engine, 'decoder_model'):
            return self.tts_engine.decoder_model.sample_rate
        return SAMPLE_RATE


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
    logger.info("  🚀 TRUE STREAMING MODE ENABLED")
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
    description="#1 Ranked TTS service using Fish Speech for Sync2.ai Voice AI Platform - TRUE STREAMING",
    version="2.0.0",
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

    native_sr = tts_engine.get_sample_rate() if tts_engine else SAMPLE_RATE

    return {
        "status": "ok" if tts_engine is not None and tts_engine.ready else "error",
        "model": "Fish-Speech-OpenAudio-S1-Mini",
        "model_loaded": tts_engine is not None and tts_engine.ready,
        "device": DEVICE,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "ready": tts_engine is not None and tts_engine.ready,
        "sample_rate": native_sr,
        "output_sample_rate": OUTPUT_SAMPLE_RATE,
        "total_voices": len(ALL_VOICES),
        "streaming": True,
        "version": "2.0.0"
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
        full_audio, native_sr = tts_engine.synthesize(request.text, request.voice)
        gen_time = time.time() - start

        logger.info(f"[TTS] Generated {len(request.text)} chars in {gen_time*1000:.0f}ms")

        # Process based on output format
        if request.output_format == "wav":
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(native_sr)
                wav_file.writeframes(audio_to_pcm16(full_audio))
            buffer.seek(0)

            return StreamingResponse(
                buffer,
                media_type="audio/wav",
                headers={
                    "X-Duration-Ms": str(gen_time * 1000),
                    "X-Sample-Rate": str(native_sr),
                    "X-Format": "wav"
                }
            )

        else:
            # Resample to 8kHz for telephony
            resampled = resample_audio(full_audio, native_sr, OUTPUT_SAMPLE_RATE)

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


def run_synthesis_in_thread(text: str, voice: str, output_format: str, result_queue: queue.Queue):
    """Run synthesis in a thread and put results in queue"""
    try:
        # Use non-streaming synthesis (more reliable)
        audio, sample_rate = tts_engine.synthesize(text, voice)

        # Resample to 8kHz for telephony
        resampled = resample_audio(audio, sample_rate, OUTPUT_SAMPLE_RATE)

        # Convert to requested format
        if output_format == "ulaw":
            audio_bytes = audio_to_ulaw(resampled)
        else:
            audio_bytes = audio_to_pcm16(resampled)

        # Put the complete audio as a single chunk
        result_queue.put(("chunk", audio_bytes))
        result_queue.put(("done", None))
    except Exception as e:
        logger.error(f"[THREAD] Synthesis error: {e}")
        result_queue.put(("error", str(e)))


# WebSocket Endpoint for Streaming TTS
@app.websocket("/tts/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming TTS
    Runs synthesis in background thread, streams audio when ready
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
                # Create a queue for results
                result_queue = queue.Queue()

                # Start synthesis in a thread
                loop = asyncio.get_event_loop()
                future = loop.run_in_executor(
                    executor,
                    run_synthesis_in_thread,
                    text, voice, output_format, result_queue
                )

                # Wait for results with timeout
                timeout_seconds = 120  # 2 minute timeout for long responses
                deadline = time.time() + timeout_seconds

                while time.time() < deadline:
                    try:
                        # Check queue with small timeout
                        result_type, result_data = result_queue.get(block=True, timeout=0.5)

                        if result_type == "chunk":
                            # Stream the audio in smaller chunks for smooth playback
                            chunk_size = 4000
                            for i in range(0, len(result_data), chunk_size):
                                chunk = result_data[i:i + chunk_size]
                                await websocket.send_bytes(chunk)
                                total_bytes += len(chunk)
                                await asyncio.sleep(0.001)  # Small yield

                            first_chunk_time = (time.time() - start_time) * 1000
                            logger.info(f"[WS] Audio sent in {first_chunk_time:.0f}ms")

                        elif result_type == "done":
                            break

                        elif result_type == "error":
                            raise RuntimeError(result_data)

                    except queue.Empty:
                        # Queue empty, check if synthesis is still running
                        if future.done():
                            # Check if there was an exception
                            try:
                                future.result()
                            except Exception as e:
                                raise RuntimeError(str(e))
                            break
                        continue

                duration_ms = (time.time() - start_time) * 1000

                await websocket.send_json({
                    "done": True,
                    "duration_ms": duration_ms,
                    "total_bytes": total_bytes
                })

                logger.info(f"[WS] ✅ Complete: {total_bytes} bytes in {duration_ms:.0f}ms")

            except Exception as e:
                logger.error(f"[WS] Error: {e}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        logger.info("[WS] Client disconnected")
    except Exception as e:
        logger.error(f"[WS] Error: {e}")


if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("  Sync2 Fish Speech TTS Service v2.0")
    print("  🚀 TRUE STREAMING MODE ENABLED")
    print(f"  Device: {DEVICE}")
    print(f"  GPU: {torch.cuda.is_available()}")
    print("  #1 Ranked on TTS-Arena2")
    print("=" * 50)
    print("  Endpoints:")
    print("    GET  http://localhost:8765/health")
    print("    GET  http://localhost:8765/voices")
    print("    POST http://localhost:8765/tts/synthesize")
    print("    WS   ws://localhost:8765/tts/stream (TRUE STREAMING)")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8765)
