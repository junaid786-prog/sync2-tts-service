"""
FastAPI + WebSocket Server for CosyVoice2 TTS Service
Ultra-low latency streaming TTS (~150ms first chunk)
Drop-in replacement for Kokoro/XTTS/Chatterbox with same interface

Uses CosyVoice2-0.5B model for high-quality streaming synthesis
"""
import asyncio
import logging
import time
import sys
import os
import io
import wave
import threading
import queue
from pathlib import Path
from typing import Optional, Iterator
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
tts_model = None
SAMPLE_RATE = 22050  # CosyVoice2 native sample rate
OUTPUT_SAMPLE_RATE = 8000  # Telephony output

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model path
MODEL_DIR = Path("/app/models/CosyVoice2-0.5B")

# Thread pool for running sync code in async context
executor = ThreadPoolExecutor(max_workers=4)

# Available voices (CosyVoice2 supports voice cloning, these are preset styles)
VOICES = {
    "default": {"description": "CosyVoice2 Default Voice", "language": "en"},
    "chinese": {"description": "Chinese Voice", "language": "zh"},
    "english": {"description": "English Voice", "language": "en"},
    "japanese": {"description": "Japanese Voice", "language": "ja"},
}

DEFAULT_VOICE = "default"
ALL_VOICES = list(VOICES.keys())


class CosyVoice2Engine:
    """
    CosyVoice2 TTS Engine with streaming support
    Ultra-low latency (~150ms first chunk)
    """

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.ready = False
        self.model = None
        self._lock = threading.Lock()

        self._load_model()

    def _download_model_if_needed(self):
        """Download CosyVoice2 model if not present"""
        if self.model_dir.exists() and any(self.model_dir.iterdir()):
            logger.info(f"Model found at {self.model_dir}")
            return True

        logger.info("Downloading CosyVoice2-0.5B model...")
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                "FunAudioLLM/CosyVoice2-0.5B",
                local_dir=str(self.model_dir),
                token=HF_TOKEN
            )
            logger.info("Model download complete")
            return True
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise RuntimeError(f"Could not download CosyVoice2 model: {e}")

    def _load_model(self):
        """Load CosyVoice2 model"""
        logger.info("Loading CosyVoice2-0.5B model...")
        start = time.time()

        try:
            # Download model if needed
            self._download_model_if_needed()

            # Add CosyVoice to path
            cosyvoice_path = Path("/app/CosyVoice")
            if cosyvoice_path.exists():
                sys.path.insert(0, str(cosyvoice_path))
                # Also add third_party matcha
                matcha_path = cosyvoice_path / "third_party" / "Matcha-TTS"
                if matcha_path.exists():
                    sys.path.insert(0, str(matcha_path))

            # Import CosyVoice
            from cosyvoice.cli.cosyvoice import CosyVoice2

            # Load model
            self.model = CosyVoice2(str(self.model_dir), load_jit=False, load_trt=False)

            self.ready = True
            load_time = time.time() - start
            logger.info(f"CosyVoice2-0.5B loaded in {load_time:.1f}s")

        except ImportError as e:
            logger.error(f"Failed to import CosyVoice modules: {e}")
            raise RuntimeError(f"CosyVoice import error: {e}")
        except Exception as e:
            logger.error(f"Failed to load CosyVoice2 model: {e}")
            raise

    def synthesize_streaming(self, text: str, voice: str = "default") -> Iterator[tuple]:
        """
        Synthesize speech with streaming - yields audio chunks as generated
        Ultra-low latency (~150ms first chunk)
        """
        if not self.ready or self.model is None:
            raise RuntimeError("CosyVoice2 engine not ready")

        logger.info(f"[STREAMING] Synthesizing: '{text[:50]}...'")
        start = time.time()
        chunk_count = 0
        total_samples = 0
        first_chunk_logged = False

        try:
            with self._lock:
                # Use streaming inference
                for chunk in self.model.inference_sft(
                    text,
                    "English",  # Speaker/style
                    stream=True,
                    speed=1.0
                ):
                    if chunk is not None and "tts_speech" in chunk:
                        audio_chunk = chunk["tts_speech"].numpy()

                        if len(audio_chunk) > 0:
                            # Normalize
                            if np.max(np.abs(audio_chunk)) > 0:
                                audio_chunk = audio_chunk / np.max(np.abs(audio_chunk)) * 0.95

                            chunk_count += 1
                            total_samples += len(audio_chunk)

                            if not first_chunk_logged:
                                first_chunk_time = (time.time() - start) * 1000
                                logger.info(f"[STREAMING] First chunk in {first_chunk_time:.0f}ms")
                                first_chunk_logged = True

                            yield audio_chunk.astype(np.float32), SAMPLE_RATE

            elapsed = time.time() - start
            audio_duration = total_samples / SAMPLE_RATE if SAMPLE_RATE > 0 else 0
            logger.info(f"[STREAMING] Complete: {chunk_count} chunks, {audio_duration:.2f}s audio in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"[STREAMING] CosyVoice2 synthesis error: {e}")
            raise

    def synthesize(self, text: str, voice: str = "default") -> tuple:
        """
        Synthesize speech (non-streaming)
        Returns numpy array of audio samples at native sample rate
        """
        if not self.ready or self.model is None:
            raise RuntimeError("CosyVoice2 engine not ready")

        logger.info(f"Synthesizing: '{text[:50]}...'")
        start = time.time()

        try:
            with self._lock:
                # Use non-streaming inference
                result = list(self.model.inference_sft(
                    text,
                    "English",
                    stream=False,
                    speed=1.0
                ))

                if not result:
                    raise RuntimeError("No audio generated")

                # Get audio from result
                audio_chunks = []
                for chunk in result:
                    if chunk is not None and "tts_speech" in chunk:
                        audio_chunks.append(chunk["tts_speech"].numpy())

                if not audio_chunks:
                    raise RuntimeError("No audio chunks generated")

                audio = np.concatenate(audio_chunks)

            elapsed = time.time() - start
            duration = len(audio) / SAMPLE_RATE
            logger.info(f"Generated {duration:.2f}s audio in {elapsed:.2f}s (RTF: {elapsed/duration:.2f})")

            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.95

            return audio.astype(np.float32), SAMPLE_RATE

        except Exception as e:
            logger.error(f"CosyVoice2 synthesis error: {e}")
            raise

    def get_sample_rate(self) -> int:
        """Get the native sample rate"""
        return SAMPLE_RATE


def load_cosyvoice_model():
    """Load CosyVoice2 TTS model"""
    global tts_model

    logger.info(f"Loading CosyVoice2 TTS on {DEVICE}...")
    start = time.time()

    try:
        tts_model = CosyVoice2Engine(MODEL_DIR)

        load_time = time.time() - start
        logger.info(f"CosyVoice2 TTS engine ready in {load_time:.1f}s")

        return True
    except Exception as e:
        logger.error(f"Failed to load CosyVoice2 TTS: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - load model on startup"""
    logger.info("=" * 50)
    logger.info("  Sync2 CosyVoice2 TTS Service - Starting")
    logger.info("  Ultra-Low Latency Streaming (~150ms)")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    logger.info(f"  Model: CosyVoice2-0.5B")
    logger.info("=" * 50)

    try:
        load_cosyvoice_model()
        logger.info("CosyVoice2 TTS ready")
    except Exception as e:
        logger.error(f"Failed to initialize CosyVoice2 TTS: {e}")
        logger.warning("Server starting in degraded mode")

    yield

    logger.info("Shutting down CosyVoice2 TTS Service...")


# Create FastAPI app
app = FastAPI(
    title="Sync2 CosyVoice2 TTS Service",
    description="Ultra-low latency streaming TTS using CosyVoice2-0.5B (~150ms)",
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
        "status": "ok" if tts_model is not None and tts_model.ready else "error",
        "model": "CosyVoice2-0.5B",
        "model_loaded": tts_model is not None and tts_model.ready,
        "device": DEVICE,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "ready": tts_model is not None and tts_model.ready,
        "sample_rate": SAMPLE_RATE,
        "output_sample_rate": OUTPUT_SAMPLE_RATE,
        "total_voices": len(ALL_VOICES),
        "streaming": True,
        "latency": "~150ms",
        "version": "1.0.0"
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

    if tts_model is None or not tts_model.ready:
        raise HTTPException(status_code=503, detail="TTS engine not ready")

    start = time.time()

    try:
        # Generate audio
        full_audio, native_sr = tts_model.synthesize(request.text, request.voice)
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


def run_synthesis_in_thread(text: str, voice: str, output_format: str, result_queue: queue.Queue):
    """Run synthesis in a thread and put results in queue"""
    try:
        # Use non-streaming synthesis
        audio, sample_rate = tts_model.synthesize(text, voice)

        # Resample to 8kHz for telephony
        resampled = resample_audio(audio, sample_rate, OUTPUT_SAMPLE_RATE)

        # Convert to requested format
        if output_format == "ulaw":
            audio_bytes = audio_to_ulaw(resampled)
        else:
            audio_bytes = audio_to_pcm16(resampled)

        # Put the complete audio
        result_queue.put(("chunk", audio_bytes))
        result_queue.put(("done", None))
    except Exception as e:
        logger.error(f"[THREAD] Synthesis error: {e}")
        result_queue.put(("error", str(e)))


def run_streaming_synthesis(text: str, voice: str, output_format: str, result_queue: queue.Queue):
    """Run streaming synthesis in a thread"""
    try:
        for audio_chunk, sample_rate in tts_model.synthesize_streaming(text, voice):
            # Resample to 8kHz for telephony
            resampled = resample_audio(audio_chunk, sample_rate, OUTPUT_SAMPLE_RATE)

            # Convert to requested format
            if output_format == "ulaw":
                audio_bytes = audio_to_ulaw(resampled)
            else:
                audio_bytes = audio_to_pcm16(resampled)

            result_queue.put(("chunk", audio_bytes))

        result_queue.put(("done", None))
    except Exception as e:
        logger.error(f"[STREAMING] Synthesis error: {e}")
        result_queue.put(("error", str(e)))


# WebSocket Endpoint for Streaming TTS
@app.websocket("/tts/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming TTS
    Streams audio chunks as they're generated for ultra-low latency
    """
    await websocket.accept()
    logger.info("[WS] Client connected")

    try:
        while True:
            data = await websocket.receive_json()

            text = data.get("text", "")
            voice = data.get("voice", DEFAULT_VOICE)
            output_format = data.get("format", "ulaw")
            stream = data.get("stream", True)  # Enable streaming by default

            if not text:
                await websocket.send_json({"error": "Text is required"})
                continue

            logger.info(f"[WS] Request: '{text[:50]}...' voice={voice} stream={stream}")

            start_time = time.time()
            total_bytes = 0

            try:
                result_queue = queue.Queue()
                loop = asyncio.get_event_loop()

                # Use streaming for lower latency
                if stream:
                    future = loop.run_in_executor(
                        executor,
                        run_streaming_synthesis,
                        text, voice, output_format, result_queue
                    )
                else:
                    future = loop.run_in_executor(
                        executor,
                        run_synthesis_in_thread,
                        text, voice, output_format, result_queue
                    )

                timeout_seconds = 120
                deadline = time.time() + timeout_seconds
                first_chunk_time = None

                while time.time() < deadline:
                    try:
                        result_type, result_data = result_queue.get(block=True, timeout=0.1)

                        if result_type == "chunk":
                            # Send audio chunk
                            await websocket.send_bytes(result_data)
                            total_bytes += len(result_data)

                            if first_chunk_time is None:
                                first_chunk_time = (time.time() - start_time) * 1000
                                logger.info(f"[WS] First chunk in {first_chunk_time:.0f}ms")

                        elif result_type == "done":
                            break

                        elif result_type == "error":
                            raise RuntimeError(result_data)

                    except queue.Empty:
                        if future.done():
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
                    "first_chunk_ms": first_chunk_time,
                    "total_bytes": total_bytes
                })

                logger.info(f"[WS] Complete: {total_bytes} bytes, first chunk {first_chunk_time:.0f}ms, total {duration_ms:.0f}ms")

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
    print("  Sync2 CosyVoice2 TTS Service v1.0")
    print("  Ultra-Low Latency Streaming (~150ms)")
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
