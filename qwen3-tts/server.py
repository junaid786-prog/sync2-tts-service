"""
Qwen3-TTS WebSocket Streaming Server
=====================================
Real-time text-to-speech with 97ms latency
Compatible with existing ARI bridge WebSocket protocol
"""

import asyncio
import json
import logging
import os
import io
import struct
from typing import Optional

import torch
import soundfile as sf
import numpy as np
import websockets
from scipy import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instance
model = None
voice_clone_prompt = None

# Configuration from environment
MODEL_NAME = os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
DEFAULT_VOICE = os.getenv("QWEN_TTS_VOICE", "Vivian")  # Warm female voice
DEFAULT_LANGUAGE = os.getenv("QWEN_TTS_LANGUAGE", "English")
VOICE_CLONE_AUDIO = os.getenv("QWEN_TTS_CLONE_AUDIO", "")  # Path to voice clone reference
VOICE_CLONE_TEXT = os.getenv("QWEN_TTS_CLONE_TEXT", "")  # Transcript of clone audio
WEBSOCKET_PORT = int(os.getenv("WEBSOCKET_PORT", "8765"))

# Available preset voices for CustomVoice model
PRESET_VOICES = {
    "vivian": "Vivian",      # Warm female
    "serena": "Serena",      # Professional female
    "ryan": "Ryan",          # Friendly male
    "dylan": "Dylan",        # Casual male
    "eric": "Eric",          # Professional male
    "aiden": "Aiden",        # Young male
    "uncle_fu": "Uncle_Fu",  # Chinese elder
    "ono_anna": "Ono_Anna",  # Japanese female
    "sohee": "Sohee"         # Korean female
}


def load_model():
    """Load Qwen3-TTS model"""
    global model, voice_clone_prompt

    logger.info(f"Loading Qwen3-TTS model: {MODEL_NAME}")

    try:
        from qwen_tts import Qwen3TTSModel

        # Try to use flash attention if available, otherwise fall back to sdpa
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            logger.info("Using FlashAttention 2 for faster inference")
        except ImportError:
            attn_impl = "sdpa"
            logger.info("FlashAttention not available, using SDPA")

        model = Qwen3TTSModel.from_pretrained(
            MODEL_NAME,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation=attn_impl,
        )

        logger.info(f"Model loaded successfully on {'CUDA' if torch.cuda.is_available() else 'CPU'}")

        # Load voice clone prompt if specified
        if VOICE_CLONE_AUDIO and os.path.exists(VOICE_CLONE_AUDIO):
            logger.info(f"Loading voice clone reference: {VOICE_CLONE_AUDIO}")
            voice_clone_prompt = model.create_voice_clone_prompt(
                ref_audio=VOICE_CLONE_AUDIO,
                ref_text=VOICE_CLONE_TEXT or ""
            )
            logger.info("Voice clone prompt created successfully")

        return True

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def resample_to_8khz_ulaw(audio_data: np.ndarray, original_sr: int) -> bytes:
    """
    Resample audio to 8kHz and convert to μ-law for telephony
    """
    # Resample to 8kHz
    if original_sr != 8000:
        num_samples = int(len(audio_data) * 8000 / original_sr)
        audio_8k = signal.resample(audio_data, num_samples)
    else:
        audio_8k = audio_data

    # Normalize to [-1, 1]
    if audio_8k.max() > 1.0 or audio_8k.min() < -1.0:
        audio_8k = audio_8k / max(abs(audio_8k.max()), abs(audio_8k.min()))

    # Convert to 16-bit PCM
    audio_pcm = (audio_8k * 32767).astype(np.int16)

    # μ-law encoding
    import audioop
    ulaw_data = audioop.lin2ulaw(audio_pcm.tobytes(), 2)

    return ulaw_data


async def generate_speech(text: str, voice: str = None, language: str = None,
                          instruct: str = None) -> tuple:
    """
    Generate speech from text using Qwen3-TTS
    Returns (audio_bytes, sample_rate)
    """
    global model, voice_clone_prompt

    if model is None:
        raise RuntimeError("Model not loaded")

    voice = voice or DEFAULT_VOICE
    language = language or DEFAULT_LANGUAGE

    try:
        # Check if using voice clone
        if voice_clone_prompt is not None:
            logger.info(f"Generating with cloned voice: {text[:50]}...")
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=voice_clone_prompt
            )
        elif "Base" in MODEL_NAME:
            # Base model - use custom voice generation
            logger.info(f"Generating with base model: {text[:50]}...")
            wavs, sr = model.generate(
                text=text,
                language=language,
            )
        else:
            # CustomVoice model - use preset voices
            speaker = PRESET_VOICES.get(voice.lower(), voice)
            logger.info(f"Generating with voice '{speaker}': {text[:50]}...")

            wavs, sr = model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct or ""
            )

        # Convert to numpy array
        audio_np = wavs[0].cpu().numpy() if torch.is_tensor(wavs[0]) else wavs[0]

        return audio_np, sr

    except Exception as e:
        logger.error(f"Speech generation failed: {e}")
        raise


async def handle_websocket(websocket):
    """Handle WebSocket connection for TTS streaming"""
    client_addr = websocket.remote_address
    path = websocket.path if hasattr(websocket, 'path') else '/tts/stream'
    logger.info(f"Client connected: {client_addr}, path: {path}")

    try:
        async for message in websocket:
            try:
                # Parse incoming message
                if isinstance(message, str):
                    data = json.loads(message)
                else:
                    data = json.loads(message.decode('utf-8'))

                text = data.get("text", "")
                voice = data.get("voice", DEFAULT_VOICE)
                language = data.get("language", DEFAULT_LANGUAGE)
                instruct = data.get("instruct", "")
                output_format = data.get("format", "ulaw")  # ulaw for telephony, wav for testing

                # Parameters (for compatibility with Chatterbox protocol)
                temperature = data.get("temperature", 0.7)
                cfg_weight = data.get("cfg_weight", 0.3)
                exaggeration = data.get("exaggeration", 0.5)
                speed = data.get("speed", 1.0)

                if not text:
                    await websocket.send(json.dumps({"error": "No text provided"}))
                    continue

                logger.info(f"TTS request: voice={voice}, lang={language}, text={text[:50]}...")

                # Generate speech
                audio_np, sr = await generate_speech(
                    text=text,
                    voice=voice,
                    language=language,
                    instruct=instruct
                )

                # Apply speed adjustment if needed
                if speed != 1.0 and speed > 0:
                    # Resample to change speed
                    new_length = int(len(audio_np) / speed)
                    audio_np = signal.resample(audio_np, new_length)

                # Convert to output format
                if output_format == "ulaw":
                    # Convert to 8kHz μ-law for telephony
                    audio_bytes = resample_to_8khz_ulaw(audio_np, sr)
                else:
                    # Return as WAV bytes
                    buffer = io.BytesIO()
                    sf.write(buffer, audio_np, sr, format='WAV')
                    audio_bytes = buffer.getvalue()

                # Stream audio in chunks (for low latency)
                chunk_size = 1600  # 200ms chunks at 8kHz
                total_bytes = len(audio_bytes)
                start_time = asyncio.get_event_loop().time()

                logger.info(f"Streaming {total_bytes} bytes in {total_bytes // chunk_size + 1} chunks")

                for i in range(0, total_bytes, chunk_size):
                    chunk = audio_bytes[i:i + chunk_size]
                    await websocket.send(chunk)
                    await asyncio.sleep(0.01)  # Small delay between chunks

                # Calculate duration and send completion message (Chatterbox protocol compatible)
                duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                await websocket.send(json.dumps({
                    "done": True,
                    "duration_ms": duration_ms,
                    "total_bytes": total_bytes
                }))

                logger.info(f"TTS complete: {total_bytes} bytes sent in {duration_ms:.0f}ms")

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                await websocket.send(json.dumps({"error": "Invalid JSON"}))
            except Exception as e:
                logger.error(f"TTS error: {e}")
                await websocket.send(json.dumps({"error": str(e)}))

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {client_addr}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


async def health_check_handler(websocket, path):
    """Handle health check requests"""
    if path == "/health":
        await websocket.send(json.dumps({
            "status": "healthy",
            "model": MODEL_NAME,
            "voice": DEFAULT_VOICE,
            "cuda": torch.cuda.is_available()
        }))
        await websocket.close()
        return True
    return False


async def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("Qwen3-TTS WebSocket Server")
    logger.info("=" * 60)

    # Load model
    if not load_model():
        logger.error("Failed to load model, exiting")
        return

    # Start WebSocket server
    logger.info(f"Starting WebSocket server on port {WEBSOCKET_PORT}")

    async with websockets.serve(
        handle_websocket,
        "0.0.0.0",
        WEBSOCKET_PORT,
        ping_interval=30,
        ping_timeout=10,
        max_size=10 * 1024 * 1024  # 10MB max message size
    ):
        logger.info(f"Server ready at ws://0.0.0.0:{WEBSOCKET_PORT}/tts/stream")
        logger.info(f"Default voice: {DEFAULT_VOICE}")
        logger.info(f"Available voices: {', '.join(PRESET_VOICES.values())}")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
