"""
CSM-1B TTS WebSocket Server
Compatible with existing Kokoro TTS interface for drop-in replacement
"""

import asyncio
import json
import logging
import time
from typing import Optional
import numpy as np

import websockets
from websockets.server import WebSocketServerProtocol

from csm_model import CSMModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instance
model: Optional[CSMModel] = None

# Server configuration
HOST = "0.0.0.0"
PORT = 8765
MAX_CONCURRENT = 10


async def handle_tts_request(websocket: WebSocketServerProtocol, path: str):
    """Handle TTS WebSocket connection"""
    client_id = id(websocket)
    logger.info(f"[{client_id}] New connection from {websocket.remote_address}")

    try:
        async for message in websocket:
            start_time = time.time()

            try:
                # Parse request
                request = json.loads(message)
                text = request.get("text", "")
                voice = request.get("voice", "default")
                output_format = request.get("format", "ulaw")  # ulaw for telephony
                temperature = request.get("temperature", 0.7)
                speed = request.get("speed", 1.0)

                if not text:
                    await websocket.send(json.dumps({"error": "No text provided"}))
                    continue

                logger.info(f"[{client_id}] Synthesizing: '{text[:50]}...' (format={output_format})")

                # Generate audio
                audio = model.synthesize(
                    text=text,
                    temperature=temperature
                )

                # Apply speed adjustment if needed
                if speed != 1.0:
                    audio = adjust_speed(audio, speed, model.sample_rate)

                # Convert to requested format
                if output_format == "ulaw":
                    audio_bytes = model.to_ulaw_bytes(audio)
                elif output_format == "wav":
                    audio_bytes = model.to_wav_bytes(audio)
                else:
                    # Raw PCM
                    audio_bytes = audio.tobytes()

                # Send audio in chunks for streaming
                chunk_size = 4000  # ~500ms of ulaw audio at 8kHz
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i:i + chunk_size]
                    await websocket.send(chunk)
                    await asyncio.sleep(0.01)  # Small delay for flow control

                # Send completion marker
                await websocket.send(b"__END__")

                elapsed = time.time() - start_time
                logger.info(f"[{client_id}] Completed in {elapsed:.2f}s, {len(audio_bytes)} bytes")

            except json.JSONDecodeError:
                logger.error(f"[{client_id}] Invalid JSON received")
                await websocket.send(json.dumps({"error": "Invalid JSON"}))
            except Exception as e:
                logger.error(f"[{client_id}] Synthesis error: {e}")
                await websocket.send(json.dumps({"error": str(e)}))

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"[{client_id}] Connection closed")
    except Exception as e:
        logger.error(f"[{client_id}] Connection error: {e}")


async def handle_streaming_tts(websocket: WebSocketServerProtocol, path: str):
    """Handle streaming TTS for real-time applications"""
    client_id = id(websocket)

    try:
        async for message in websocket:
            request = json.loads(message)
            text = request.get("text", "")

            if not text:
                continue

            logger.info(f"[{client_id}] Streaming: '{text[:50]}...'")

            # Stream audio chunks
            for chunk in model.synthesize_streaming(text, chunk_size_ms=250):
                ulaw_chunk = model.to_ulaw_bytes(chunk)
                await websocket.send(ulaw_chunk)

            await websocket.send(b"__END__")

    except websockets.exceptions.ConnectionClosed:
        pass


def adjust_speed(audio: np.ndarray, speed: float, sample_rate: int) -> np.ndarray:
    """Adjust audio playback speed using resampling"""
    import torchaudio
    import torch

    if speed == 1.0:
        return audio

    # Speed up = downsample then upsample, slow down = opposite
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)

    # Resample to adjust speed
    target_rate = int(sample_rate * speed)
    resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
    audio_tensor = resampler(audio_tensor)

    # Resample back to original rate
    resampler_back = torchaudio.transforms.Resample(target_rate, sample_rate)
    audio_tensor = resampler_back(audio_tensor)

    return audio_tensor.squeeze(0).numpy()


async def health_check(websocket: WebSocketServerProtocol, path: str):
    """Health check endpoint"""
    await websocket.send(json.dumps({
        "status": "healthy",
        "model": "csm-1b",
        "device": model.device if model else "not loaded"
    }))


async def router(websocket: WebSocketServerProtocol, path: str):
    """Route requests to appropriate handler"""
    if path == "/health":
        await health_check(websocket, path)
    elif path == "/tts/stream":
        await handle_tts_request(websocket, path)
    elif path == "/tts/realtime":
        await handle_streaming_tts(websocket, path)
    else:
        # Default to standard TTS
        await handle_tts_request(websocket, path)


async def main():
    """Start the TTS server"""
    global model

    logger.info("Loading CSM-1B model...")
    model = CSMModel()
    logger.info(f"Model loaded on {model.device}")

    # Warmup
    logger.info("Warming up model...")
    _ = model.synthesize("Hello, warming up the model.")
    logger.info("Warmup complete")

    # Start server
    logger.info(f"Starting CSM TTS server on ws://{HOST}:{PORT}")

    async with websockets.serve(
        router,
        HOST,
        PORT,
        max_size=10 * 1024 * 1024,  # 10MB max message
        ping_interval=30,
        ping_timeout=10
    ):
        logger.info("Server ready!")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
