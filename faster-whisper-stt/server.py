"""
Faster Whisper STT WebSocket Streaming Server
==============================================
Real-time speech-to-text with VAD, barge-in detection, and silence detection
Replaces AWS Transcribe for Sync2.ai platform
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional, Dict, Any
from collections import deque

import numpy as np
import torch
import websockets
from scipy import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instances
whisper_model = None
vad_model = None
vad_utils = None

# Configuration from environment
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")  # tiny, base, small, medium, large-v2, large-v3
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")  # float16, int8, int8_float16
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")  # cuda, cpu
WEBSOCKET_PORT = int(os.getenv("STT_WEBSOCKET_PORT", "8766"))
LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")

# Audio configuration
INPUT_SAMPLE_RATE = 8000   # Kinesis/telephony audio is 8kHz
WHISPER_SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHUNK_DURATION_MS = 100  # Process audio in 100ms chunks

# VAD configuration
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))  # Voice activity threshold
SILENCE_THRESHOLD_MS = int(os.getenv("SILENCE_THRESHOLD_MS", "800"))  # Silence duration to consider end of utterance
BARGE_IN_THRESHOLD_MS = int(os.getenv("BARGE_IN_THRESHOLD_MS", "200"))  # Minimum speech duration for barge-in

# Transcription buffer settings
MIN_AUDIO_LENGTH_MS = 500  # Minimum audio length before attempting transcription
MAX_BUFFER_LENGTH_MS = 30000  # Maximum buffer length (30 seconds)


def load_models():
    """Load Faster Whisper and Silero VAD models"""
    global whisper_model, vad_model, vad_utils

    logger.info(f"Loading Faster Whisper model: {MODEL_SIZE} (compute: {COMPUTE_TYPE})")

    try:
        from faster_whisper import WhisperModel

        # Load Whisper model
        whisper_model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            download_root="/home/ec2-user/.cache/huggingface"
        )
        logger.info(f"Whisper model loaded successfully on {DEVICE.upper()}")

        # Load Silero VAD model
        logger.info("Loading Silero VAD model...")
        vad_model, vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        vad_model = vad_model.to(DEVICE if DEVICE == "cuda" and torch.cuda.is_available() else "cpu")
        logger.info("Silero VAD model loaded successfully")

        return True

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        return False


def resample_audio(audio_data: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resample audio from one sample rate to another"""
    if from_sr == to_sr:
        return audio_data

    # Calculate new length
    new_length = int(len(audio_data) * to_sr / from_sr)
    resampled = signal.resample(audio_data, new_length)
    return resampled.astype(np.float32)


def detect_voice_activity(audio_chunk: np.ndarray, sample_rate: int) -> float:
    """
    Detect voice activity using Silero VAD
    Returns confidence score (0.0 - 1.0)
    """
    global vad_model

    if vad_model is None:
        return 0.5  # Default if VAD not loaded

    try:
        # Ensure audio is at 16kHz for VAD
        if sample_rate != 16000:
            audio_chunk = resample_audio(audio_chunk, sample_rate, 16000)

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_chunk).float()

        # Silero VAD expects audio in range [-1, 1]
        if audio_tensor.abs().max() > 1.0:
            audio_tensor = audio_tensor / audio_tensor.abs().max()

        # Get VAD confidence
        with torch.no_grad():
            confidence = vad_model(audio_tensor, 16000).item()

        return confidence

    except Exception as e:
        logger.error(f"VAD error: {e}")
        return 0.5


def transcribe_audio(audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Transcribe audio using Faster Whisper
    Returns transcript with metadata
    """
    global whisper_model

    if whisper_model is None:
        return {"error": "Model not loaded"}

    try:
        # Ensure audio is at 16kHz for Whisper
        if sample_rate != WHISPER_SAMPLE_RATE:
            audio_data = resample_audio(audio_data, sample_rate, WHISPER_SAMPLE_RATE)

        # Normalize audio
        if audio_data.max() > 1.0 or audio_data.min() < -1.0:
            audio_data = audio_data / max(abs(audio_data.max()), abs(audio_data.min()))

        # Transcribe
        start_time = time.time()
        segments, info = whisper_model.transcribe(
            audio_data,
            language=LANGUAGE,
            beam_size=5,
            best_of=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
            word_timestamps=False,
            condition_on_previous_text=False,
        )

        # Collect segments
        transcript_parts = []
        for segment in segments:
            transcript_parts.append(segment.text.strip())

        transcript = " ".join(transcript_parts).strip()
        elapsed_ms = (time.time() - start_time) * 1000

        return {
            "transcript": transcript,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration_ms": elapsed_ms,
            "audio_duration_ms": len(audio_data) / WHISPER_SAMPLE_RATE * 1000,
        }

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


class StreamingSTTSession:
    """Manages a streaming STT session for a single connection"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.audio_buffer = np.array([], dtype=np.float32)
        self.sample_rate = INPUT_SAMPLE_RATE
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.last_transcript = ""
        self.transcript_count = 0

    def add_audio(self, audio_bytes: bytes, encoding: str = "pcm16") -> Dict[str, Any]:
        """
        Add audio chunk to buffer and process
        Returns events (barge_in, transcript, silence, etc.)
        """
        events = []

        try:
            # Decode audio based on encoding
            if encoding == "pcm16":
                # 16-bit signed PCM
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif encoding == "mulaw" or encoding == "ulaw":
                # μ-law encoded (telephony)
                import audioop
                pcm_bytes = audioop.ulaw2lin(audio_bytes, 2)
                audio_chunk = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                # Assume raw float32
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)

            # Resample to 16kHz for processing
            audio_16k = resample_audio(audio_chunk, self.sample_rate, WHISPER_SAMPLE_RATE)

            # Detect voice activity
            vad_confidence = detect_voice_activity(audio_16k, WHISPER_SAMPLE_RATE)
            current_time = time.time()

            # Check for speech start (barge-in detection)
            if vad_confidence >= VAD_THRESHOLD:
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_start_time = current_time
                    logger.info(f"[{self.session_id}] Speech started (VAD: {vad_confidence:.2f})")

                self.last_speech_time = current_time

                # Barge-in detection: if speech detected for minimum duration
                speech_duration_ms = (current_time - self.speech_start_time) * 1000
                if speech_duration_ms >= BARGE_IN_THRESHOLD_MS and len(events) == 0:
                    events.append({
                        "type": "barge_in",
                        "confidence": vad_confidence,
                        "speech_duration_ms": speech_duration_ms,
                    })

            # Add audio to buffer
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_16k])

            # Limit buffer size
            max_samples = int(MAX_BUFFER_LENGTH_MS * WHISPER_SAMPLE_RATE / 1000)
            if len(self.audio_buffer) > max_samples:
                self.audio_buffer = self.audio_buffer[-max_samples:]

            # Check for silence (end of utterance)
            if self.is_speaking and self.last_speech_time:
                silence_duration_ms = (current_time - self.last_speech_time) * 1000

                if silence_duration_ms >= SILENCE_THRESHOLD_MS:
                    # End of utterance detected
                    self.is_speaking = False

                    # Transcribe the buffered audio
                    buffer_duration_ms = len(self.audio_buffer) / WHISPER_SAMPLE_RATE * 1000

                    if buffer_duration_ms >= MIN_AUDIO_LENGTH_MS:
                        result = transcribe_audio(self.audio_buffer, WHISPER_SAMPLE_RATE)

                        if "transcript" in result and result["transcript"]:
                            self.transcript_count += 1
                            self.last_transcript = result["transcript"]

                            events.append({
                                "type": "transcript",
                                "transcript": result["transcript"],
                                "is_partial": False,
                                "language": result.get("language", LANGUAGE),
                                "transcription_ms": result.get("duration_ms", 0),
                                "audio_duration_ms": result.get("audio_duration_ms", 0),
                            })
                            logger.info(f"[{self.session_id}] Final transcript: {result['transcript'][:50]}...")

                    # Clear buffer after transcription
                    self.audio_buffer = np.array([], dtype=np.float32)
                    self.speech_start_time = None
                    self.last_speech_time = None

                    events.append({
                        "type": "silence",
                        "duration_ms": silence_duration_ms,
                    })

            # Generate partial transcript if buffer is getting large
            buffer_duration_ms = len(self.audio_buffer) / WHISPER_SAMPLE_RATE * 1000
            if self.is_speaking and buffer_duration_ms >= 2000:  # Every 2 seconds during speech
                # Check if we should generate a partial
                if buffer_duration_ms >= MIN_AUDIO_LENGTH_MS:
                    result = transcribe_audio(self.audio_buffer, WHISPER_SAMPLE_RATE)

                    if "transcript" in result and result["transcript"]:
                        # Only send if different from last
                        if result["transcript"] != self.last_transcript:
                            events.append({
                                "type": "transcript",
                                "transcript": result["transcript"],
                                "is_partial": True,
                                "language": result.get("language", LANGUAGE),
                                "transcription_ms": result.get("duration_ms", 0),
                            })
                            logger.debug(f"[{self.session_id}] Partial: {result['transcript'][:30]}...")

            return {"events": events, "vad_confidence": vad_confidence}

        except Exception as e:
            logger.error(f"[{self.session_id}] Error processing audio: {e}")
            import traceback
            traceback.print_exc()
            return {"events": [], "error": str(e)}

    def force_transcribe(self) -> Dict[str, Any]:
        """Force transcription of current buffer (e.g., on disconnect)"""
        if len(self.audio_buffer) < MIN_AUDIO_LENGTH_MS * WHISPER_SAMPLE_RATE / 1000:
            return {"transcript": "", "is_partial": False}

        result = transcribe_audio(self.audio_buffer, WHISPER_SAMPLE_RATE)
        self.audio_buffer = np.array([], dtype=np.float32)

        return {
            "transcript": result.get("transcript", ""),
            "is_partial": False,
            "language": result.get("language", LANGUAGE),
        }

    def reset(self):
        """Reset session state"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None


# Active sessions
sessions: Dict[str, StreamingSTTSession] = {}


async def handle_websocket(websocket):
    """Handle WebSocket connection for STT streaming"""
    client_addr = websocket.remote_address
    session_id = f"{client_addr[0]}:{client_addr[1]}-{int(time.time())}"

    logger.info(f"Client connected: {session_id}")

    # Create session
    session = StreamingSTTSession(session_id)
    sessions[session_id] = session

    try:
        async for message in websocket:
            try:
                if isinstance(message, bytes):
                    # Binary audio data
                    result = session.add_audio(message, encoding="pcm16")

                    # Send events back to client
                    for event in result.get("events", []):
                        await websocket.send(json.dumps(event))

                else:
                    # JSON control message
                    data = json.loads(message)
                    msg_type = data.get("type", "")

                    if msg_type == "config":
                        # Configure session
                        session.sample_rate = data.get("sample_rate", INPUT_SAMPLE_RATE)
                        encoding = data.get("encoding", "pcm16")
                        logger.info(f"[{session_id}] Config: sr={session.sample_rate}, enc={encoding}")
                        await websocket.send(json.dumps({"type": "config_ack", "status": "ok"}))

                    elif msg_type == "audio":
                        # Audio with metadata
                        audio_b64 = data.get("audio", "")
                        encoding = data.get("encoding", "pcm16")

                        import base64
                        audio_bytes = base64.b64decode(audio_b64)
                        result = session.add_audio(audio_bytes, encoding=encoding)

                        for event in result.get("events", []):
                            await websocket.send(json.dumps(event))

                    elif msg_type == "end":
                        # End of stream - force final transcription
                        final = session.force_transcribe()
                        if final.get("transcript"):
                            await websocket.send(json.dumps({
                                "type": "transcript",
                                "transcript": final["transcript"],
                                "is_partial": False,
                                "final": True,
                            }))
                        await websocket.send(json.dumps({"type": "end_ack"}))
                        session.reset()

                    elif msg_type == "reset":
                        # Reset session
                        session.reset()
                        await websocket.send(json.dumps({"type": "reset_ack"}))

                    elif msg_type == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))

            except json.JSONDecodeError as e:
                logger.error(f"[{session_id}] Invalid JSON: {e}")
            except Exception as e:
                logger.error(f"[{session_id}] Error: {e}")
                import traceback
                traceback.print_exc()

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Cleanup session
        if session_id in sessions:
            del sessions[session_id]


async def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("Faster Whisper STT WebSocket Server")
    logger.info("=" * 60)
    logger.info(f"Model: {MODEL_SIZE} ({COMPUTE_TYPE})")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Language: {LANGUAGE}")
    logger.info(f"VAD Threshold: {VAD_THRESHOLD}")
    logger.info(f"Silence Threshold: {SILENCE_THRESHOLD_MS}ms")
    logger.info(f"Barge-in Threshold: {BARGE_IN_THRESHOLD_MS}ms")
    logger.info("=" * 60)

    # Load models
    if not load_models():
        logger.error("Failed to load models, exiting")
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
        logger.info(f"Server ready at ws://0.0.0.0:{WEBSOCKET_PORT}")
        logger.info("Accepting connections...")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
