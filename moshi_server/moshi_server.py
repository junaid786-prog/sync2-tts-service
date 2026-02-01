"""
Moshi Speech-to-Speech Server
==============================
WebSocket server that provides real-time speech-to-speech conversation using Moshi model.

This server:
1. Accepts incoming audio via WebSocket (16kHz PCM)
2. Processes speech with Moshi S2S model
3. Returns audio output in real-time (16kHz PCM)
4. Provides transcript and emotion detection

Installation:
pip install fastapi uvicorn websockets torch torchaudio transformers

Usage:
python moshi_server.py --host 0.0.0.0 --port 8770
"""

import asyncio
import json
import logging
import argparse
from typing import Optional, Dict
from datetime import datetime

import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Moshi S2S Server", version="1.0.0")


class MoshiSession:
    """
    Manages a single Moshi conversation session
    """

    def __init__(self, session_id: str, config: dict):
        self.session_id = session_id
        self.config = config
        self.sample_rate = config.get('sample_rate', 16000)
        self.voice = config.get('voice', 'default')
        self.system_prompt = config.get('system_prompt', '')

        # Audio buffers
        self.input_buffer = bytearray()
        self.output_buffer = bytearray()

        # State
        self.is_listening = True
        self.is_speaking = False
        self.conversation_history = []

        # Statistics
        self.start_time = datetime.now()
        self.audio_bytes_received = 0
        self.audio_bytes_sent = 0
        self.response_count = 0

        logger.info(f"[{self.session_id}] Session created with sample_rate={self.sample_rate}")

    async def process_audio(self, audio_data: bytes) -> Optional[bytes]:
        """
        Process incoming audio and return response audio

        Args:
            audio_data: Raw PCM audio bytes (16-bit, mono)

        Returns:
            Response audio bytes or None
        """
        self.audio_bytes_received += len(audio_data)
        self.input_buffer.extend(audio_data)

        # TODO: Integrate actual Moshi model here
        # For now, this is a placeholder that would:
        # 1. Convert audio to model input format
        # 2. Run Moshi inference
        # 3. Generate response audio
        # 4. Return audio bytes

        # Placeholder: Echo back silence (in production, this would be Moshi's response)
        # In real implementation, you would:
        # - Use moshi.generate(audio_tensor, text_prompt=self.system_prompt)
        # - Process streaming output
        # - Return generated audio

        return None

    async def send_transcript(self, speaker: str, text: str, is_final: bool = True, emotion: Optional[str] = None):
        """
        Send transcript update
        """
        return {
            'type': 'transcript',
            'speaker': speaker,
            'text': text,
            'is_final': is_final,
            'emotion': emotion
        }

    def get_stats(self) -> dict:
        """Get session statistics"""
        duration = (datetime.now() - self.start_time).total_seconds()
        return {
            'session_id': self.session_id,
            'duration_seconds': duration,
            'audio_bytes_received': self.audio_bytes_received,
            'audio_bytes_sent': self.audio_bytes_sent,
            'response_count': self.response_count
        }


# Active sessions
sessions: Dict[str, MoshiSession] = {}


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Moshi S2S Server",
        "status": "running",
        "active_sessions": len(sessions)
    }


@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    stats = {
        "active_sessions": len(sessions),
        "sessions": [session.get_stats() for session in sessions.values()]
    }
    return JSONResponse(stats)


@app.websocket("/moshi")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for Moshi S2S

    Protocol:
    1. Client connects and sends config message:
       {"type": "config", "session_id": "...", "sample_rate": 16000, "system_prompt": "..."}

    2. Client sends binary audio chunks (PCM 16-bit mono)

    3. Server sends:
       - Binary audio responses
       - JSON messages: {"type": "transcript", "text": "...", "speaker": "agent/user"}
       - JSON messages: {"type": "speech_start"}, {"type": "speech_end"}
    """
    await websocket.accept()
    session: Optional[MoshiSession] = None
    session_id = None

    try:
        logger.info("New WebSocket connection established")

        # Wait for configuration message
        config_data = await websocket.receive_text()
        config = json.loads(config_data)

        if config.get('type') != 'config':
            await websocket.send_json({"type": "error", "error": "First message must be config"})
            await websocket.close()
            return

        session_id = config.get('session_id', f"session_{datetime.now().timestamp()}")
        session = MoshiSession(session_id, config)
        sessions[session_id] = session

        logger.info(f"[{session_id}] Session configured: {config}")

        # Send ready signal
        await websocket.send_json({"type": "ready", "session_id": session_id})

        # Main processing loop
        while True:
            # Receive audio or control messages
            try:
                # Try to receive binary data (audio)
                data = await websocket.receive_bytes()

                # Process audio with Moshi
                response_audio = await session.process_audio(data)

                if response_audio:
                    # Send speech start notification
                    if not session.is_speaking:
                        session.is_speaking = True
                        await websocket.send_json({"type": "speech_start"})

                    # Send audio response
                    await websocket.send_bytes(response_audio)
                    session.audio_bytes_sent += len(response_audio)

                    # Send speech end notification
                    session.is_speaking = False
                    session.response_count += 1
                    await websocket.send_json({"type": "speech_end"})

            except WebSocketDisconnect:
                break
            except Exception as e:
                # Try to receive text (control messages)
                try:
                    text_data = await websocket.receive_text()
                    message = json.loads(text_data)

                    if message.get('type') == 'barge_in':
                        # Handle barge-in request
                        logger.info(f"[{session_id}] Barge-in requested")
                        session.is_speaking = False
                        session.output_buffer.clear()

                    elif message.get('type') == 'text':
                        # Handle text injection
                        logger.info(f"[{session_id}] Text injection: {message.get('text')}")
                        # In production, this would update the conversation context

                except Exception as inner_e:
                    logger.error(f"[{session_id}] Error processing message: {inner_e}")
                    break

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] Client disconnected")
    except Exception as e:
        logger.error(f"[{session_id}] WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except:
            pass
    finally:
        # Cleanup
        if session_id and session_id in sessions:
            stats = sessions[session_id].get_stats()
            logger.info(f"[{session_id}] Session ended. Stats: {stats}")
            del sessions[session_id]

        try:
            await websocket.close()
        except:
            pass


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Moshi S2S Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8770, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger.info("=" * 70)
    logger.info("🚀 Moshi Speech-to-Speech Server")
    logger.info("=" * 70)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"WebSocket endpoint: ws://{args.host}:{args.port}/moshi")
    logger.info("=" * 70)

    uvicorn.run(
        "moshi_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
