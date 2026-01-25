"""
Kinesis Audio Consumer for Faster Whisper STT
==============================================
Receives audio from Kinesis Video Stream and forwards to STT WebSocket server
Sends transcripts to SQS (same format as AWS Transcribe)
"""

import asyncio
import json
import logging
import os
import time
import base64
from typing import Optional, Dict, Any

import boto3
import websockets
from botocore.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SQS_QUEUE_URL = os.getenv("TRANSCRIPT_QUEUE_URL_GROUP", "")
KINESIS_STREAM_NAME = os.getenv("KINESIS_STREAM_NAME", "")

# STT Server Configuration
STT_SERVER_URL = os.getenv("STT_SERVER_URL", "ws://localhost:8766")

# Initialize AWS clients
boto_config = Config(
    region_name=AWS_REGION,
    retries={'max_attempts': 3, 'mode': 'adaptive'}
)
sqs_client = boto3.client('sqs', config=boto_config)
kinesis_client = boto3.client('kinesisvideo', config=boto_config)


class KinesisSTTBridge:
    """Bridges Kinesis audio stream to Faster Whisper STT"""

    def __init__(self, call_session_id: str, channel_id: str, from_e164: str = None, to_e164: str = None):
        self.call_session_id = call_session_id
        self.channel_id = channel_id
        self.from_e164 = from_e164
        self.to_e164 = to_e164
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_running = False
        self.last_transcript_time = 0

    async def connect_stt(self):
        """Connect to STT WebSocket server"""
        try:
            self.ws = await websockets.connect(
                STT_SERVER_URL,
                ping_interval=20,
                ping_timeout=10,
            )
            logger.info(f"[{self.call_session_id}] Connected to STT server")

            # Configure the session
            await self.ws.send(json.dumps({
                "type": "config",
                "sample_rate": 8000,
                "encoding": "pcm16",
            }))

            return True
        except Exception as e:
            logger.error(f"[{self.call_session_id}] Failed to connect to STT: {e}")
            return False

    async def send_audio(self, audio_bytes: bytes):
        """Send audio chunk to STT server"""
        if self.ws and self.ws.open:
            await self.ws.send(audio_bytes)

    async def receive_events(self):
        """Receive and process events from STT server"""
        if not self.ws:
            return

        try:
            while self.is_running:
                try:
                    message = await asyncio.wait_for(self.ws.recv(), timeout=0.1)
                    event = json.loads(message)

                    event_type = event.get("type", "")

                    if event_type == "transcript":
                        await self.handle_transcript(event)
                    elif event_type == "barge_in":
                        await self.handle_barge_in(event)
                    elif event_type == "silence":
                        logger.debug(f"[{self.call_session_id}] Silence detected: {event.get('duration_ms')}ms")

                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.warning(f"[{self.call_session_id}] STT connection closed")
                    break

        except Exception as e:
            logger.error(f"[{self.call_session_id}] Error receiving events: {e}")

    async def handle_transcript(self, event: Dict[str, Any]):
        """Handle transcript event - send to SQS"""
        transcript = event.get("transcript", "").strip()
        is_partial = event.get("is_partial", False)

        if not transcript:
            return

        # Rate limit transcripts (avoid sending too many partials)
        current_time = time.time()
        if is_partial and (current_time - self.last_transcript_time) < 0.5:
            return
        self.last_transcript_time = current_time

        logger.info(f"[{self.call_session_id}] {'Partial' if is_partial else 'Final'}: {transcript[:50]}...")

        # Send to SQS in same format as AWS Transcribe
        try:
            message_body = {
                "transcript": transcript,
                "isPartial": is_partial,
                "asteriskChannelId": self.channel_id,
                "fromE164": self.from_e164,
                "toE164": self.to_e164,
                "callSessionId": self.call_session_id,
                "source": "faster-whisper",  # Identify source
                "timestamp": int(current_time * 1000),
            }

            if SQS_QUEUE_URL:
                response = sqs_client.send_message(
                    QueueUrl=SQS_QUEUE_URL,
                    MessageBody=json.dumps(message_body),
                    MessageGroupId=f"call-{self.call_session_id}",
                    MessageDeduplicationId=f"{self.call_session_id}-{int(current_time * 1000)}-{hash(transcript) % 10000}",
                )
                logger.debug(f"[{self.call_session_id}] Sent to SQS: {response['MessageId']}")

        except Exception as e:
            logger.error(f"[{self.call_session_id}] Failed to send to SQS: {e}")

    async def handle_barge_in(self, event: Dict[str, Any]):
        """Handle barge-in event - notify ARI bridge to stop TTS"""
        logger.info(f"[{self.call_session_id}] Barge-in detected! Confidence: {event.get('confidence', 0):.2f}")

        # Send barge-in notification to SQS
        try:
            message_body = {
                "type": "barge_in",
                "asteriskChannelId": self.channel_id,
                "callSessionId": self.call_session_id,
                "confidence": event.get("confidence", 0),
                "timestamp": int(time.time() * 1000),
            }

            if SQS_QUEUE_URL:
                sqs_client.send_message(
                    QueueUrl=SQS_QUEUE_URL,
                    MessageBody=json.dumps(message_body),
                    MessageGroupId=f"call-{self.call_session_id}",
                    MessageDeduplicationId=f"bargein-{self.call_session_id}-{int(time.time() * 1000)}",
                )

        except Exception as e:
            logger.error(f"[{self.call_session_id}] Failed to send barge-in to SQS: {e}")

    async def end_session(self):
        """End the STT session"""
        if self.ws and self.ws.open:
            await self.ws.send(json.dumps({"type": "end"}))
            try:
                # Wait for final transcript
                message = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
                event = json.loads(message)
                if event.get("type") == "transcript":
                    await self.handle_transcript(event)
            except:
                pass
            await self.ws.close()

        self.is_running = False

    async def close(self):
        """Close connection"""
        self.is_running = False
        if self.ws:
            await self.ws.close()


async def process_kinesis_record(record: Dict, bridge: KinesisSTTBridge):
    """Process a single Kinesis record containing audio data"""
    try:
        # Decode the base64 audio data
        data = base64.b64decode(record['kinesis']['data'])

        # Check if it's audio data or metadata
        try:
            # Try to parse as JSON (metadata)
            json_data = json.loads(data.decode('utf-8'))

            # Handle different message types
            if 'streamingStatus' in json_data:
                status = json_data['streamingStatus']
                if status == 'ENDED':
                    await bridge.end_session()
                    return False  # Signal to stop processing
            return True

        except (json.JSONDecodeError, UnicodeDecodeError):
            # It's binary audio data
            await bridge.send_audio(data)
            return True

    except Exception as e:
        logger.error(f"Error processing Kinesis record: {e}")
        return True


# For Lambda handler
def lambda_handler(event, context):
    """AWS Lambda handler for Kinesis trigger"""
    import asyncio

    async def process():
        bridges = {}

        for record in event.get('Records', []):
            try:
                # Extract call metadata from record
                partition_key = record['kinesis']['partitionKey']
                sequence_number = record['kinesis']['sequenceNumber']

                # Get or create bridge for this call
                if partition_key not in bridges:
                    # Parse partition key for call info (format: callId-channelId)
                    parts = partition_key.split('-')
                    call_session_id = parts[0] if parts else partition_key
                    channel_id = parts[1] if len(parts) > 1 else ""

                    bridge = KinesisSTTBridge(
                        call_session_id=call_session_id,
                        channel_id=channel_id,
                    )
                    if await bridge.connect_stt():
                        bridges[partition_key] = bridge
                        bridge.is_running = True
                    else:
                        continue

                bridge = bridges.get(partition_key)
                if bridge:
                    should_continue = await process_kinesis_record(record, bridge)
                    if not should_continue:
                        await bridge.close()
                        del bridges[partition_key]

            except Exception as e:
                logger.error(f"Error processing record: {e}")

        # Cleanup remaining bridges
        for bridge in bridges.values():
            await bridge.close()

    asyncio.run(process())

    return {'statusCode': 200, 'body': 'Processed'}


if __name__ == "__main__":
    # Test mode - connect to STT server and send test audio
    import sys

    async def test():
        bridge = KinesisSTTBridge(
            call_session_id="test-session",
            channel_id="test-channel",
        )

        if await bridge.connect_stt():
            bridge.is_running = True

            # Start receiving events in background
            receive_task = asyncio.create_task(bridge.receive_events())

            # Send test audio (silence + speech simulation)
            print("Sending test audio... (press Ctrl+C to stop)")

            try:
                import numpy as np

                # Generate 8kHz test tone
                duration = 0.1  # 100ms chunks
                sample_rate = 8000
                t = np.linspace(0, duration, int(sample_rate * duration), False)

                for i in range(100):  # 10 seconds
                    # Alternate between silence and tone
                    if i % 20 < 10:
                        # Silence
                        audio = np.zeros_like(t)
                    else:
                        # 440Hz tone (simulates speech)
                        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

                    audio_bytes = (audio * 32767).astype(np.int16).tobytes()
                    await bridge.send_audio(audio_bytes)
                    await asyncio.sleep(duration)

            except KeyboardInterrupt:
                pass

            receive_task.cancel()
            await bridge.end_session()

    asyncio.run(test())
