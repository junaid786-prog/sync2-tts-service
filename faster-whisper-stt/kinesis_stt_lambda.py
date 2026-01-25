"""
Faster Whisper STT Lambda - Kinesis Video Stream Consumer
==========================================================
Replaces AWS Transcribe with GPU-backed Faster Whisper STT.

This Lambda is triggered by Kinesis Video Stream and:
1. Extracts audio frames from the stream
2. Sends audio to Faster Whisper STT WebSocket server
3. Forwards transcripts to the same SQS FIFO queue as AWS Transcribe

Environment Variables:
  - STT_SERVER_URL: WebSocket URL of Faster Whisper server (e.g., ws://44.216.12.223:8766)
  - TRANSCRIPT_QUEUE_URL_GROUP: SQS FIFO queue URL for transcripts
  - PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD: PostgreSQL connection for session lookups

Deployment:
  - Package with websockets, psycopg2-binary, boto3
  - Set Lambda timeout to 60s+
  - Memory: 512MB minimum
  - VPC: Must be in same VPC as STT server or have public access
"""

import asyncio
import json
import logging
import os
import time
import base64
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import boto3
import psycopg2
from psycopg2.extras import RealDictCursor
from botocore.config import Config

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
TRANSCRIPT_QUEUE_URL = os.getenv("TRANSCRIPT_QUEUE_URL_GROUP", "")

# STT Server Configuration
STT_SERVER_URL = os.getenv("STT_SERVER_URL", "ws://44.216.12.223:8766")

# PostgreSQL Configuration (for session lookups)
PG_CONFIG = {
    "host": os.getenv("PGHOST", ""),
    "port": int(os.getenv("PGPORT", "5432")),
    "database": os.getenv("PGDATABASE", "pharmasync"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", ""),
}

# Initialize AWS clients
boto_config = Config(
    region_name=AWS_REGION,
    retries={'max_attempts': 3, 'mode': 'adaptive'}
)
sqs_client = boto3.client('sqs', config=boto_config)

# Session cache (mip_id -> call_session_id)
session_cache: Dict[str, str] = {}

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)


def get_pg_connection():
    """Get PostgreSQL connection"""
    return psycopg2.connect(
        host=PG_CONFIG["host"],
        port=PG_CONFIG["port"],
        database=PG_CONFIG["database"],
        user=PG_CONFIG["user"],
        password=PG_CONFIG["password"],
        sslmode="require"
    )


def find_call_session_id(mip_id: str, from_e164: str = None, to_e164: str = None) -> Optional[str]:
    """Find call_session_id by MIP ID or phone numbers"""
    # Check cache first
    if mip_id in session_cache:
        return session_cache[mip_id]

    try:
        conn = get_pg_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Try direct lookup from call_mip_map
            cur.execute("""
                SELECT cs.id
                FROM call_sessions cs
                JOIN call_mip_map m ON cs.id = m.call_session_id
                WHERE m.mip_id = %s
                ORDER BY cs.started_at DESC
                LIMIT 1
            """, (mip_id,))

            row = cur.fetchone()
            if row:
                session_cache[mip_id] = row['id']
                return row['id']

            # Try phone number matching
            if from_e164 and to_e164:
                cur.execute("""
                    SELECT id FROM call_sessions
                    WHERE outcome = 'IN_PROGRESS'
                      AND from_e164 = %s
                      AND to_e164 = %s
                    ORDER BY started_at DESC
                    LIMIT 1
                """, (from_e164, to_e164))

                row = cur.fetchone()
                if row:
                    # Auto-link this MIP
                    cur.execute("""
                        INSERT INTO call_mip_map (call_session_id, mip_id)
                        VALUES (%s, %s)
                        ON CONFLICT (mip_id) DO NOTHING
                    """, (row['id'], mip_id))
                    conn.commit()

                    session_cache[mip_id] = row['id']
                    return row['id']

            # Fallback: latest IN_PROGRESS call
            cur.execute("""
                SELECT id FROM call_sessions
                WHERE outcome = 'IN_PROGRESS'
                ORDER BY started_at DESC
                LIMIT 1
            """)

            row = cur.fetchone()
            if row:
                logger.warning(f"Using fallback session for MIP {mip_id}")
                session_cache[mip_id] = row['id']
                return row['id']

    except Exception as e:
        logger.error(f"DB lookup error: {e}")
    finally:
        if conn:
            conn.close()

    return None


def get_channel_for_session(call_session_id: str) -> str:
    """Get Asterisk channel ID for a call session"""
    try:
        conn = get_pg_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT asterisk_channel_id
                FROM call_sessions
                WHERE id = %s
                LIMIT 1
            """, (call_session_id,))

            row = cur.fetchone()
            return row['asterisk_channel_id'] if row else "unknown"
    except Exception as e:
        logger.error(f"Channel lookup error: {e}")
        return "unknown"
    finally:
        if conn:
            conn.close()


def send_transcript_to_sqs(
    transcript: str,
    is_partial: bool,
    call_session_id: str,
    channel_id: str,
    from_e164: str = None,
    to_e164: str = None
):
    """Send transcript to SQS FIFO queue (same format as AWS Transcribe)"""
    if not TRANSCRIPT_QUEUE_URL:
        logger.error("TRANSCRIPT_QUEUE_URL_GROUP not set!")
        return

    try:
        message_body = {
            "transcript": transcript,
            "isPartial": is_partial,
            "asteriskChannelId": channel_id,
            "fromE164": from_e164,
            "toE164": to_e164,
            "callSessionId": call_session_id,
            "source": "faster-whisper",  # Identify source
        }

        message_group_id = f"call-{call_session_id}"
        dedup_id = f"{call_session_id}-{int(time.time() * 1000)}-{hash(transcript) % 10000}"

        response = sqs_client.send_message(
            QueueUrl=TRANSCRIPT_QUEUE_URL,
            MessageBody=json.dumps(message_body),
            MessageGroupId=message_group_id,
            MessageDeduplicationId=dedup_id,
        )

        logger.info(f"Sent {'partial' if is_partial else 'final'} transcript to SQS: {response['MessageId']}")

    except Exception as e:
        logger.error(f"SQS send error: {e}")


def send_barge_in_to_sqs(call_session_id: str, channel_id: str, confidence: float):
    """Send barge-in notification to SQS"""
    if not TRANSCRIPT_QUEUE_URL:
        return

    try:
        message_body = {
            "type": "barge_in",
            "asteriskChannelId": channel_id,
            "callSessionId": call_session_id,
            "confidence": confidence,
            "timestamp": int(time.time() * 1000),
        }

        sqs_client.send_message(
            QueueUrl=TRANSCRIPT_QUEUE_URL,
            MessageBody=json.dumps(message_body),
            MessageGroupId=f"call-{call_session_id}",
            MessageDeduplicationId=f"bargein-{call_session_id}-{int(time.time() * 1000)}",
        )

        logger.info(f"Sent barge-in notification to SQS")

    except Exception as e:
        logger.error(f"Failed to send barge-in: {e}")


async def process_with_stt_server(
    audio_chunks: list,
    call_session_id: str,
    channel_id: str,
    from_e164: str = None,
    to_e164: str = None
):
    """Connect to STT server and process audio"""
    import websockets

    try:
        async with websockets.connect(
            STT_SERVER_URL,
            ping_interval=20,
            ping_timeout=10,
        ) as ws:
            logger.info(f"Connected to STT server: {STT_SERVER_URL}")

            # Configure session
            await ws.send(json.dumps({
                "type": "config",
                "sample_rate": 8000,
                "encoding": "pcm16",
            }))

            # Send all audio chunks
            for chunk in audio_chunks:
                await ws.send(chunk)

            # Signal end of audio
            await ws.send(json.dumps({"type": "end"}))

            # Receive events until connection closes
            last_transcript_time = 0

            async for message in ws:
                try:
                    event = json.loads(message)
                    event_type = event.get("type", "")

                    if event_type == "transcript":
                        transcript = event.get("transcript", "").strip()
                        is_partial = event.get("is_partial", False)

                        if not transcript:
                            continue

                        # Rate limit partials
                        current_time = time.time()
                        if is_partial and (current_time - last_transcript_time) < 0.5:
                            continue
                        last_transcript_time = current_time

                        logger.info(f"{'Partial' if is_partial else 'Final'}: {transcript[:50]}...")

                        send_transcript_to_sqs(
                            transcript=transcript,
                            is_partial=is_partial,
                            call_session_id=call_session_id,
                            channel_id=channel_id,
                            from_e164=from_e164,
                            to_e164=to_e164
                        )

                    elif event_type == "barge_in":
                        confidence = event.get("confidence", 0)
                        logger.info(f"Barge-in detected! Confidence: {confidence:.2f}")
                        send_barge_in_to_sqs(call_session_id, channel_id, confidence)

                    elif event_type == "silence":
                        logger.debug(f"Silence: {event.get('duration_ms')}ms")

                except json.JSONDecodeError:
                    continue

    except Exception as e:
        logger.error(f"STT server error: {e}")


def lambda_handler(event, context):
    """
    AWS Lambda handler for Kinesis Video Stream trigger.

    Processes audio from Kinesis and sends to Faster Whisper STT.
    """
    logger.info(f"Processing {len(event.get('Records', []))} Kinesis records")

    # Group records by partition key (call session)
    sessions: Dict[str, Dict] = {}

    for record in event.get('Records', []):
        try:
            partition_key = record['kinesis']['partitionKey']

            if partition_key not in sessions:
                sessions[partition_key] = {
                    'audio_chunks': [],
                    'mip_id': None,
                    'from_e164': None,
                    'to_e164': None,
                    'ended': False,
                }

            session = sessions[partition_key]

            # Decode the data
            data = base64.b64decode(record['kinesis']['data'])

            # Check if it's JSON metadata or binary audio
            try:
                json_data = json.loads(data.decode('utf-8'))

                # Extract metadata
                if 'mediaInsightsPipelineId' in json_data:
                    session['mip_id'] = json_data['mediaInsightsPipelineId']
                if 'MediaInsightsPipelineId' in json_data:
                    session['mip_id'] = json_data['MediaInsightsPipelineId']

                # Check for phone metadata
                if 'EventMetadata' in json_data:
                    meta = json_data['EventMetadata']
                    session['from_e164'] = meta.get('fromNumber')
                    session['to_e164'] = meta.get('toNumber')

                # Check for stream end
                if json_data.get('streamingStatus') == 'ENDED':
                    session['ended'] = True
                    logger.info(f"Stream ended for {partition_key}")

            except (json.JSONDecodeError, UnicodeDecodeError):
                # Binary audio data
                session['audio_chunks'].append(data)

        except Exception as e:
            logger.error(f"Error processing record: {e}")

    # Process each session
    for partition_key, session in sessions.items():
        if not session['audio_chunks']:
            continue

        mip_id = session['mip_id'] or partition_key

        # Find call session ID
        call_session_id = find_call_session_id(
            mip_id,
            session['from_e164'],
            session['to_e164']
        )

        if not call_session_id:
            logger.warning(f"No call session found for {mip_id}")
            continue

        # Get channel ID
        channel_id = get_channel_for_session(call_session_id)

        logger.info(f"Processing {len(session['audio_chunks'])} chunks for session {call_session_id}")

        # Process with STT server
        try:
            asyncio.run(process_with_stt_server(
                audio_chunks=session['audio_chunks'],
                call_session_id=call_session_id,
                channel_id=channel_id,
                from_e164=session['from_e164'],
                to_e164=session['to_e164']
            ))
        except Exception as e:
            logger.error(f"STT processing error: {e}")

    return {'statusCode': 200, 'body': 'Processed'}
