# Moshi Speech-to-Speech Server

Real-time speech-to-speech conversational system with ultra-low latency (~200ms).

## Features

- **Real-time duplex communication** - Simultaneous speaking and listening
- **Ultra-low latency** - ~200ms end-to-end response time
- **Emotional prosody** - Natural tone and empathy detection
- **Native barge-in** - Instant interruption handling
- **WebSocket API** - Simple integration

## Quick Start

### Using Docker (Recommended)

```bash
# From sync2-tts-service directory
docker-compose up moshi-s2s -d --build

# Verify deployment
curl http://localhost:8770/

# Check logs
docker logs -f sync2-moshi-service
```

### Manual Installation

```bash
cd moshi_server

# Install dependencies
pip install -r requirements.txt

# Run server
python moshi_server.py --host 0.0.0.0 --port 8770
```

## Configuration

Environment variables:
- `PYTHONUNBUFFERED=1` - Real-time logging
- `NVIDIA_VISIBLE_DEVICES=all` - GPU access

## API Endpoints

### HTTP Endpoints

- `GET /` - Health check
- `GET /stats` - Server statistics

### WebSocket Endpoint

- `WS /moshi` - Main S2S endpoint

## WebSocket Protocol

### 1. Connect and Configure

```json
{
  "type": "config",
  "session_id": "unique-session-id",
  "sample_rate": 16000,
  "system_prompt": "You are a helpful assistant...",
  "voice": "default"
}
```

### 2. Send Audio

Send binary PCM audio chunks (16-bit, mono, 16kHz)

### 3. Receive Responses

- **Binary**: Audio responses (16-bit PCM, 16kHz)
- **JSON**: Control messages

```json
{"type": "ready"}
{"type": "speech_start"}
{"type": "speech_end"}
{"type": "transcript", "speaker": "agent", "text": "..."}
{"type": "error", "error": "..."}
```

## Integration with ARI Bridge

Update `.env` in Sync2ARIend:

```bash
VOICE_AI_MODE=moshi
MOSHI_SERVER_URL=ws://44.216.12.223:8770/moshi
MOSHI_SAMPLE_RATE=16000
MOSHI_VOICE=default
MOSHI_SYSTEM_PROMPT_ENABLED=true
MOSHI_TIMEOUT=30000
```

## Resource Requirements

- **CPU**: 4+ cores (8+ recommended)
- **RAM**: 4-8GB
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **Network**: Low-latency connection for real-time performance

## Troubleshooting

### Connection refused

```bash
# Check if service is running
docker ps | grep moshi

# Start service
docker-compose up moshi-s2s -d
```

### High latency

- Check network latency between ARI bridge and server
- Verify GPU is being used (not CPU fallback)
- Monitor resource usage: `docker stats sync2-moshi-service`

## Performance

- **Latency**: ~200ms end-to-end
- **Throughput**: 10+ concurrent sessions per GPU
- **Audio Quality**: 16kHz, 16-bit PCM

## Documentation

See parent directory documentation:
- [MOSHI_INTEGRATION.md](../../MOSHI_INTEGRATION.md)
- [MIGRATION_SUMMARY.md](../../MIGRATION_SUMMARY.md)
- [QUICKSTART_MOSHI.md](../../QUICKSTART_MOSHI.md)
