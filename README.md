# sync2-tts-service

Open-source TTS (Text-to-Speech) service for the Sync2.ai Voice AI platform using Kokoro TTS.

## Features

- Kokoro TTS model integration (open-source, no API costs)
- FastAPI + WebSocket streaming server
- Audio processing pipeline (24kHz → 8kHz for telephony)
- GPU-accelerated inference
- Docker support for easy deployment

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/voices` | GET | List available voices |
| `/tts/synthesize` | POST | Synthesize text to audio |
| `/tts/stream` | WebSocket | Stream audio in real-time |

## Quick Start

### Local Development (Mock Server)

```bash
pip install -r requirements.txt
python mock_server.py
```

### Production (with GPU)

```bash
docker-compose up -d
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

## Testing

```bash
python test_tts.py "Hello, this is a test"
```

## License

MIT
