# CSM-1B TTS Server

WebSocket server for Sesame's CSM-1B (Conversational Speech Model) text-to-speech.

## Requirements

- AWS g5.xlarge (A10G 24GB GPU) or similar
- Python 3.10+
- CUDA 12.1+

## Quick Start

### 1. Install Dependencies

```bash
# Install PyTorch with CUDA
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Login to HuggingFace (required for model access)
huggingface-cli login
```

### 2. Run Server

```bash
python server.py
```

Server starts on `ws://0.0.0.0:8765`

### 3. Test

```bash
python test_client.py "Hello, this is a test."
```

## API

### WebSocket Endpoint: `/tts/stream`

**Request (JSON):**
```json
{
  "text": "Hello, world!",
  "format": "ulaw",
  "temperature": 0.9,
  "speaker_id": 0
}
```

**Response:**
- Binary audio chunks
- `b"__END__"` marker when complete

### Formats

- `ulaw` - 8kHz u-law for telephony (default)
- `wav` - 24kHz WAV PCM

## Integration with ARI Bridge

Update your `.env`:
```
USE_CSM_TTS=true
CSM_TTS_URL=ws://localhost:8765/tts/stream
```

## Systemd Service

```bash
sudo cp csm-tts.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable csm-tts
sudo systemctl start csm-tts
```

## Performance

- ~200-400ms latency for short sentences
- RTF < 1.0 (faster than real-time)
- ~4GB VRAM usage

## References

- [CSM-1B on HuggingFace](https://huggingface.co/sesame/csm-1b)
- [CSM GitHub](https://github.com/SesameAILabs/csm)
