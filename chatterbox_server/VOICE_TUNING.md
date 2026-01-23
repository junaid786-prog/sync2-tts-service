# Chatterbox Voice Tuning Guide

This guide explains how to adjust Chatterbox TTS parameters to create more realistic, human-like voices.

## Quick Start

The Chatterbox server now supports 4 synthesis parameters that dramatically affect voice quality:

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `temperature` | 0.05-5.0 | 0.7 | Controls randomness and natural variation |
| `cfg_weight` | 0.0-1.0 | 0.5 | Controls expressiveness vs consistency |
| `exaggeration` | 0.0-1.0 | 0.5 | Controls emotional intensity |
| `speed` | 0.5-2.0 | 1.0 | Speech rate multiplier |

## Parameter Details

### 1. Temperature (Randomness)

Controls how much variation exists in the speech generation.

- **Low (0.1-0.4)**: Consistent, predictable, robotic
- **Medium (0.5-0.8)**: Natural variation ✅ **RECOMMENDED**
- **High (0.9-2.0)**: Creative, varied, potentially unpredictable

**For human-like voices**: Use `0.6-0.8`

### 2. CFG Weight (Expressiveness)

Balances between strict text adherence and expressive delivery.

- **Low (0.1-0.3)**: Very expressive, dynamic prosody ✅ **For natural conversations**
- **Medium (0.4-0.6)**: Balanced
- **High (0.7-1.0)**: Monotone, literal interpretation

**For human-like voices**: Use `0.2-0.4`

### 3. Exaggeration (Emotional Intensity)

Controls how dramatically emotions are expressed.

- **Low (0.1-0.3)**: Subtle, professional
- **Medium (0.4-0.6)**: Natural emotional expression ✅ **RECOMMENDED**
- **High (0.7-1.0)**: Dramatic, theatrical

**For human-like voices**: Use `0.5-0.6`

### 4. Speed (Speaking Rate)

Multiplier for speech rate.

- **Slow (0.7-0.9)**: Deliberate, clear
- **Normal (0.92-1.05)**: Natural pace ✅ **RECOMMENDED**
- **Fast (1.1-1.5)**: Quick, energetic

**For human-like voices**: Use `0.92-0.98` (slightly slower feels more natural)

## Recommended Presets

### Natural (Best for Most Use Cases)
```json
{
  "temperature": 0.7,
  "cfg_weight": 0.3,
  "exaggeration": 0.5,
  "speed": 0.95
}
```

### Professional (Medical/Business AI)
```json
{
  "temperature": 0.5,
  "cfg_weight": 0.4,
  "exaggeration": 0.4,
  "speed": 0.93
}
```

### Conversational (Friendly Chat)
```json
{
  "temperature": 0.8,
  "cfg_weight": 0.25,
  "exaggeration": 0.6,
  "speed": 0.92
}
```

### Expressive (Storytelling)
```json
{
  "temperature": 1.0,
  "cfg_weight": 0.2,
  "exaggeration": 0.8,
  "speed": 0.95
}
```

## Usage

### REST API
```bash
curl -X POST http://localhost:8765/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how can I help you today?",
    "voice": "default",
    "temperature": 0.7,
    "cfg_weight": 0.3,
    "exaggeration": 0.5,
    "speed": 0.95
  }'
```

### WebSocket
```javascript
{
  "text": "Hello, how can I help you today?",
  "voice": "default",
  "format": "ulaw",
  "temperature": 0.7,
  "cfg_weight": 0.3,
  "exaggeration": 0.5,
  "speed": 0.95
}
```

## Tips for Human-Like Voices

1. **Start with the "Natural" preset** and adjust from there
2. **Lower CFG weight** (0.2-0.4) for more expressiveness
3. **Slightly reduce speed** (0.92-0.95) for more deliberate speech
4. **Moderate temperature** (0.6-0.8) for variation without unpredictability
5. **Test different exaggeration** values based on your use case

## Common Use Cases

| Use Case | Temperature | CFG Weight | Exaggeration | Speed |
|----------|-------------|------------|--------------|-------|
| Medical AI | 0.5 | 0.4 | 0.4 | 0.93 |
| Customer Service | 0.6 | 0.35 | 0.5 | 0.95 |
| Virtual Assistant | 0.7 | 0.3 | 0.5 | 0.98 |
| Storytelling | 1.0 | 0.2 | 0.8 | 0.95 |
| News Reading | 0.4 | 0.6 | 0.3 | 1.0 |
| Meditation | 0.3 | 0.5 | 0.2 | 0.85 |

## Next Steps

After adjusting synthesis parameters, consider:

1. **Voice cloning** with reference audio (add .wav files to `/app/voices/`)
2. **Text preprocessing** (add commas for pauses, expand abbreviations)
3. **Audio post-processing** (EQ, compression, subtle reverb)

## References

- [Chatterbox-Turbo on HuggingFace](https://huggingface.co/ResembleAI/chatterbox-turbo)
- [Resemble AI Chatterbox](https://www.resemble.ai/chatterbox/)
- [GitHub Repository](https://github.com/resemble-ai/chatterbox)
