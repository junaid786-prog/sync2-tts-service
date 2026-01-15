"""
Configuration settings for TTS Service
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TTSConfig:
    """TTS Service Configuration"""

    # Server settings
    host: str = os.getenv("TTS_HOST", "0.0.0.0")
    port: int = int(os.getenv("TTS_PORT", "8765"))

    # Model settings
    model_name: str = os.getenv("TTS_MODEL", "kokoro")
    default_voice: str = os.getenv("TTS_DEFAULT_VOICE", "af_sarah")

    # Audio settings
    input_sample_rate: int = 24000   # Kokoro outputs 24kHz
    output_sample_rate: int = 8000   # Asterisk needs 8kHz
    chunk_duration_ms: int = 40      # 40ms chunks (320 samples at 8kHz)
    fade_duration_ms: int = 2        # 2ms fade to prevent clicks
    normalize_db: float = -3.0       # Target normalization level

    # Performance settings
    use_gpu: bool = os.getenv("TTS_USE_GPU", "true").lower() == "true"
    max_concurrent_requests: int = int(os.getenv("TTS_MAX_CONCURRENT", "10"))
    request_timeout_seconds: int = int(os.getenv("TTS_TIMEOUT", "30"))

    # Caching
    cache_enabled: bool = os.getenv("TTS_CACHE_ENABLED", "true").lower() == "true"
    cache_max_size: int = int(os.getenv("TTS_CACHE_MAX_SIZE", "1000"))

    # Logging
    log_level: str = os.getenv("TTS_LOG_LEVEL", "INFO")


# Available voices for Kokoro model
AVAILABLE_VOICES = {
    "af_sarah": "American Female - Sarah (default)",
    "af_bella": "American Female - Bella",
    "am_adam": "American Male - Adam",
    "am_michael": "American Male - Michael",
    "bf_emma": "British Female - Emma",
    "bm_george": "British Male - George",
}


# Global config instance
config = TTSConfig()
