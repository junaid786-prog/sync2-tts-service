"""
Kokoro TTS Engine Wrapper
Handles model loading and audio synthesis
"""
import logging
import time
from typing import Generator, Optional
import numpy as np

logger = logging.getLogger(__name__)


class KokoroTTSEngine:
    """
    Wrapper for Kokoro Text-to-Speech model
    Supports both batch and streaming synthesis
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize the TTS engine

        Args:
            use_gpu: Whether to use GPU acceleration (requires CUDA)
        """
        self.use_gpu = use_gpu
        self.model = None
        self.device = None
        self.sample_rate = 24000  # Kokoro native sample rate
        self._load_model()

    def _load_model(self):
        """Load Kokoro model onto GPU/CPU"""
        try:
            import torch
            import kokoro

            # Determine device
            if self.use_gpu and torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                logger.warning("GPU not available, using CPU (slower)")

            # Load model
            logger.info("Loading Kokoro TTS model...")
            start_time = time.time()

            self.model = kokoro.KokoroPipeline(lang_code='a')  # 'a' for American English

            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s on {self.device}")

        except ImportError as e:
            logger.error(f"Failed to import required packages: {e}")
            logger.error("Install with: pip install kokoro torch")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def synthesize(self, text: str, voice: str = "af_sarah") -> np.ndarray:
        """
        Synthesize speech from text (batch mode)

        Args:
            text: Text to synthesize
            voice: Voice ID to use

        Returns:
            numpy array of audio samples (24kHz, float32)
        """
        if not self.model:
            raise RuntimeError("Model not loaded")

        if not text or not text.strip():
            logger.warning("Empty text provided, returning silence")
            return np.zeros(self.sample_rate, dtype=np.float32)  # 1 second silence

        try:
            start_time = time.time()

            # Generate audio using Kokoro
            generator = self.model(text, voice=voice)

            # Collect all audio chunks
            audio_chunks = []
            for chunk in generator:
                if hasattr(chunk, 'audio') and chunk.audio is not None:
                    audio_chunks.append(chunk.audio)

            if not audio_chunks:
                logger.warning("No audio generated")
                return np.zeros(self.sample_rate, dtype=np.float32)

            # Concatenate all chunks
            audio = np.concatenate(audio_chunks)

            synthesis_time = time.time() - start_time
            duration = len(audio) / self.sample_rate
            rtf = synthesis_time / duration  # Real-time factor

            logger.info(f"Synthesized {duration:.2f}s audio in {synthesis_time:.3f}s (RTF: {rtf:.2f})")

            return audio.astype(np.float32)

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise

    def synthesize_stream(self, text: str, voice: str = "af_sarah") -> Generator[np.ndarray, None, None]:
        """
        Synthesize speech from text (streaming mode)
        Yields audio chunks as they're generated

        Args:
            text: Text to synthesize
            voice: Voice ID to use

        Yields:
            numpy arrays of audio chunks (24kHz, float32)
        """
        if not self.model:
            raise RuntimeError("Model not loaded")

        if not text or not text.strip():
            logger.warning("Empty text provided")
            yield np.zeros(int(self.sample_rate * 0.1), dtype=np.float32)
            return

        try:
            start_time = time.time()
            total_samples = 0

            # Generate audio using Kokoro (streaming)
            generator = self.model(text, voice=voice)

            for chunk in generator:
                if hasattr(chunk, 'audio') and chunk.audio is not None:
                    audio_chunk = chunk.audio.astype(np.float32)
                    total_samples += len(audio_chunk)
                    yield audio_chunk

            synthesis_time = time.time() - start_time
            duration = total_samples / self.sample_rate
            logger.info(f"Streamed {duration:.2f}s audio in {synthesis_time:.3f}s")

        except Exception as e:
            logger.error(f"Streaming synthesis failed: {e}")
            raise

    def is_ready(self) -> bool:
        """Check if the engine is ready to process requests"""
        return self.model is not None

    def get_info(self) -> dict:
        """Get engine information"""
        import torch

        return {
            "model": "kokoro",
            "device": self.device,
            "sample_rate": self.sample_rate,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "ready": self.is_ready()
        }
