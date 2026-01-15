"""
Audio Processing Pipeline
Handles resampling, normalization, chunking, and encoding for Asterisk compatibility
"""
import logging
from typing import Generator
import numpy as np

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Processes audio from Kokoro TTS (24kHz) to Asterisk format (8kHz ulaw)

    Pipeline:
    1. Resample: 24kHz → 8kHz
    2. Normalize: Adjust levels to -3dB
    3. Chunk: Split into 40ms frames (320 samples)
    4. Fade: Apply fade-in/out to prevent clicks
    5. Encode: Convert to 8-bit ulaw
    """

    def __init__(
        self,
        input_sample_rate: int = 24000,
        output_sample_rate: int = 8000,
        chunk_duration_ms: int = 40,
        fade_duration_ms: int = 2,
        normalize_db: float = -3.0
    ):
        self.input_sr = input_sample_rate
        self.output_sr = output_sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.fade_duration_ms = fade_duration_ms
        self.normalize_db = normalize_db

        # Calculate chunk size in samples
        self.chunk_samples = int(output_sample_rate * chunk_duration_ms / 1000)
        self.fade_samples = int(output_sample_rate * fade_duration_ms / 1000)

        logger.info(f"AudioProcessor initialized: {input_sample_rate}Hz → {output_sample_rate}Hz")
        logger.info(f"Chunk size: {self.chunk_samples} samples ({chunk_duration_ms}ms)")

    def resample(self, audio: np.ndarray) -> np.ndarray:
        """
        Resample audio from input sample rate to output sample rate

        Args:
            audio: Input audio (float32, input_sr Hz)

        Returns:
            Resampled audio (float32, output_sr Hz)
        """
        if self.input_sr == self.output_sr:
            return audio

        try:
            from scipy import signal

            # Calculate resampling ratio
            num_samples = int(len(audio) * self.output_sr / self.input_sr)

            # Use polyphase resampling for quality
            resampled = signal.resample(audio, num_samples)

            return resampled.astype(np.float32)

        except ImportError:
            # Fallback to simple linear interpolation
            logger.warning("scipy not available, using linear interpolation")
            ratio = self.output_sr / self.input_sr
            indices = np.arange(0, len(audio), 1/ratio)
            indices = indices[indices < len(audio) - 1].astype(int)
            return audio[indices].astype(np.float32)

    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to target dB level

        Args:
            audio: Input audio (float32)

        Returns:
            Normalized audio (float32)
        """
        if len(audio) == 0:
            return audio

        # Calculate current peak
        peak = np.max(np.abs(audio))

        if peak < 1e-6:  # Silence
            return audio

        # Calculate target amplitude from dB
        target_amplitude = 10 ** (self.normalize_db / 20)

        # Normalize
        normalized = audio * (target_amplitude / peak)

        # Clip to prevent overflow
        normalized = np.clip(normalized, -1.0, 1.0)

        return normalized.astype(np.float32)

    def apply_fade(self, audio: np.ndarray, fade_in: bool = True, fade_out: bool = True) -> np.ndarray:
        """
        Apply fade-in and/or fade-out to prevent clicking sounds

        Args:
            audio: Input audio (float32)
            fade_in: Apply fade-in at start
            fade_out: Apply fade-out at end

        Returns:
            Audio with fades applied
        """
        if len(audio) < self.fade_samples * 2:
            return audio  # Too short for fades

        audio = audio.copy()

        if fade_in and self.fade_samples > 0:
            # Linear fade-in
            fade_curve = np.linspace(0, 1, self.fade_samples)
            audio[:self.fade_samples] *= fade_curve

        if fade_out and self.fade_samples > 0:
            # Linear fade-out
            fade_curve = np.linspace(1, 0, self.fade_samples)
            audio[-self.fade_samples:] *= fade_curve

        return audio

    def chunk_audio(self, audio: np.ndarray) -> Generator[np.ndarray, None, None]:
        """
        Split audio into fixed-size chunks

        Args:
            audio: Input audio (float32)

        Yields:
            Audio chunks of chunk_samples size
        """
        for i in range(0, len(audio), self.chunk_samples):
            chunk = audio[i:i + self.chunk_samples]

            # Pad last chunk if necessary
            if len(chunk) < self.chunk_samples:
                chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)))

            yield chunk

    def encode_ulaw(self, audio: np.ndarray) -> bytes:
        """
        Encode audio to 8-bit ulaw format (for Asterisk)

        Args:
            audio: Input audio (float32, -1 to 1)

        Returns:
            ulaw encoded bytes
        """
        # Convert to 16-bit PCM first
        pcm16 = (audio * 32767).astype(np.int16)

        # ulaw encoding constants
        BIAS = 0x84
        CLIP = 32635

        # ulaw encoding table
        exp_table = [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                     4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                     5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                     5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]

        ulaw_bytes = bytearray(len(pcm16))

        for i, sample in enumerate(pcm16):
            # Get sign
            sign = (sample >> 8) & 0x80
            if sign:
                sample = -sample

            # Clip
            if sample > CLIP:
                sample = CLIP

            # Add bias
            sample += BIAS

            # Get exponent and mantissa
            exponent = exp_table[(sample >> 7) & 0xFF]
            mantissa = (sample >> (exponent + 3)) & 0x0F

            # Combine and complement
            ulaw_bytes[i] = ~(sign | (exponent << 4) | mantissa) & 0xFF

        return bytes(ulaw_bytes)

    def encode_pcm16(self, audio: np.ndarray) -> bytes:
        """
        Encode audio to 16-bit PCM format

        Args:
            audio: Input audio (float32, -1 to 1)

        Returns:
            PCM16 encoded bytes (little-endian)
        """
        pcm16 = (audio * 32767).astype(np.int16)
        return pcm16.tobytes()

    def process(self, audio: np.ndarray, output_format: str = "ulaw") -> bytes:
        """
        Full processing pipeline: resample → normalize → fade → encode

        Args:
            audio: Input audio from TTS (24kHz float32)
            output_format: "ulaw" or "pcm16"

        Returns:
            Processed audio bytes ready for Asterisk
        """
        # Step 1: Resample 24kHz → 8kHz
        resampled = self.resample(audio)

        # Step 2: Normalize to target dB
        normalized = self.normalize(resampled)

        # Step 3: Apply fades to prevent clicks
        faded = self.apply_fade(normalized)

        # Step 4: Encode to output format
        if output_format == "ulaw":
            return self.encode_ulaw(faded)
        else:
            return self.encode_pcm16(faded)

    def process_stream(self, audio_generator: Generator, output_format: str = "ulaw") -> Generator[bytes, None, None]:
        """
        Streaming processing pipeline

        Args:
            audio_generator: Generator yielding audio chunks (24kHz float32)
            output_format: "ulaw" or "pcm16"

        Yields:
            Processed audio chunks ready for Asterisk
        """
        buffer = np.array([], dtype=np.float32)

        for audio_chunk in audio_generator:
            # Resample chunk
            resampled = self.resample(audio_chunk)

            # Add to buffer
            buffer = np.concatenate([buffer, resampled])

            # Yield complete chunks from buffer
            while len(buffer) >= self.chunk_samples:
                chunk = buffer[:self.chunk_samples]
                buffer = buffer[self.chunk_samples:]

                # Normalize chunk
                normalized = self.normalize(chunk)

                # Encode
                if output_format == "ulaw":
                    yield self.encode_ulaw(normalized)
                else:
                    yield self.encode_pcm16(normalized)

        # Yield remaining buffer (padded)
        if len(buffer) > 0:
            # Pad to chunk size
            padded = np.pad(buffer, (0, self.chunk_samples - len(buffer)))
            normalized = self.normalize(padded)

            if output_format == "ulaw":
                yield self.encode_ulaw(normalized)
            else:
                yield self.encode_pcm16(normalized)
