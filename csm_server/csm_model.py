"""
CSM-1B Model Wrapper for Text-to-Speech
Sesame's Conversational Speech Model - HuggingFace Transformers Implementation
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Generator, List
import logging
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSMModel:
    """CSM-1B Text-to-Speech Model using HuggingFace Transformers"""

    def __init__(
        self,
        model_id: str = "sesame/csm-1b",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.model_id = model_id
        self.sample_rate = 24000  # CSM/Mimi native sample rate

        self.model = None
        self.processor = None

        logger.info(f"Initializing CSM-1B on {self.device}")
        self._load_model()

    def _load_model(self):
        """Load CSM-1B model from HuggingFace"""
        from transformers import CsmForConditionalGeneration, AutoProcessor

        logger.info(f"Loading model: {self.model_id}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # Load model with optimizations
        self.model = CsmForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map=self.device,
            low_cpu_mem_usage=True
        )
        self.model.eval()

        logger.info(f"CSM-1B loaded successfully on {self.device}")

        # Warmup
        self._warmup()

    def _warmup(self):
        """Warmup model with a short synthesis"""
        logger.info("Warming up model...")
        try:
            _ = self.synthesize("Hello.", speaker_id=0)
            logger.info("Warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        context_audio: Optional[torch.Tensor] = None,
        context_text: Optional[str] = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.9,
        top_k: int = 50,
    ) -> np.ndarray:
        """
        Synthesize speech from text

        Args:
            text: Input text to synthesize
            speaker_id: Speaker ID (0-based)
            context_audio: Optional context audio for voice cloning
            context_text: Optional context text (transcript of context_audio)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter

        Returns:
            Audio waveform as numpy array (float32, mono, 24kHz)
        """
        # Format text with speaker ID
        formatted_text = f"[{speaker_id}]{text}"

        # Prepare inputs
        if context_audio is not None and context_text is not None:
            # With context for voice cloning
            context_formatted = f"[{speaker_id}]{context_text}"
            inputs = self.processor(
                text=formatted_text,
                context_text=context_formatted,
                context_audio=context_audio,
                add_special_tokens=True
            ).to(self.device)
        else:
            # Without context
            inputs = self.processor(
                text=formatted_text,
                add_special_tokens=True
            ).to(self.device)

        # Generate audio
        with torch.cuda.amp.autocast(enabled=self.device == "cuda"):
            audio_output = self.model.generate(
                **inputs,
                output_audio=True,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
            )

        # Convert to numpy
        if isinstance(audio_output, torch.Tensor):
            audio = audio_output.squeeze().cpu().numpy().astype(np.float32)
        else:
            audio = np.array(audio_output, dtype=np.float32)

        # Normalize
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        return audio

    @torch.inference_mode()
    def synthesize_conversation(
        self,
        turns: List[dict],
        temperature: float = 0.9,
    ) -> np.ndarray:
        """
        Synthesize a multi-turn conversation

        Args:
            turns: List of dicts with 'speaker_id' and 'text' keys
            temperature: Sampling temperature

        Returns:
            Full conversation audio
        """
        all_audio = []
        context_audio = None
        context_text = None

        for turn in turns:
            speaker_id = turn.get("speaker_id", 0)
            text = turn["text"]

            audio = self.synthesize(
                text=text,
                speaker_id=speaker_id,
                context_audio=context_audio,
                context_text=context_text,
                temperature=temperature
            )

            all_audio.append(audio)

            # Update context for next turn
            context_audio = torch.from_numpy(audio).unsqueeze(0)
            context_text = text

        # Concatenate all audio
        return np.concatenate(all_audio)

    @torch.inference_mode()
    def synthesize_streaming(
        self,
        text: str,
        chunk_size_ms: int = 500,
        speaker_id: int = 0
    ) -> Generator[np.ndarray, None, None]:
        """
        Stream audio generation in chunks

        Note: CSM doesn't support true token-by-token streaming yet,
        so we generate full audio and chunk it.

        Args:
            text: Input text
            chunk_size_ms: Size of each audio chunk in milliseconds
            speaker_id: Speaker ID

        Yields:
            Audio chunks as numpy arrays
        """
        chunk_samples = int(self.sample_rate * chunk_size_ms / 1000)

        # Generate full audio
        full_audio = self.synthesize(text, speaker_id=speaker_id)

        # Yield in chunks
        for i in range(0, len(full_audio), chunk_samples):
            chunk = full_audio[i:i + chunk_samples]
            if len(chunk) > 0:
                yield chunk

    def to_wav_bytes(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> bytes:
        """Convert audio array to WAV bytes"""
        import soundfile as sf

        sr = sample_rate or self.sample_rate
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        return buffer.read()

    def to_ulaw_bytes(self, audio: np.ndarray, target_rate: int = 8000) -> bytes:
        """Convert audio to u-law format for telephony (8kHz)"""
        import audioop

        # Resample to target rate if needed
        if self.sample_rate != target_rate:
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(self.sample_rate, target_rate)
            audio_tensor = resampler(audio_tensor)
            audio = audio_tensor.squeeze(0).numpy()

        # Normalize to int16 range
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        # Convert to u-law
        ulaw_bytes = audioop.lin2ulaw(audio_int16.tobytes(), 2)

        return ulaw_bytes

    def save_audio(self, audio: np.ndarray, filepath: str):
        """Save audio to file using processor"""
        self.processor.save_audio(audio, filepath)


# Singleton instance for server use
_model_instance: Optional[CSMModel] = None


def get_model() -> CSMModel:
    """Get or create singleton model instance"""
    global _model_instance
    if _model_instance is None:
        _model_instance = CSMModel()
    return _model_instance


if __name__ == "__main__":
    # Test the model
    model = CSMModel()

    # Basic synthesis
    print("Testing basic synthesis...")
    audio = model.synthesize("Hello, this is a test of the CSM speech model.")
    model.save_audio(audio, "test_basic.wav")
    print(f"Saved test_basic.wav ({len(audio) / model.sample_rate:.2f}s)")

    # Test u-law conversion
    print("Testing u-law conversion...")
    ulaw = model.to_ulaw_bytes(audio)
    print(f"U-law bytes: {len(ulaw)}")

    print("Done!")
