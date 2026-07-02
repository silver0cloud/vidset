"""
inference.py
-------------
TTS generation using the real F5-TTS API.

Uses:
    - f5_tts.api.F5TTS          (high-level inference class)
    - f5_tts.infer.utils_infer  (load_model, load_vocoder, preprocess_ref_audio_text, infer_process)
    - peft PeftModel             (loads LoRA adapter on top of base)

F5-TTS inference is zero-shot voice cloning:
    ref_audio + ref_text + gen_text → generated_audio

After fine-tuning, we use the best recorded take as the reference audio
(fine-tuned weights now encode the voice, ref_audio anchors prosody/style).

30s TOTAL duration limit per call (ref_audio + generated combined).
Longer texts are auto-chunked into sentences.

Install:
    pip install f5-tts peft
"""

import io
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Generator

import numpy as np
import soundfile as sf
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SR           = 24000     # F5-TTS native sample rate
CHUNK_DURATION_S    = 1.5       # Audio chunk size for streaming (seconds)
MAX_TEXT_LEN        = 1000      # Safety cap per generation request
MAX_GEN_DURATION    = 25.0      # Seconds — leave room within F5-TTS 30s cap
REF_AUDIO_DURATION  = 4.0      # Target ref_audio length (seconds)
BASE_MODEL_HF       = "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
VOCAB_HF            = "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt"
NFE_STEPS           = 16        # Lower = faster, slightly less quality (default 32)


# ---------------------------------------------------------------------------
# Engine detection
# ---------------------------------------------------------------------------

def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class AudioChunk:
    chunk_index: int
    audio: np.ndarray
    sample_rate: int
    is_final: bool
    elapsed_sec: float

    def to_bytes(self) -> bytes:
        clipped = np.clip(self.audio, -1.0, 1.0)
        return (clipped * 32767).astype(np.int16).tobytes()


@dataclass
class GenerationResult:
    text: str
    total_duration_sec: float
    generation_time_sec: float
    real_time_factor: float
    engine_used: str
    output_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "total_duration_sec": round(self.total_duration_sec, 2),
            "generation_time_sec": round(self.generation_time_sec, 2),
            "real_time_factor": round(self.real_time_factor, 3),
            "engine_used": self.engine_used,
            "output_path": self.output_path,
        }


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class TTSInferenceEngine:
    """
    Generates speech from text using the user's fine-tuned voice model.
    Uses real F5-TTS API with LoRA adapter loaded on top of base model.

    Usage:
        engine = TTSInferenceEngine(
            model_dir="models/user_voice",
            ref_audio_path="dataset/wavs/S001_take1_cleaned.wav",
            ref_text="The morning light filtered through the curtains...",
        )
        for chunk in engine.generate_stream("Hello, this is my voice."):
            play(chunk.audio)
    """

    def __init__(
        self,
        model_dir: str,
        ref_audio_path: str,
        ref_text: str,
        speaker_embedding_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_dir = Path(model_dir)
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.device = device or detect_device()
        self._model = None
        self._vocoder = None
        self._loaded = False

        print(f"Inference engine: device={self.device}")

    # ------------------------------------------------------------------
    # Public — streaming generation
    # ------------------------------------------------------------------

    def generate_stream(self, text: str) -> Generator[AudioChunk, None, GenerationResult]:
        if len(text) > MAX_TEXT_LEN:
            raise ValueError(f"Text exceeds max {MAX_TEXT_LEN} characters")

        self._ensure_loaded()
        start_time = time.time()
        chunk_samples = int(CHUNK_DURATION_S * TARGET_SR)
        sentences = self._split_sentences(text)

        all_audio: list[np.ndarray] = []
        chunk_idx = 0

        for i, sentence in enumerate(sentences):
            segment_audio = self._synthesize(sentence)
            all_audio.append(segment_audio)

            # Stream in fixed-size chunks
            pos = 0
            while pos < len(segment_audio):
                end = min(pos + chunk_samples, len(segment_audio))
                is_final = (i == len(sentences) - 1) and (end == len(segment_audio))
                yield AudioChunk(
                    chunk_index=chunk_idx,
                    audio=segment_audio[pos:end],
                    sample_rate=TARGET_SR,
                    is_final=is_final,
                    elapsed_sec=time.time() - start_time,
                )
                chunk_idx += 1
                pos = end

        full_audio = np.concatenate(all_audio) if all_audio else np.array([])
        gen_time = time.time() - start_time
        audio_duration = len(full_audio) / TARGET_SR

        return GenerationResult(
            text=text,
            total_duration_sec=audio_duration,
            generation_time_sec=gen_time,
            real_time_factor=gen_time / audio_duration if audio_duration > 0 else 0.0,
            engine_used=f"f5tts_lora_{self.device}",
        )

    def generate_to_file(self, text: str, output_path: str) -> GenerationResult:
        """Non-streaming generation — saves full audio to file."""
        chunks = []
        gen = self.generate_stream(text)
        result = None
        try:
            while True:
                chunk = next(gen)
                chunks.append(chunk.audio)
        except StopIteration as e:
            result = e.value

        full = np.concatenate(chunks) if chunks else np.array([])
        sf.write(output_path, full, TARGET_SR, subtype="PCM_16")
        result.output_path = output_path
        return result

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._load_model()
        self._loaded = True

    def _load_model(self) -> None:
        from f5_tts.model import DiT
        from f5_tts.infer.utils_infer import load_model, load_vocoder
        from peft import PeftModel
        from cached_path import cached_path

        print("Loading F5-TTS base model for inference...")
        ckpt_path = str(cached_path(BASE_MODEL_HF))
        vocab_path = str(cached_path(VOCAB_HF))

        model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2,
            text_dim=512, conv_layers=4
        )

        base_model = load_model(
            model_cls=DiT,
            model_cfg=model_cfg,
            ckpt_path=ckpt_path,
            mel_spec_type="vocos",
            vocab_file=vocab_path,
            ode_method="euler",
            use_ema=True,
            device=self.device,
        )

        # Load LoRA adapter if present
        adapter_path = self.model_dir / "lora_adapter"
        if adapter_path.exists():
            print(f"Loading LoRA adapter from {adapter_path}...")
            self._model = PeftModel.from_pretrained(
                base_model, str(adapter_path)
            )
            self._model = self._model.merge_and_unload()
            print("LoRA weights merged into base model.")
        else:
            print("No LoRA adapter found — using base model (zero-shot mode)")
            self._model = base_model

        self._model.eval()
        self._model.to(self.device)

        # Load Vocos vocoder
        self._vocoder = load_vocoder(
            vocoder_name="vocos",
            is_local=False,
            device=self.device,
        )
        print("Model ready.")

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _synthesize(self, text: str) -> np.ndarray:
        """
        Synthesizes a single sentence using F5-TTS zero-shot voice cloning.
        Uses the best recorded take as ref_audio for voice identity.
        """
        from f5_tts.infer.utils_infer import (
            preprocess_ref_audio_text,
            infer_process,
        )

        # Preprocess reference audio + text
        ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(
            ref_audio_orig_path=self.ref_audio_path,
            ref_text=self.ref_text,
            show_info=False,
            device=self.device,
        )

        # Run flow-matching inference
        with torch.inference_mode():
            audio, _, _ = infer_process(
                ref_audio=ref_audio_proc,
                ref_text=ref_text_proc,
                gen_text=text,
                model_obj=self._model,
                vocoder=self._vocoder,
                mel_spec_type="vocos",
                nfe_step=NFE_STEPS,
                cfg_strength=2.0,
                sway_sampling_coef=-1.0,   # Sway Sampling — improves quality
                speed=1.0,
                device=self.device,
            )

        return audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences. Each sentence stays within F5-TTS's
        30s total duration limit when combined with the ref_audio.
        Max ~25s of generated audio per sentence to be safe.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        result = []
        current = ""

        for s in sentences:
            # Rough estimate: ~15 chars/sec of speech
            if len(current) + len(s) < int(MAX_GEN_DURATION * 15):
                current = f"{current} {s}".strip() if current else s
            else:
                if current:
                    result.append(current)
                current = s

        if current:
            result.append(current)

        return result or [text]


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python inference.py <model_dir> <ref_audio.wav> <ref_text>")
        print('Example: python inference.py models/user_voice dataset/wavs/S001_take1_cleaned.wav "The morning light filtered through the curtains."')
        sys.exit(1)

    model_dir, ref_audio, ref_text = sys.argv[1], sys.argv[2], sys.argv[3]
    text = sys.argv[4] if len(sys.argv) > 4 else "Hello, this is my cloned voice."

    engine = TTSInferenceEngine(
        model_dir=model_dir,
        ref_audio_path=ref_audio,
        ref_text=ref_text,
    )
    result = engine.generate_to_file(text, output_path="output.wav")
    print(json.dumps(result.to_dict(), indent=2))
