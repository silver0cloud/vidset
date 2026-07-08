"""
inference.py
-------------
TTS generation engine. Auto-detects best runtime (GPU/PyTorch vs
CPU/ONNX-int8) and streams audio chunks as they're generated.

Dependencies:
    pip install torch torchaudio onnxruntime onnxruntime-gpu numpy soundfile
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Generator, Literal

import numpy as np
import torch
import soundfile as sf


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE      = 22050
CHUNK_DURATION_S = 1.0          # Seconds of audio yielded per stream chunk
MAX_TEXT_LEN     = 1000         # Safety cap per generation request


# ---------------------------------------------------------------------------
# Engine detection
# ---------------------------------------------------------------------------

EngineType = Literal["pytorch_gpu", "onnx_cpu"]


def detect_best_engine() -> EngineType:
    """
    Auto-detect best inference engine.
    GPU available → PyTorch (full precision, fastest on CUDA).
    No GPU → ONNX + int8 quantization (optimized for CPU).
    """
    if torch.cuda.is_available():
        return "pytorch_gpu"
    return "onnx_cpu"


@dataclass
class EngineInfo:
    engine: EngineType
    device: str
    reason: str

    def to_dict(self) -> dict:
        return {"engine": self.engine, "device": self.device, "reason": self.reason}


def get_engine_info() -> EngineInfo:
    engine = detect_best_engine()
    if engine == "pytorch_gpu":
        device_name = torch.cuda.get_device_name(0)
        return EngineInfo(
            engine=engine,
            device=device_name,
            reason="CUDA GPU detected — using full-precision PyTorch model",
        )
    return EngineInfo(
        engine=engine,
        device="cpu",
        reason="No GPU detected — using ONNX int8-quantized model for CPU efficiency",
    )


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class AudioChunk:
    chunk_index: int
    audio: np.ndarray          # float32, shape (n_samples,)
    sample_rate: int
    is_final: bool
    elapsed_sec: float

    def to_bytes(self) -> bytes:
        """Convert to 16-bit PCM bytes for streaming over websocket."""
        clipped = np.clip(self.audio, -1.0, 1.0)
        return (clipped * 32767).astype(np.int16).tobytes()


@dataclass
class GenerationResult:
    text: str
    total_duration_sec: float
    generation_time_sec: float
    real_time_factor: float    # generation_time / audio_duration (lower = faster)
    engine_used: EngineType
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
    Auto-selects PyTorch (GPU) or ONNX int8 (CPU) backend.

    Usage:
        engine = TTSInferenceEngine(
            model_dir="models/user_voice",
            speaker_embedding_path="voice_profile/speaker.npy",
        )
        for chunk in engine.generate_stream("Hello, this is my voice."):
            play(chunk.audio)   # or send over websocket
    """

    def __init__(
        self,
        model_dir: str,
        speaker_embedding_path: str,
        engine: Optional[EngineType] = None,
    ):
        self.model_dir = Path(model_dir)
        self.speaker_embedding = np.load(speaker_embedding_path)
        self.engine_type = engine or detect_best_engine()

        self._model = None
        self._onnx_session = None
        self._loaded = False

        info = get_engine_info() if engine is None else None
        if info:
            print(f"Engine selected: {info.engine} ({info.reason})")

    # ------------------------------------------------------------------
    # Public — streaming generation
    # ------------------------------------------------------------------

    def generate_stream(
        self, text: str
    ) -> Generator[AudioChunk, None, GenerationResult]:
        """
        Streams audio chunks as they're generated.
        Yields AudioChunk objects; caller plays/sends each as it arrives.
        """
        if len(text) > MAX_TEXT_LEN:
            raise ValueError(f"Text exceeds max length of {MAX_TEXT_LEN} characters")

        self._ensure_loaded()
        start_time = time.time()

        chunk_samples = int(CHUNK_DURATION_S * SAMPLE_RATE)
        sentences = self._split_into_sentences(text)

        all_audio: list[np.ndarray] = []
        chunk_idx = 0

        for sentence in sentences:
            # Generate audio for this sentence segment
            sentence_audio = self._synthesize_segment(sentence)
            all_audio.append(sentence_audio)

            # Yield in fixed-size streaming chunks
            pos = 0
            while pos < len(sentence_audio):
                end = min(pos + chunk_samples, len(sentence_audio))
                chunk_audio = sentence_audio[pos:end]
                is_final_chunk = (
                    sentence == sentences[-1] and end == len(sentence_audio)
                )

                yield AudioChunk(
                    chunk_index=chunk_idx,
                    audio=chunk_audio,
                    sample_rate=SAMPLE_RATE,
                    is_final=is_final_chunk,
                    elapsed_sec=time.time() - start_time,
                )
                chunk_idx += 1
                pos = end

        # Build final result summary
        full_audio = np.concatenate(all_audio) if all_audio else np.array([])
        generation_time = time.time() - start_time
        audio_duration = len(full_audio) / SAMPLE_RATE

        return GenerationResult(
            text=text,
            total_duration_sec=audio_duration,
            generation_time_sec=generation_time,
            real_time_factor=(
                generation_time / audio_duration if audio_duration > 0 else 0.0
            ),
            engine_used=self.engine_type,
        )

    # ------------------------------------------------------------------
    # Public — non-streaming (full file generation)
    # ------------------------------------------------------------------

    def generate_to_file(self, text: str, output_path: str) -> GenerationResult:
        """Generates full audio and saves to a .wav file (no streaming)."""
        all_chunks: list[np.ndarray] = []
        gen = self.generate_stream(text)

        result = None
        try:
            while True:
                chunk = next(gen)
                all_chunks.append(chunk.audio)
        except StopIteration as e:
            result = e.value

        full_audio = np.concatenate(all_chunks) if all_chunks else np.array([])
        sf.write(output_path, full_audio, SAMPLE_RATE, subtype="PCM_16")

        result.output_path = output_path
        return result

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        if self.engine_type == "pytorch_gpu":
            self._load_pytorch_model()
        else:
            self._load_onnx_model()

        self._loaded = True

    def _load_pytorch_model(self) -> None:
        """Loads fine-tuned model (base + LoRA adapter) for GPU inference."""
        from f5_tts.model import F5TTS
        from peft import PeftModel

        base_path = self.model_dir.parent / "f5_tts_base.pt"  # base model location
        lora_path = self.model_dir / "lora_adapter"

        print("Loading PyTorch model for GPU inference...")
        base_model = F5TTS.from_pretrained(str(base_path))

        if lora_path.exists():
            self._model = PeftModel.from_pretrained(base_model, str(lora_path))
            print("LoRA adapter loaded")
        else:
            # Full fine-tune checkpoint
            full_path = self.model_dir / "final_model.pt"
            base_model.load_state_dict(torch.load(str(full_path)))
            self._model = base_model

        self._model.to("cuda")
        self._model.eval()

    def _load_onnx_model(self) -> None:
        """Loads int8-quantized ONNX model for CPU inference."""
        import onnxruntime as ort

        onnx_path = self.model_dir / "model_int8.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_path}. "
                "Run export_to_onnx.py after training to generate it."
            )

        print("Loading ONNX int8 model for CPU inference...")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self._onnx_session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _synthesize_segment(self, text: str) -> np.ndarray:
        """
        Synthesizes a single sentence/segment.
        NOTE: Actual F5-TTS generation call (flow-matching sampling)
        depends on the F5-TTS package API — placeholder shown here.
        """
        if self.engine_type == "pytorch_gpu":
            return self._synthesize_pytorch(text)
        return self._synthesize_onnx(text)

    def _synthesize_pytorch(self, text: str) -> np.ndarray:
        with torch.no_grad():
            speaker_tensor = torch.from_numpy(self.speaker_embedding).unsqueeze(0).to("cuda")
            output = self._model.generate(
                text=text,
                speaker_embedding=speaker_tensor,
                sample_rate=SAMPLE_RATE,
            )
            return output.cpu().numpy().squeeze()

    def _synthesize_onnx(self, text: str) -> np.ndarray:
        # Text and speaker embedding fed as ONNX inputs;
        # exact input names depend on the exported graph signature.
        inputs = {
            "text_input": self._encode_text_for_onnx(text),
            "speaker_embedding": self.speaker_embedding.astype(np.float32)[None, :],
        }
        outputs = self._onnx_session.run(None, inputs)
        return outputs[0].squeeze()

    def _encode_text_for_onnx(self, text: str) -> np.ndarray:
        """Placeholder tokenization — actual tokenizer depends on F5-TTS text frontend."""
        # Real implementation uses F5-TTS's phoneme/text tokenizer
        return np.array([ord(c) for c in text], dtype=np.int64)[None, :]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_into_sentences(self, text: str) -> list[str]:
        """Simple sentence splitter for natural streaming chunk boundaries."""
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sentences if s]


# ---------------------------------------------------------------------------
# ONNX export utility — run once after training completes
# ---------------------------------------------------------------------------

class ONNXExporter:
    """
    Exports a fine-tuned PyTorch model to int8-quantized ONNX
    for fast CPU inference. Run once after training.

    Usage:
        exporter = ONNXExporter(model_dir="models/user_voice")
        exporter.export()
    """

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)

    def export(self) -> str:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        print("Exporting model to ONNX...")
        fp32_path = self.model_dir / "model_fp32.onnx"
        int8_path = self.model_dir / "model_int8.onnx"

        self._export_pytorch_to_onnx(str(fp32_path))

        print("Quantizing to int8...")
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )

        fp32_size = fp32_path.stat().st_size / (1024 * 1024)
        int8_size = int8_path.stat().st_size / (1024 * 1024)
        print(f"  FP32 size: {fp32_size:.1f} MB")
        print(f"  INT8 size: {int8_size:.1f} MB ({100 * int8_size / fp32_size:.0f}% of original)")

        return str(int8_path)

    def _export_pytorch_to_onnx(self, output_path: str) -> None:
        """Placeholder — actual export depends on F5-TTS model's forward signature."""
        from f5_tts.model import F5TTS
        from peft import PeftModel

        base_model = F5TTS.from_pretrained(str(self.model_dir.parent / "f5_tts_base.pt"))
        lora_path = self.model_dir / "lora_adapter"

        if lora_path.exists():
            model = PeftModel.from_pretrained(base_model, str(lora_path))
            model = model.merge_and_unload()  # Merge LoRA weights for export
        else:
            model = base_model

        model.eval()
        dummy_text = torch.randint(0, 100, (1, 50))
        dummy_speaker = torch.randn(1, 256)

        torch.onnx.export(
            model,
            (dummy_text, dummy_speaker),
            output_path,
            input_names=["text_input", "speaker_embedding"],
            output_names=["audio_output"],
            dynamic_axes={
                "text_input": {1: "text_length"},
                "audio_output": {1: "audio_length"},
            },
            opset_version=17,
        )


# ---------------------------------------------------------------------------
# Quick CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    info = get_engine_info()
    print(json.dumps(info.to_dict(), indent=2))

    engine = TTSInferenceEngine(
        model_dir="models/user_voice",
        speaker_embedding_path="voice_profile/speaker.npy",
    )

    text = sys.argv[1] if len(sys.argv) > 1 else "Hello, this is my cloned voice speaking."
    print(f"\nGenerating: \"{text}\"\n")

    result = engine.generate_to_file(text, output_path="output.wav")
    print(json.dumps(result.to_dict(), indent=2))
