"""
setup.py
--------
Handles one-time model download before the user starts recording.
Downloads F5-TTS base model + Vocos vocoder from HuggingFace with
progress reporting, and checks if already cached.

Key design:
    - Runs ONCE on first launch (checks cache first, no re-download)
    - Reports download progress via a callback (wired to WebSocket /ws/setup)
    - Suppresses all HuggingFace noise (warnings, progress bars, token prompts)
    - Sets HF_HUB_DISABLE_SYMLINKS_WARNING for Windows compatibility
"""

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Suppress ALL HuggingFace noise before any HF imports happen
# ---------------------------------------------------------------------------
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"   # Windows symlink warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"         # tokenizers fork warning
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"               # TensorFlow noise
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*Annotated type alias.*")
warnings.filterwarnings("ignore", message=".*Field().*")
warnings.filterwarnings("ignore", message=".*ffmpeg.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*avconv.*", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Model references
# ---------------------------------------------------------------------------

MODEL_REPO       = "SWivid/F5-TTS"
MODEL_FILE       = "F5TTS_v1_Base/model_1250000.safetensors"
VOCAB_FILE       = "F5TTS_v1_Base/vocab.txt"
VOCODER_REPO     = "charactr/vocos-mel-24khz"

TOTAL_STEPS      = 3   # model weights + vocab + vocoder


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SetupProgress:
    step: int
    total_steps: int
    label: str
    done: bool = False
    error: Optional[str] = None

    @property
    def percent(self) -> float:
        return round((self.step / max(self.total_steps, 1)) * 100, 1)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "total_steps": self.total_steps,
            "percent": self.percent,
            "label": self.label,
            "done": self.done,
            "error": self.error,
        }


SetupCallback = Callable[[SetupProgress], None]


# ---------------------------------------------------------------------------
# Cache checker
# ---------------------------------------------------------------------------

def is_model_cached() -> bool:
    """
    Returns True if the F5-TTS model weights are already in the
    HuggingFace cache — no download needed.
    """
    try:
        from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST
        result = try_to_load_from_cache(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
        )
        return isinstance(result, str)   # str = path exists in cache
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

class ModelSetup:
    """
    Downloads F5-TTS base model + Vocos vocoder on first launch.
    Skips everything silently if already cached.

    Usage:
        setup = ModelSetup()
        setup.run(on_progress=my_callback)
    """

    def run(self, on_progress: Optional[SetupCallback] = None) -> bool:
        """
        Returns True if ready (cached or just downloaded), False on error.
        """
        def emit(step: int, label: str, done=False, error=None):
            if on_progress:
                on_progress(SetupProgress(
                    step=step,
                    total_steps=TOTAL_STEPS,
                    label=label,
                    done=done,
                    error=error,
                ))

        # Fast path — already downloaded
        if is_model_cached():
            emit(TOTAL_STEPS, "Model already downloaded.", done=True)
            return True

        try:
            from huggingface_hub import hf_hub_download

            emit(0, "Downloading F5-TTS model weights (~1.3 GB)...")
            hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILE,
            )

            emit(1, "Downloading vocabulary file...")
            hf_hub_download(
                repo_id=MODEL_REPO,
                filename=VOCAB_FILE,
            )

            emit(2, "Downloading Vocos vocoder...")
            from huggingface_hub import hf_hub_download
            # Download only the essential Vocos files — not the full repo
            for fname in [
                "config.json",
                "pytorch_model.bin",
                "feature_extractor_config.json",
            ]:
                hf_hub_download(
                    repo_id=VOCODER_REPO,
                    filename=fname,
                )

            emit(TOTAL_STEPS, "All models ready!", done=True)
            return True

        except Exception as e:
            emit(0, "", error=str(e))
            return False


# ---------------------------------------------------------------------------
# CLI utility
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def print_progress(p: SetupProgress):
        print(f"[{p.percent:5.1f}%] {p.label}")
        if p.error:
            print(f"ERROR: {p.error}")

    setup = ModelSetup()
    success = setup.run(on_progress=print_progress)
    print("Ready:", success)