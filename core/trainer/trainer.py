"""
trainer.py
----------
F5-TTS LoRA fine-tuning pipeline.

Strategy:
    - LoRA fine-tuning by default (low VRAM, fast)
    - Optional full fine-tune for high-end GPUs
    - Auto checkpoint + resume on crash
    - Live loss curve + ETA reporting via callback

Dependencies:
    pip install torch torchaudio transformers accelerate peft
    # F5-TTS base model loaded via its own package (assumed installed separately)
"""

import json
import time
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EPOCHS          = 100
DEFAULT_BATCH_SIZE      = 4
DEFAULT_LR              = 1e-4
DEFAULT_LR_LORA         = 5e-4       # LoRA tolerates higher LR
LORA_RANK               = 16
LORA_ALPHA              = 32
LORA_DROPOUT            = 0.05
CHECKPOINT_EVERY_EPOCHS = 5
SAMPLE_RATE             = 22050
GRAD_ACCUM_STEPS        = 4          # Helps low-VRAM GPUs simulate larger batch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    base_model_path: str
    filelist_path: str
    output_dir: str
    speaker_embedding_path: str

    mode: str = "lora"               # "lora" | "full"
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = field(default=DEFAULT_LR_LORA)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_every: int = CHECKPOINT_EVERY_EPOCHS

    def __post_init__(self):
        if self.mode == "full":
            self.learning_rate = DEFAULT_LR
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return {
            "base_model_path": self.base_model_path,
            "filelist_path": self.filelist_path,
            "output_dir": self.output_dir,
            "mode": self.mode,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "device": self.device,
        }


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------

@dataclass
class TrainingProgress:
    epoch: int
    total_epochs: int
    step: int
    total_steps: int
    loss: float
    avg_loss_last_10: float
    elapsed_sec: float
    eta_sec: float
    checkpoint_saved: bool = False

    @property
    def percent(self) -> float:
        return round((self.epoch / self.total_epochs) * 100, 1)

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "step": self.step,
            "total_steps": self.total_steps,
            "percent": self.percent,
            "loss": round(self.loss, 4),
            "avg_loss_last_10": round(self.avg_loss_last_10, 4),
            "elapsed_sec": round(self.elapsed_sec, 1),
            "eta_sec": round(self.eta_sec, 1),
            "eta_human": self._human_time(self.eta_sec),
            "checkpoint_saved": self.checkpoint_saved,
        }

    @staticmethod
    def _human_time(seconds: float) -> str:
        if seconds < 60:
            return f"{int(seconds)}s"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        if minutes < 60:
            return f"{minutes}m {secs}s"
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}h {minutes}m"


# Type alias for progress callbacks (e.g. feeds FastAPI websocket / Tauri events)
ProgressCallback = Callable[[TrainingProgress], None]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VoiceDataset(Dataset):
    """
    Loads the filelist.txt produced by segmenter.py:
        path|text|speaker_id
    """

    def __init__(self, filelist_path: str, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.entries: list[tuple[str, str, str]] = []

        with open(filelist_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) == 3:
                    self.entries.append(tuple(parts))

        if len(self.entries) == 0:
            raise ValueError(f"No valid entries found in {filelist_path}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        wav_path, text, speaker_id = self.entries[idx]
        audio, sr = torchaudio.load(wav_path)

        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        return {
            "audio": audio.squeeze(0),
            "text": text,
            "speaker_id": speaker_id,
            "wav_path": wav_path,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pads audio in batch to the same length."""
    max_len = max(item["audio"].shape[0] for item in batch)
    padded_audio = torch.zeros(len(batch), max_len)
    lengths = torch.zeros(len(batch), dtype=torch.long)

    for i, item in enumerate(batch):
        a = item["audio"]
        padded_audio[i, : a.shape[0]] = a
        lengths[i] = a.shape[0]

    return {
        "audio": padded_audio,
        "lengths": lengths,
        "texts": [item["text"] for item in batch],
        "speaker_ids": [item["speaker_id"] for item in batch],
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class F5TTSTrainer:
    """
    Fine-tunes F5-TTS on a user's voice dataset.

    Usage:
        config = TrainingConfig(
            base_model_path="models/f5_tts_base.pt",
            filelist_path="dataset/filelist.txt",
            output_dir="models/user_voice",
            speaker_embedding_path="voice_profile/speaker.npy",
            mode="lora",
        )
        trainer = F5TTSTrainer(config)
        trainer.train(on_progress=my_callback)
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.start_epoch = 0
        self.loss_history: list[float] = []

        self._checkpoint_dir = Path(config.output_dir) / "checkpoints"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._latest_checkpoint = self._checkpoint_dir / "latest.pt"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def train(self, on_progress: Optional[ProgressCallback] = None) -> dict:
        """
        Runs the full training loop. Resumes automatically if a checkpoint
        exists. Returns a final training report.
        """
        self._load_base_model()
        self._apply_finetune_strategy()
        self._maybe_resume()

        dataset = VoiceDataset(self.config.filelist_path, sample_rate=SAMPLE_RATE)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2 if self.config.device == "cuda" else 0,
        )

        total_steps_per_epoch = len(loader)
        total_steps = total_steps_per_epoch * self.config.epochs
        start_time = time.time()
        global_step = self.start_epoch * total_steps_per_epoch

        print(f"\nTraining started — mode: {self.config.mode}, device: {self.config.device}")
        print(f"Dataset: {len(dataset)} samples, {total_steps_per_epoch} steps/epoch\n")

        for epoch in range(self.start_epoch, self.config.epochs):
            epoch_losses = []

            for step, batch in enumerate(loader):
                loss = self._train_step(batch, step)
                epoch_losses.append(loss)
                self.loss_history.append(loss)
                global_step += 1

                if on_progress and step % 5 == 0:
                    elapsed = time.time() - start_time
                    progress = self._build_progress(
                        epoch, global_step, total_steps, loss, elapsed
                    )
                    on_progress(progress)

            avg_epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            print(f"  Epoch {epoch + 1}/{self.config.epochs} — avg loss: {avg_epoch_loss:.4f}")

            checkpoint_saved = False
            if (epoch + 1) % self.config.checkpoint_every == 0 or epoch == self.config.epochs - 1:
                self._save_checkpoint(epoch + 1)
                checkpoint_saved = True

            if on_progress:
                elapsed = time.time() - start_time
                progress = self._build_progress(
                    epoch, global_step, total_steps, avg_epoch_loss, elapsed,
                    checkpoint_saved=checkpoint_saved,
                )
                on_progress(progress)

        final_path = self._save_final_model()
        report = self._build_report(final_path)
        print(f"\nTraining complete! Final model: {final_path}\n")
        return report

    def resume_check(self) -> bool:
        """Returns True if a checkpoint exists to resume from."""
        return self._latest_checkpoint.exists()

    # ------------------------------------------------------------------
    # Model loading & fine-tune strategy
    # ------------------------------------------------------------------

    def _load_base_model(self) -> None:
        """
        Loads the pretrained F5-TTS base model.
        NOTE: Actual F5-TTS model class import path depends on the
        f5-tts package structure — placeholder shown here.
        """
        from f5_tts.model import F5TTS   # actual import depends on package

        print(f"Loading base model from {self.config.base_model_path}...")
        self.model = F5TTS.from_pretrained(self.config.base_model_path)
        self.model.to(self.config.device)

    def _apply_finetune_strategy(self) -> None:
        """Applies LoRA adapters or unfreezes all layers for full fine-tune."""
        if self.config.mode == "lora":
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)
            print(f"LoRA applied — rank={LORA_RANK}, alpha={LORA_ALPHA}")

            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

        else:
            for param in self.model.parameters():
                param.requires_grad = True
            print("Full fine-tune mode — all layers unfrozen")

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
        )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _train_step(self, batch: dict, step: int) -> float:
        """
        Single training step with gradient accumulation.
        NOTE: Actual loss computation depends on F5-TTS's flow-matching
        objective — placeholder forward pass shown here.
        """
        self.model.train()
        audio = batch["audio"].to(self.config.device)
        texts = batch["texts"]

        outputs = self.model(audio=audio, texts=texts)
        loss = outputs.loss / GRAD_ACCUM_STEPS

        loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item() * GRAD_ACCUM_STEPS

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int) -> None:
        """Saves a resumable checkpoint with optimizer state."""
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss_history": self.loss_history,
            "config": self.config.to_dict(),
        }
        torch.save(checkpoint, str(self._latest_checkpoint))

        # Keep a numbered backup every checkpoint too (crash safety)
        numbered_path = self._checkpoint_dir / f"epoch_{epoch}.pt"
        shutil.copy2(str(self._latest_checkpoint), str(numbered_path))

        print(f"    Checkpoint saved → epoch {epoch}")

    def _maybe_resume(self) -> None:
        """Resumes from latest.pt if it exists."""
        if not self._latest_checkpoint.exists():
            return

        print(f"Found checkpoint — resuming from {self._latest_checkpoint}")
        checkpoint = torch.load(str(self._latest_checkpoint), map_location=self.config.device)

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.start_epoch = checkpoint["epoch"]
        self.loss_history = checkpoint.get("loss_history", [])

        print(f"Resumed from epoch {self.start_epoch}\n")

    def _save_final_model(self) -> str:
        """Saves the final fine-tuned model (LoRA adapter or full weights)."""
        final_path = Path(self.config.output_dir) / "final_model.pt"

        if self.config.mode == "lora":
            # Save only LoRA adapter weights — much smaller file
            self.model.save_pretrained(str(Path(self.config.output_dir) / "lora_adapter"))
            final_path = Path(self.config.output_dir) / "lora_adapter"
        else:
            torch.save(self.model.state_dict(), str(final_path))

        return str(final_path)

    # ------------------------------------------------------------------
    # Progress helpers
    # ------------------------------------------------------------------

    def _build_progress(
        self,
        epoch: int,
        global_step: int,
        total_steps: int,
        loss: float,
        elapsed: float,
        checkpoint_saved: bool = False,
    ) -> TrainingProgress:
        recent_losses = self.loss_history[-10:] if self.loss_history else [loss]
        avg_recent = sum(recent_losses) / len(recent_losses)

        steps_remaining = max(total_steps - global_step, 0)
        rate = global_step / elapsed if elapsed > 0 else 0
        eta = steps_remaining / rate if rate > 0 else 0.0

        return TrainingProgress(
            epoch=epoch + 1,
            total_epochs=self.config.epochs,
            step=global_step,
            total_steps=total_steps,
            loss=loss,
            avg_loss_last_10=avg_recent,
            elapsed_sec=elapsed,
            eta_sec=eta,
            checkpoint_saved=checkpoint_saved,
        )

    def _build_report(self, final_path: str) -> dict:
        return {
            "status": "completed",
            "mode": self.config.mode,
            "final_model_path": final_path,
            "epochs_trained": self.config.epochs,
            "final_loss": round(self.loss_history[-1], 4) if self.loss_history else None,
            "best_loss": round(min(self.loss_history), 4) if self.loss_history else None,
            "loss_history": [round(l, 4) for l in self.loss_history],
        }


# ---------------------------------------------------------------------------
# Quick CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def print_progress(p: TrainingProgress):
        bar_len = 30
        filled = int(bar_len * p.percent / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\r  [{bar}] {p.percent}% | loss: {p.loss:.4f} | "
            f"ETA: {p.to_dict()['eta_human']}",
            end="",
        )

    config = TrainingConfig(
        base_model_path="models/f5_tts_base.pt",
        filelist_path="dataset/filelist.txt",
        output_dir="models/user_voice",
        speaker_embedding_path="voice_profile/speaker.npy",
        mode="lora",
        epochs=100,
    )

    trainer = F5TTSTrainer(config)

    if trainer.resume_check():
        print("Resuming previous training session...\n")

    report = trainer.train(on_progress=print_progress)
    print("\n\nFinal report:")
    print(json.dumps({k: v for k, v in report.items() if k != "loss_history"}, indent=2))
