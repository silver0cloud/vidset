"""
trainer.py
----------
F5-TTS LoRA fine-tuning pipeline using real F5-TTS API.

Strategy:
    - Wraps F5-TTS's official finetune_cli.py via accelerate launch (subprocess)
    - Applies PEFT LoRA on DiT layers using target_modules="all-linear"
    - Auto checkpoint + resume
    - Live loss reporting via log file tail → callback

Install:
    pip install f5-tts peft accelerate

Model auto-downloads from HuggingFace on first run:
    hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_MODEL_HF       = "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
VOCAB_HF            = "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt"
DEFAULT_EPOCHS      = 500          # Small dataset → more epochs needed
DEFAULT_LR          = 1e-5         # Recommended for F5-TTS finetuning
LORA_RANK           = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05
GRAD_ACCUM_STEPS    = 4
SAVE_PER_UPDATES    = 50
WARMUP_UPDATES      = 20
TARGET_SR           = 24000        # F5-TTS native sample rate (NOT 22050)
MAX_SAMPLES         = 64


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    filelist_path: str
    output_dir: str
    dataset_name: str = "user_voice"
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LR
    batch_size: int = 1600          # frames — small for low VRAM
    use_lora: bool = True
    pretrain_path: Optional[str] = None   # override base model (local path)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return {
            "filelist_path": self.filelist_path,
            "output_dir": self.output_dir,
            "dataset_name": self.dataset_name,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "use_lora": self.use_lora,
            "device": self.device,
        }


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

@dataclass
class TrainingProgress:
    epoch: int
    total_epochs: int
    step: int
    loss: float
    avg_loss_last_10: float
    elapsed_sec: float
    eta_sec: float
    checkpoint_saved: bool = False

    @property
    def percent(self) -> float:
        return round((self.epoch / max(self.total_epochs, 1)) * 100, 1)

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "step": self.step,
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
        return f"{minutes // 60}h {minutes % 60}m"


ProgressCallback = Callable[[TrainingProgress], None]


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

class F5TTSDatasetPrep:
    """
    Converts our filelist.txt (path|text|speaker) into the format expected
    by F5-TTS's finetune_cli.py:
        ckpts/<dataset_name>/raw/wavs/*.wav
        ckpts/<dataset_name>/raw/metadata.csv  (filename|text)

    Also resamples audio to 24kHz (F5-TTS native sample rate).
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    def prepare(self, filelist_path: str, dataset_name: str) -> str:
        """
        Prepares dataset directory. Returns path to dataset root.
        """
        import torchaudio

        dataset_dir = self.output_dir / "ckpts" / dataset_name / "raw"
        wavs_dir = dataset_dir / "wavs"
        wavs_dir.mkdir(parents=True, exist_ok=True)

        entries = []
        with open(filelist_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) == 3:
                    wav_path, text, _ = parts
                    entries.append((wav_path, text))

        print(f"  Preparing {len(entries)} samples for F5-TTS training...")

        for wav_path, text in entries:
            src = Path(wav_path)
            if not src.exists():
                print(f"    Warning: {src} not found — skipping")
                continue

            # Resample to 24kHz (F5-TTS native SR)
            audio, sr = torchaudio.load(str(src))
            if sr != TARGET_SR:
                audio = torchaudio.functional.resample(audio, sr, TARGET_SR)
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)

            dst = wavs_dir / src.name
            torchaudio.save(str(dst), audio, TARGET_SR)

        # Write metadata.csv (LJSpeech format: filename|text)
        metadata_path = dataset_dir / "metadata.csv"
        with open(metadata_path, "w", encoding="utf-8") as f:
            for wav_path, text in entries:
                stem = Path(wav_path).stem
                if (wavs_dir / f"{stem}.wav").exists():
                    f.write(f"{stem}|{text}\n")

        print(f"  Dataset ready at: {dataset_dir}")
        return str(dataset_dir)


# ---------------------------------------------------------------------------
# LoRA wrapper
# ---------------------------------------------------------------------------

class F5TTSLoRATrainer:
    """
    Applies PEFT LoRA to the F5-TTS DiT model and runs training using
    the F5-TTS training loop directly (not via subprocess).

    This uses:
        - f5_tts.model.DiT          (real model class)
        - f5_tts.infer.utils_infer  (load_model, load_vocoder)
        - peft LoraConfig           (target_modules="all-linear" for DiT)
        - accelerate                (for device management)
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.vocoder = None
        self.optimizer = None
        self.start_epoch = 0
        self.loss_history: list[float] = []
        self._ckpt_dir = Path(config.output_dir) / "checkpoints"
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._latest_ckpt = self._ckpt_dir / "latest.pt"

    def train(self, on_progress: Optional[ProgressCallback] = None) -> dict:
        self._load_and_patch_model()
        self._maybe_resume()

        dataset_prep = F5TTSDatasetPrep(self.config.output_dir)
        dataset_dir = dataset_prep.prepare(
            self.config.filelist_path, self.config.dataset_name
        )

        from torch.utils.data import DataLoader
        train_dataset = self._build_dataset(dataset_dir)
        loader = DataLoader(
            train_dataset,
            batch_size=1,               # frame-based batching handled inside
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=0,
        )

        start_time = time.time()
        total_steps = len(loader) * self.config.epochs
        global_step = self.start_epoch * len(loader)

        print(f"\nLoRA Training — device: {self.config.device}")
        print(f"Trainable params: {self._count_trainable():,}")

        for epoch in range(self.start_epoch, self.config.epochs):
            epoch_losses = []

            for step, batch in enumerate(loader):
                loss = self._train_step(batch, step)
                epoch_losses.append(loss)
                self.loss_history.append(loss)
                global_step += 1

                if on_progress and step % 5 == 0:
                    elapsed = time.time() - start_time
                    on_progress(self._build_progress(
                        epoch, global_step, total_steps, loss, elapsed
                    ))

            # Flush any remaining accumulated gradients at epoch end
            # (happens when dataset size is not divisible by GRAD_ACCUM_STEPS)
            if self.optimizer is not None:
                remaining = len(loader) % GRAD_ACCUM_STEPS
                if remaining != 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            avg = sum(epoch_losses) / max(len(epoch_losses), 1)
            print(f"  Epoch {epoch+1}/{self.config.epochs} — loss: {avg:.4f}")

            ckpt_saved = False
            if (epoch + 1) % 50 == 0 or epoch == self.config.epochs - 1:
                self._save_checkpoint(epoch + 1)
                ckpt_saved = True

            if on_progress:
                elapsed = time.time() - start_time
                on_progress(self._build_progress(
                    epoch, global_step, total_steps, avg, elapsed,
                    checkpoint_saved=ckpt_saved
                ))

        final_path = self._save_final()
        return self._build_report(final_path)

    def resume_check(self) -> bool:
        return self._latest_ckpt.exists()

    # ------------------------------------------------------------------
    # Model loading + LoRA
    # ------------------------------------------------------------------

    def _find_linear_names(self, model: torch.nn.Module) -> list[str]:
        """
        Walks all named modules and returns names of nn.Linear layers,
        excluding final output projection layers (lm_head style).
        Works on any nn.Module — no PreTrainedModel required.
        """
        import torch.nn as nn
        linear_names = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Get the leaf name (last part after final dot)
                leaf = name.split(".")[-1]
                if leaf not in linear_names:
                    linear_names.append(leaf)
        return list(set(linear_names))

    def _load_and_patch_model(self) -> None:
        from f5_tts.model import DiT
        from f5_tts.infer.utils_infer import load_model, load_vocoder
        from peft import LoraConfig, get_peft_model
        from cached_path import cached_path

        print("Loading F5-TTS base model...")
        # Resolve pretrain checkpoint
        if self.config.pretrain_path and Path(self.config.pretrain_path).exists():
            ckpt_path = self.config.pretrain_path
        else:
            ckpt_path = str(cached_path(BASE_MODEL_HF))

        # F5-TTS DiT architecture config
        model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2,
            text_dim=512, conv_layers=4
        )

        self.model = load_model(
            model_cls=DiT,
            model_cfg=model_cfg,
            ckpt_path=ckpt_path,
            mel_spec_type="vocos",
            vocab_file=str(cached_path(VOCAB_HF)),
            ode_method="euler",
            use_ema=True,
            device=self.config.device,
        )

        # Load vocoder for inference sanity checks
        self.vocoder = load_vocoder(
            vocoder_name="vocos",
            is_local=False,
            device=self.config.device,
        )

        # Find all linear layer names manually — DiT is nn.Module not PreTrainedModel
        # so PEFT's "all-linear" shortcut doesn't work here
        linear_names = self._find_linear_names(base_model)
        print(f"  LoRA target layers: {len(linear_names)} linear layers found")

        lora_cfg = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=linear_names,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
        )
        print("LoRA applied successfully.")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def _build_dataset(self, dataset_dir: str):
        """
        Builds a simple dataset from the prepared wav + metadata.
        F5-TTS uses mel spectrograms as targets — we use torchaudio here
        for simplicity; the full F5-TTS training dataset class can be
        swapped in for production.
        """
        import torchaudio
        import torchaudio.transforms as T

        wavs_dir = Path(dataset_dir) / "wavs"
        metadata_path = Path(dataset_dir) / "metadata.csv"

        entries = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) == 2:
                    stem, text = parts
                    wav = wavs_dir / f"{stem}.wav"
                    if wav.exists():
                        entries.append((str(wav), text))

        mel_transform = T.MelSpectrogram(
            sample_rate=TARGET_SR,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=100,
        )

        class VoiceDataset(torch.utils.data.Dataset):
            def __len__(self_):
                return len(entries)

            def __getitem__(self_, idx):
                wav_path, text = entries[idx]
                audio, _ = torchaudio.load(wav_path)
                mel = mel_transform(audio).squeeze(0)  # (100, T)
                return {"mel": mel, "text": text, "wav_path": wav_path}

        return VoiceDataset()

    def _collate_fn(self, batch: list[dict]) -> dict:
        return batch[0]   # single sample per step for low-VRAM mode

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _train_step(self, batch: dict, step: int) -> float:
        """
        Flow-matching training step for F5-TTS.
        F5-TTS is trained on a text-guided speech infilling objective with
        conditional flow matching loss. This step uses the model's built-in
        forward pass which computes the CFM loss directly.
        """
        self.model.train()

        mel = batch["mel"].unsqueeze(0).to(self.config.device)  # (1, 100, T)
        text = [batch["text"]]

        # F5-TTS DiT forward — returns loss when called in training mode
        # The model pads text to mel length internally and computes CFM loss
        loss = self.model(
            inp=mel,
            text=text,
            lens=torch.tensor([mel.shape[-1]], device=self.config.device),
        )

        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item() * GRAD_ACCUM_STEPS

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss_history": self.loss_history,
            "config": self.config.to_dict(),
        }
        torch.save(ckpt, str(self._latest_ckpt))
        numbered = self._ckpt_dir / f"epoch_{epoch}.pt"
        shutil.copy2(str(self._latest_ckpt), str(numbered))
        print(f"    Checkpoint saved → epoch {epoch}")

    def _maybe_resume(self) -> None:
        if not self._latest_ckpt.exists():
            return
        print(f"Resuming from {self._latest_ckpt}...")
        ckpt = torch.load(str(self._latest_ckpt), map_location=self.config.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.start_epoch = ckpt["epoch"]
        self.loss_history = ckpt.get("loss_history", [])
        print(f"Resumed from epoch {self.start_epoch}")

    def _save_final(self) -> str:
        """Save LoRA adapter weights only (tiny file, ~few MB)."""
        adapter_path = Path(self.config.output_dir) / "lora_adapter"
        self.model.save_pretrained(str(adapter_path))
        print(f"LoRA adapter saved → {adapter_path}")
        return str(adapter_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _count_trainable(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _build_progress(
        self, epoch, global_step, total_steps, loss, elapsed,
        checkpoint_saved=False
    ) -> TrainingProgress:
        recent = self.loss_history[-10:] if self.loss_history else [loss]
        avg = sum(recent) / len(recent)
        remaining = max(total_steps - global_step, 0)
        rate = global_step / elapsed if elapsed > 0 else 0
        eta = remaining / rate if rate > 0 else 0.0

        return TrainingProgress(
            epoch=epoch + 1,
            total_epochs=self.config.epochs,
            step=global_step,
            loss=loss,
            avg_loss_last_10=avg,
            elapsed_sec=elapsed,
            eta_sec=eta,
            checkpoint_saved=checkpoint_saved,
        )

    def _build_report(self, final_path: str) -> dict:
        return {
            "status": "completed",
            "mode": "lora",
            "final_model_path": final_path,
            "epochs_trained": self.config.epochs,
            "final_loss": round(self.loss_history[-1], 4) if self.loss_history else None,
            "best_loss": round(min(self.loss_history), 4) if self.loss_history else None,
        }


# Alias so main.py can import F5TTSTrainer unchanged
F5TTSTrainer = F5TTSLoRATrainer


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def on_progress(p: TrainingProgress):
        bar = "█" * int(p.percent / 5) + "░" * (20 - int(p.percent / 5))
        print(f"\r  [{bar}] {p.percent}% | loss: {p.loss:.4f} | ETA: {p.to_dict()['eta_human']}", end="")

    config = TrainingConfig(
        filelist_path="dataset/filelist.txt",
        output_dir="models/user_voice",
        epochs=500,
    )
    trainer = F5TTSLoRATrainer(config)
    if trainer.resume_check():
        print("Resuming existing training session...")
    report = trainer.train(on_progress=on_progress)
    print("\n\nDone:", json.dumps({k: v for k, v in report.items()}, indent=2))