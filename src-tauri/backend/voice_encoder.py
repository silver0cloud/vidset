"""
voice_encoder.py
----------------
Extracts a speaker embedding vector from the user's cleaned takes
and verifies voice consistency across all recordings.

Pipeline:
    1. Load all cleaned takes from dataset/wavs/
    2. Extract per-take embeddings via resemblyzer
    3. Average into a single speaker embedding
    4. Compute pairwise cosine similarity for consistency check
    5. Save speaker.npy + voice_profile.json

Dependencies:
    pip install resemblyzer numpy soundfile
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESEMBLYZER_SR      = 16000     # Resemblyzer expects 16kHz
EMBEDDING_DIM       = 256       # Resemblyzer output dimension
MIN_CONSISTENCY     = 0.75      # Cosine similarity floor for a healthy profile
WARN_CONSISTENCY    = 0.85      # Below this → soft warning
MIN_TAKES_REQUIRED  = 10        # Minimum takes to build a reliable embedding


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TakeEmbedding:
    take_id: str
    wav_path: str
    embedding: np.ndarray       # shape: (256,)
    duration_sec: float

    def to_dict(self) -> dict:
        return {
            "take_id": self.take_id,
            "wav_path": self.wav_path,
            "duration_sec": round(self.duration_sec, 2),
            "embedding_norm": round(float(np.linalg.norm(self.embedding)), 4),
        }


@dataclass
class ConsistencyWarning:
    take_id: str
    similarity: float
    message: str

    def to_dict(self) -> dict:
        return {
            "take_id": self.take_id,
            "similarity": round(self.similarity, 4),
            "message": self.message,
        }


@dataclass
class VoiceProfile:
    speaker_id: str
    embedding_path: str             # path to speaker.npy
    mean_consistency: float         # avg pairwise cosine similarity
    min_consistency: float
    total_takes: int
    total_duration_sec: float
    warnings: list[ConsistencyWarning] = field(default_factory=list)
    ready_for_training: bool = True

    @property
    def total_duration_min(self) -> float:
        return round(self.total_duration_sec / 60, 2)

    def to_dict(self) -> dict:
        return {
            "speaker_id": self.speaker_id,
            "embedding_path": self.embedding_path,
            "consistency": {
                "mean": round(self.mean_consistency, 4),
                "min": round(self.min_consistency, 4),
                "threshold": MIN_CONSISTENCY,
                "status": self._consistency_status(),
            },
            "dataset": {
                "total_takes": self.total_takes,
                "total_duration_sec": round(self.total_duration_sec, 2),
                "total_duration_min": self.total_duration_min,
            },
            "warnings": [w.to_dict() for w in self.warnings],
            "ready_for_training": self.ready_for_training,
        }

    def _consistency_status(self) -> str:
        if self.mean_consistency >= WARN_CONSISTENCY:
            return "excellent"
        elif self.mean_consistency >= MIN_CONSISTENCY:
            return "acceptable"
        else:
            return "poor"


# ---------------------------------------------------------------------------
# Voice Encoder
# ---------------------------------------------------------------------------

class VoiceEncoder:
    """
    Extracts speaker embeddings from cleaned wav files using resemblyzer.

    Usage:
        encoder = VoiceEncoder(output_dir="voice_profile")
        profile = encoder.encode_from_dataset(dataset_dir="dataset")
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._encoder = None    # Lazy load — resemblyzer is slow to import

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def encode_from_dataset(
        self,
        dataset_dir: str,
        speaker_id: str = "speaker_0",
    ) -> VoiceProfile:
        """
        Main entry point.
        Reads all wavs from dataset/wavs/, builds speaker embedding + profile.
        """
        wavs_dir = Path(dataset_dir) / "wavs"
        wav_files = sorted(wavs_dir.glob("*.wav"))

        if len(wav_files) == 0:
            raise FileNotFoundError(f"No .wav files found in {wavs_dir}")

        print(f"  Found {len(wav_files)} wav files. Extracting embeddings...")

        self._load_encoder()

        # Extract per-take embeddings
        take_embeddings: list[TakeEmbedding] = []
        for wav_path in wav_files:
            te = self._encode_single(wav_path)
            if te:
                take_embeddings.append(te)
                print(f"    ✓ {wav_path.name}")
            else:
                print(f"    ✗ {wav_path.name} — skipped (too short or unreadable)")

        if len(take_embeddings) < MIN_TAKES_REQUIRED:
            raise ValueError(
                f"Only {len(take_embeddings)} valid takes found. "
                f"Minimum required: {MIN_TAKES_REQUIRED}."
            )

        # Build mean speaker embedding
        stacked    = np.stack([te.embedding for te in take_embeddings], axis=0)
        mean_emb   = stacked.mean(axis=0)
        mean_emb   = mean_emb / (np.linalg.norm(mean_emb) + 1e-9)  # L2 normalize

        # Save embedding
        emb_path = self.output_dir / "speaker.npy"
        np.save(str(emb_path), mean_emb)
        print(f"\n  Speaker embedding saved → {emb_path}")

        # Consistency analysis
        warnings, mean_sim, min_sim = self._analyze_consistency(
            take_embeddings, mean_emb
        )

        ready = min_sim >= MIN_CONSISTENCY

        profile = VoiceProfile(
            speaker_id=speaker_id,
            embedding_path=str(emb_path),
            mean_consistency=mean_sim,
            min_consistency=min_sim,
            total_takes=len(take_embeddings),
            total_duration_sec=sum(te.duration_sec for te in take_embeddings),
            warnings=warnings,
            ready_for_training=ready,
        )

        # Save profile
        profile_path = self.output_dir / "voice_profile.json"
        with open(profile_path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
        print(f"  Voice profile saved   → {profile_path}")

        self._print_summary(profile)
        return profile

    def encode_single_take(self, wav_path: str) -> Optional[np.ndarray]:
        """Encode a single wav file. Returns embedding or None."""
        self._load_encoder()
        te = self._encode_single(Path(wav_path))
        return te.embedding if te else None

    def similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        return float(np.dot(emb_a, emb_b) / (
            np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-9
        ))

    def load_speaker_embedding(self, profile_dir: str) -> np.ndarray:
        """Load a saved speaker.npy embedding."""
        path = Path(profile_dir) / "speaker.npy"
        if not path.exists():
            raise FileNotFoundError(f"speaker.npy not found in {profile_dir}")
        return np.load(str(path))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_encoder(self) -> None:
        """Lazy-load resemblyzer encoder (slow first import)."""
        if self._encoder is None:
            from resemblyzer import VoiceEncoder as ResemblyzerEncoder
            self._encoder = ResemblyzerEncoder()

    def _encode_single(self, wav_path: Path) -> Optional[TakeEmbedding]:
        """Extract embedding from a single wav file."""
        try:
            from resemblyzer import preprocess_wav

            audio, sr = sf.read(str(wav_path), dtype="float32")

            # Mono flatten
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            duration = len(audio) / sr

            # Resemblyzer needs at least ~1.6s
            if duration < 1.6:
                return None

            # Resemblyzer handles its own resampling via preprocess_wav
            processed = preprocess_wav(audio, source_sr=sr)
            embedding  = self._encoder.embed_utterance(processed)

            return TakeEmbedding(
                take_id=wav_path.stem,
                wav_path=str(wav_path),
                embedding=embedding,
                duration_sec=duration,
            )

        except Exception as e:
            print(f"    Warning: could not encode {wav_path.name} — {e}")
            return None

    def _analyze_consistency(
        self,
        takes: list[TakeEmbedding],
        mean_emb: np.ndarray,
    ) -> tuple[list[ConsistencyWarning], float, float]:
        """
        Compute cosine similarity of each take against the mean embedding.
        Flag outliers as soft warnings.
        """
        similarities: list[float] = []
        warnings: list[ConsistencyWarning] = []

        for te in takes:
            sim = self.similarity(te.embedding, mean_emb)
            similarities.append(sim)

            if sim < MIN_CONSISTENCY:
                warnings.append(ConsistencyWarning(
                    take_id=te.take_id,
                    similarity=sim,
                    message=(
                        f"This take sounds significantly different from your other recordings "
                        f"(similarity: {sim:.2f}). Consider re-recording this sentence."
                    ),
                ))
            elif sim < WARN_CONSISTENCY:
                warnings.append(ConsistencyWarning(
                    take_id=te.take_id,
                    similarity=sim,
                    message=(
                        f"This take is slightly inconsistent with your voice profile "
                        f"(similarity: {sim:.2f}). Acceptable but worth reviewing."
                    ),
                ))

        mean_sim = float(np.mean(similarities))
        min_sim  = float(np.min(similarities))

        return warnings, mean_sim, min_sim

    def _print_summary(self, profile: VoiceProfile) -> None:
        s = profile.to_dict()
        print("\n" + "=" * 50)
        print("  Voice Profile Built!")
        print(f"  Takes         : {s['dataset']['total_takes']}")
        print(f"  Duration      : {s['dataset']['total_duration_min']} minutes")
        print(f"  Consistency   : {s['consistency']['mean']} ({s['consistency']['status']})")
        print(f"  Ready         : {'✅ Yes' if profile.ready_for_training else '❌ No — re-record flagged takes'}")
        if profile.warnings:
            print(f"  Warnings      : {len(profile.warnings)} takes flagged")
        print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# Embedding comparator — verify a new recording matches saved profile
# ---------------------------------------------------------------------------

class VoiceVerifier:
    """
    Verifies that a new voice recording matches a saved speaker profile.
    Useful for: confirming user identity before inference,
                or checking if a re-recorded take fits the profile.

    Usage:
        verifier = VoiceVerifier(profile_dir="voice_profile")
        result   = verifier.verify(wav_path="new_recording.wav")
    """

    def __init__(self, profile_dir: str):
        self.encoder      = VoiceEncoder(output_dir=profile_dir)
        self.profile_dir  = Path(profile_dir)
        self._speaker_emb = None

    def verify(self, wav_path: str) -> dict:
        speaker_emb = self._load_speaker()
        new_emb     = self.encoder.encode_single_take(wav_path)

        if new_emb is None:
            return {
                "match": False,
                "similarity": 0.0,
                "message": "Could not encode the provided audio.",
            }

        sim = self.encoder.similarity(new_emb, speaker_emb)

        return {
            "match": sim >= MIN_CONSISTENCY,
            "similarity": round(sim, 4),
            "status": (
                "excellent" if sim >= WARN_CONSISTENCY
                else "acceptable" if sim >= MIN_CONSISTENCY
                else "poor"
            ),
            "message": (
                "Voice matches your profile." if sim >= MIN_CONSISTENCY
                else "Voice does not match your profile. Are you the same speaker?"
            ),
        }

    def _load_speaker(self) -> np.ndarray:
        if self._speaker_emb is None:
            self._speaker_emb = self.encoder.load_speaker_embedding(str(self.profile_dir))
        return self._speaker_emb


# ---------------------------------------------------------------------------
# Quick CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "encode"

    if mode == "encode":
        encoder = VoiceEncoder(output_dir="voice_profile")
        profile = encoder.encode_from_dataset(dataset_dir="dataset")
        print(json.dumps(profile.to_dict(), indent=2))

    elif mode == "verify":
        if len(sys.argv) < 3:
            print("Usage: python voice_encoder.py verify <path_to_wav>")
            sys.exit(1)
        verifier = VoiceVerifier(profile_dir="voice_profile")
        result   = verifier.verify(wav_path=sys.argv[2])
        print(json.dumps(result, indent=2))
