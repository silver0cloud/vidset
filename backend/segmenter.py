"""
segmenter.py
------------
Organizes cleaned takes into a structured dataset directory and generates
F5-TTS ready training manifests (metadata.csv + filelist.txt).

Output structure:
    dataset/
    ├── wavs/
    │   ├── S001_take2_cleaned.wav
    │   ├── S002_take1_cleaned.wav
    │   └── ...
    ├── metadata.csv          # LJSpeech-style: filename|text
    ├── filelist.txt          # F5-TTS filelist: path|text|speaker
    └── dataset_info.json     # Full dataset summary

Dependencies:
    pip install soundfile numpy
"""

import csv
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPEAKER_ID = "speaker_0"    # Default speaker ID for F5-TTS single-speaker


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DatasetEntry:
    sentence_id: str
    take_id: str
    text: str
    mood: str
    phonetic_focus: str
    wav_filename: str        # filename only, e.g. S001_take2_cleaned.wav
    wav_path: str            # absolute path in dataset/wavs/
    duration_sec: float
    sample_rate: int

    def to_dict(self) -> dict:
        return {
            "sentence_id": self.sentence_id,
            "take_id": self.take_id,
            "text": self.text,
            "mood": self.mood,
            "phonetic_focus": self.phonetic_focus,
            "wav_filename": self.wav_filename,
            "wav_path": self.wav_path,
            "duration_sec": round(self.duration_sec, 2),
            "sample_rate": self.sample_rate,
        }


@dataclass
class SegmentationReport:
    total_sentences: int
    total_entries: int
    total_duration_sec: float
    missing_sentences: list[str]
    entries: list[DatasetEntry] = field(default_factory=list)

    @property
    def total_duration_min(self) -> float:
        return round(self.total_duration_sec / 60, 2)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total_sentences_in_script": self.total_sentences,
                "total_entries_in_dataset": self.total_entries,
                "total_duration_sec": round(self.total_duration_sec, 2),
                "total_duration_min": self.total_duration_min,
                "missing_sentences": self.missing_sentences,
                "coverage_percent": round(
                    (self.total_entries / max(self.total_sentences, 1)) * 100, 1
                ),
            },
            "entries": [e.to_dict() for e in self.entries],
        }


# ---------------------------------------------------------------------------
# Segmenter
# ---------------------------------------------------------------------------

class DatasetSegmenter:
    """
    Builds a clean dataset directory from:
        - session.json        (recording session state)
        - cleaning_report.json (paths to cleaned .wav files)
        - curated_script.json  (sentence texts)

    Outputs:
        - dataset/wavs/       organized .wav files
        - dataset/metadata.csv         LJSpeech format
        - dataset/filelist.txt         F5-TTS format
        - dataset/dataset_info.json    full report

    Usage:
        segmenter = DatasetSegmenter(
            session_json="recordings/session.json",
            cleaning_report="recordings/cleaned/cleaning_report.json",
            script_json="curated_script.json",
            dataset_dir="dataset",
        )
        report = segmenter.build()
    """

    def __init__(
        self,
        session_json: str,
        cleaning_report: str,
        script_json: str,
        dataset_dir: str,
    ):
        self.session_json     = Path(session_json)
        self.cleaning_report  = Path(cleaning_report)
        self.script_json      = Path(script_json)
        self.dataset_dir      = Path(dataset_dir)
        self.wavs_dir         = self.dataset_dir / "wavs"

        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.wavs_dir.mkdir(parents=True, exist_ok=True)

        self._script    = self._load_script()
        self._session   = self._load_session()
        self._cleaned   = self._load_cleaning_report()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build(self) -> SegmentationReport:
        """Run full segmentation pipeline. Returns SegmentationReport."""
        entries: list[DatasetEntry] = []
        missing: list[str] = []

        for sentence in self._script:
            sid  = sentence["id"]
            text = sentence["text"]
            mood = sentence["mood"]
            focus = sentence["phonetic_focus"]

            # Get selected take id from session
            selected_take_id = self._get_selected_take(sid)
            if not selected_take_id:
                missing.append(sid)
                continue

            # Get cleaned file path
            cleaned_path = self._get_cleaned_path(selected_take_id)
            if not cleaned_path or not Path(cleaned_path).exists():
                missing.append(sid)
                continue

            # Copy wav to dataset/wavs/
            wav_filename = Path(cleaned_path).name
            dst_path     = self.wavs_dir / wav_filename
            shutil.copy2(cleaned_path, str(dst_path))

            # Read audio stats
            duration, sr = self._get_audio_info(str(dst_path))

            entry = DatasetEntry(
                sentence_id=sid,
                take_id=selected_take_id,
                text=text,
                mood=mood,
                phonetic_focus=focus,
                wav_filename=wav_filename,
                wav_path=str(dst_path),
                duration_sec=duration,
                sample_rate=sr,
            )
            entries.append(entry)

        # Write manifests
        self._write_metadata_csv(entries)
        self._write_filelist_txt(entries)

        report = SegmentationReport(
            total_sentences=len(self._script),
            total_entries=len(entries),
            total_duration_sec=sum(e.duration_sec for e in entries),
            missing_sentences=missing,
            entries=entries,
        )

        self._write_dataset_info(report)
        self._print_summary(report)

        return report

    # ------------------------------------------------------------------
    # Manifest writers
    # ------------------------------------------------------------------

    def _write_metadata_csv(self, entries: list[DatasetEntry]) -> None:
        """
        LJSpeech-style metadata.csv:
            filename|normalized_text|normalized_text
        Used by many TTS trainers as standard input format.
        """
        path = self.dataset_dir / "metadata.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="|")
            for e in entries:
                # LJSpeech format: no extension in filename column
                stem = Path(e.wav_filename).stem
                writer.writerow([stem, e.text, e.text])

        print(f"  metadata.csv → {path} ({len(entries)} entries)")

    def _write_filelist_txt(self, entries: list[DatasetEntry]) -> None:
        """
        F5-TTS filelist.txt format:
            /absolute/path/to/wav.wav|transcript text|speaker_id

        F5-TTS trainer expects absolute paths for reliable loading.
        """
        path = self.dataset_dir / "filelist.txt"
        with open(path, "w", encoding="utf-8") as f:
            for e in entries:
                abs_path = str(Path(e.wav_path).resolve())
                f.write(f"{abs_path}|{e.text}|{SPEAKER_ID}\n")

        print(f"  filelist.txt  → {path} ({len(entries)} entries)")

    def _write_dataset_info(self, report: SegmentationReport) -> None:
        path = self.dataset_dir / "dataset_info.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"  dataset_info  → {path}")

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_script(self) -> list[dict]:
        with open(self.script_json, "r", encoding="utf-8") as f:
            return json.load(f)["sentences"]

    def _load_session(self) -> dict:
        with open(self.session_json, "r", encoding="utf-8") as f:
            return json.load(f).get("sessions", {})

    def _load_cleaning_report(self) -> dict:
        """Build a lookup: take_id → cleaned_path"""
        with open(self.cleaning_report, "r", encoding="utf-8") as f:
            report = json.load(f)
        return {
            r["take_id"]: r["cleaned_path"]
            for r in report.get("results", [])
            if r["success"]
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_selected_take(self, sentence_id: str) -> Optional[str]:
        session = self._session.get(sentence_id)
        if not session:
            return None
        return session.get("selected_take_id")

    def _get_cleaned_path(self, take_id: str) -> Optional[str]:
        return self._cleaned.get(take_id)

    def _get_audio_info(self, wav_path: str) -> tuple[float, int]:
        try:
            info = sf.info(wav_path)
            return info.duration, info.samplerate
        except Exception:
            return 0.0, 24000

    def _print_summary(self, report: SegmentationReport) -> None:
        s = report.to_dict()["summary"]
        print("\n" + "=" * 50)
        print("  Dataset built successfully!")
        print(f"  Entries     : {s['total_entries_in_dataset']} / {s['total_sentences_in_script']}")
        print(f"  Duration    : {s['total_duration_min']} minutes")
        print(f"  Coverage    : {s['coverage_percent']}%")
        if report.missing_sentences:
            print(f"  Missing     : {', '.join(report.missing_sentences)}")
        print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# Manifest validator — verify filelist before training
# ---------------------------------------------------------------------------

class ManifestValidator:
    """
    Sanity-checks filelist.txt before handing off to trainer.
    Catches missing files, empty transcripts, bad sample rates.
    """

    def validate_filelist(self, filelist_path: str) -> dict:
        issues  = []
        valid   = 0
        path    = Path(filelist_path)

        if not path.exists():
            return {"valid": 0, "issues": [f"filelist.txt not found at {filelist_path}"]}

        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        for i, line in enumerate(lines):
            parts = line.split("|")
            if len(parts) != 3:
                issues.append(f"Line {i+1}: malformed (expected 3 pipe-separated fields)")
                continue

            wav_path, text, speaker = parts

            if not Path(wav_path).exists():
                issues.append(f"Line {i+1}: file not found → {wav_path}")
                continue

            if not text.strip():
                issues.append(f"Line {i+1}: empty transcript → {wav_path}")
                continue

            try:
                info = sf.info(wav_path)
                if info.samplerate != 24000:
                    issues.append(
                        f"Line {i+1}: sample rate {info.samplerate}Hz (expected 24000) → {wav_path}"
                    )
                    continue
            except Exception as e:
                issues.append(f"Line {i+1}: cannot read audio → {e}")
                continue

            valid += 1

        return {
            "total_lines": len(lines),
            "valid_entries": valid,
            "issues": issues,
            "ready_for_training": len(issues) == 0,
        }


# ---------------------------------------------------------------------------
# Quick CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "build"

    if mode == "build":
        segmenter = DatasetSegmenter(
            session_json="recordings/session.json",
            cleaning_report="recordings/cleaned/cleaning_report.json",
            script_json="curated_script.json",
            dataset_dir="dataset",
        )
        report = segmenter.build()
        print(json.dumps(report.to_dict()["summary"], indent=2))

    elif mode == "validate":
        validator = ManifestValidator()
        result    = validator.validate_filelist("dataset/filelist.txt")
        print(json.dumps(result, indent=2))