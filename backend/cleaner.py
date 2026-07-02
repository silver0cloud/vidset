"""
cleaner.py
----------
Light audio post-processing pipeline for selected final takes.
Runs only on user-selected takes before they enter the training pipeline.

Pipeline order:
    1. Load & mono flatten
    2. High-pass filter      — remove low-frequency rumble (< 80Hz)
    3. Trim silence          — strip leading/trailing dead air
    4. RMS normalization     — consistent volume across all takes
    5. Save cleaned file

Dependencies:
    pip install numpy soundfile scipy
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE         = 24000     # F5-TTS native sample rate — cleaner outputs at this rate
                                # recorder.py captures at 22050Hz; cleaner resamples on save
TARGET_RMS_DB       = -20.0     # Target loudness after normalization
HIGHPASS_CUTOFF_HZ  = 80        # Cut frequencies below this (rumble removal)
HIGHPASS_ORDER      = 4         # Butterworth filter order
SILENCE_TOP_DB      = 40        # librosa-style: silence = peak - 40dB
SILENCE_FRAME_MS    = 25        # Frame length for silence detection (ms)
SILENCE_HOP_MS      = 10        # Hop length for silence detection (ms)
MIN_SILENCE_PAD_MS  = 50        # Pad this much silence back after trimming


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class CleanResult:
    take_id: str
    original_path: str
    cleaned_path: str
    original_duration: float
    cleaned_duration: float
    steps_applied: list[str]
    success: bool
    error: Optional[str] = None

    @property
    def duration_delta(self) -> float:
        return round(self.original_duration - self.cleaned_duration, 3)

    def to_dict(self) -> dict:
        return {
            "take_id": self.take_id,
            "original_path": self.original_path,
            "cleaned_path": self.cleaned_path,
            "original_duration_sec": round(self.original_duration, 2),
            "cleaned_duration_sec": round(self.cleaned_duration, 2),
            "trimmed_sec": self.duration_delta,
            "steps_applied": self.steps_applied,
            "success": self.success,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Cleaner
# ---------------------------------------------------------------------------

class AudioCleaner:
    """
    Light post-processing pipeline for a single selected take.

    Usage:
        cleaner = AudioCleaner(cleaned_dir="recordings/cleaned")
        result = cleaner.clean(take_id="S001_take2", file_path="recordings/S001/S001_take2.wav")
    """

    def __init__(self, cleaned_dir: str):
        self.cleaned_dir = Path(cleaned_dir)
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)

    def clean(self, take_id: str, file_path: str) -> CleanResult:
        """Run full cleaning pipeline. Returns CleanResult."""
        src = Path(file_path)
        dst = self.cleaned_dir / f"{take_id}_cleaned.wav"
        steps: list[str] = []

        try:
            # 1. Load
            audio, sr = sf.read(str(src), dtype="float32")
            original_duration = len(audio) / sr

            # 2. Mono flatten
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
                steps.append("stereo_to_mono")

            # 2. Resample to 24kHz (F5-TTS native SR)
            if sr != SAMPLE_RATE:
                audio = self._resample(audio, sr, SAMPLE_RATE)
                sr = SAMPLE_RATE
                steps.append(f"resampled_to_{SAMPLE_RATE}hz")

            # 3. High-pass filter
            audio = self._highpass_filter(audio, sr)
            steps.append("highpass_filter_80hz")

            # 4. Trim silence
            audio = self._trim_silence(audio, sr)
            steps.append("silence_trimmed")

            # 5. RMS normalize
            audio = self._rms_normalize(audio)
            steps.append(f"rms_normalized_{TARGET_RMS_DB}db")

            # 6. Final clip guard — ensure no values exceed [-1, 1]
            audio = np.clip(audio, -1.0, 1.0)
            steps.append("clip_guard")

            # 7. Save
            sf.write(str(dst), audio, SAMPLE_RATE, subtype="PCM_16")
            cleaned_duration = len(audio) / SAMPLE_RATE

            return CleanResult(
                take_id=take_id,
                original_path=str(src),
                cleaned_path=str(dst),
                original_duration=original_duration,
                cleaned_duration=cleaned_duration,
                steps_applied=steps,
                success=True,
            )

        except Exception as e:
            # On failure, copy original as fallback so pipeline isn't blocked
            shutil.copy2(str(src), str(dst))
            return CleanResult(
                take_id=take_id,
                original_path=str(src),
                cleaned_path=str(dst),
                original_duration=0.0,
                cleaned_duration=0.0,
                steps_applied=steps,
                success=False,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Step 1 — High-pass filter
    # ------------------------------------------------------------------

    def _highpass_filter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Butterworth high-pass filter to remove low-frequency rumble
        (AC hum, desk vibrations, mic handling noise < 80Hz).
        """
        nyquist = sr / 2.0
        normalized_cutoff = HIGHPASS_CUTOFF_HZ / nyquist
        sos = butter(HIGHPASS_ORDER, normalized_cutoff, btype="high", output="sos")
        return sosfilt(sos, audio).astype(np.float32)

    # ------------------------------------------------------------------
    # Step 2 — Trim silence
    # ------------------------------------------------------------------

    def _trim_silence(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Trim leading and trailing silence using frame-based RMS.
        Adds a small pad after trimming so words aren't clipped.
        """
        frame_len = int(sr * SILENCE_FRAME_MS / 1000)
        hop_len   = int(sr * SILENCE_HOP_MS / 1000)
        pad_len   = int(sr * MIN_SILENCE_PAD_MS / 1000)

        # Compute per-frame RMS
        frames = [
            audio[i: i + frame_len]
            for i in range(0, len(audio) - frame_len, hop_len)
        ]
        if not frames:
            return audio

        frame_rms = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])

        peak_rms  = frame_rms.max()
        threshold = peak_rms * (10 ** (-SILENCE_TOP_DB / 20))

        # Find first and last non-silent frame
        non_silent = np.where(frame_rms > threshold)[0]
        if len(non_silent) == 0:
            return audio  # Fully silent — return as-is, validator will flag it

        start_frame = max(0, non_silent[0] - 1)
        end_frame   = min(len(frames) - 1, non_silent[-1] + 1)

        start_sample = max(0, start_frame * hop_len - pad_len)
        end_sample   = min(len(audio), end_frame * hop_len + frame_len + pad_len)

        return audio[start_sample:end_sample]

    # ------------------------------------------------------------------
    # Step 3 — RMS normalization
    # ------------------------------------------------------------------

    def _rms_normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to a target RMS level in dB.
        Ensures consistent loudness across all 35 takes for training.
        """
        current_rms = float(np.sqrt(np.mean(audio ** 2)))
        if current_rms < 1e-9:
            return audio  # Silent signal — skip

        target_rms = 10 ** (TARGET_RMS_DB / 20)
        gain = target_rms / current_rms
        return (audio * gain).astype(np.float32)

    # ------------------------------------------------------------------
    # Step 4 — Resample (fallback, should rarely be needed)
    # ------------------------------------------------------------------

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple linear resample. scipy.signal.resample_poly preferred if available."""
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(orig_sr, target_sr)
            return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)
        except Exception:
            # Fallback: numpy linear interpolation
            duration = len(audio) / orig_sr
            new_len  = int(duration * target_sr)
            return np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)


# ---------------------------------------------------------------------------
# Batch cleaner — process all selected takes from session.json
# ---------------------------------------------------------------------------

class SessionCleaner:
    """
    Reads session.json, cleans only selected takes, outputs cleaned files
    and a cleaning report (cleaning_report.json).

    Usage:
        cleaner = SessionCleaner(
            session_json="recordings/session.json",
            cleaned_dir="recordings/cleaned"
        )
        report = cleaner.run()
    """

    def __init__(self, session_json: str, cleaned_dir: str):
        self.session_json = Path(session_json)
        self.cleaner      = AudioCleaner(cleaned_dir=cleaned_dir)
        self.cleaned_dir  = Path(cleaned_dir)

    def run(self) -> dict:
        with open(self.session_json, "r") as f:
            session = json.load(f)

        results      = []
        skipped      = []

        for sid, sdata in session.get("sessions", {}).items():
            selected_id = sdata.get("selected_take_id")

            if not selected_id:
                skipped.append(sid)
                continue

            take = next(
                (t for t in sdata["takes"] if t["take_id"] == selected_id), None
            )
            if not take:
                skipped.append(sid)
                continue

            result = self.cleaner.clean(
                take_id=take["take_id"],
                file_path=take["file_path"],
            )
            results.append(result.to_dict())

        report = {
            "summary": {
                "total_processed": len(results),
                "successful": sum(1 for r in results if r["success"]),
                "failed": sum(1 for r in results if not r["success"]),
                "skipped_no_selection": len(skipped),
            },
            "skipped_sentences": skipped,
            "results": results,
        }

        report_path = self.cleaned_dir / "cleaning_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Cleaning report saved to {report_path}")
        return report


# ---------------------------------------------------------------------------
# Quick CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cleaner.py <path_to_wav>")
        sys.exit(1)

    wav_path = sys.argv[1]
    cleaner  = AudioCleaner(cleaned_dir="recordings/cleaned")
    result   = cleaner.clean(take_id="test_take", file_path=wav_path)

    print(json.dumps(result.to_dict(), indent=2))
