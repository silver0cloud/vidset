"""
validator.py
------------
Real-time audio quality validation for recorded takes.
Runs automatically on release, returns soft warnings (never hard rejects).

Dependencies:
    pip install numpy soundfile
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Constants — tunable thresholds
# ---------------------------------------------------------------------------

MIN_DURATION_SEC = 1.5       # Too short to be a real sentence
MAX_DURATION_SEC = 25.0      # Unreasonably long take
MIN_RMS_DB = -40.0           # Below this = too quiet / mic not working
MAX_PEAK_DB = -1.0           # Above this = clipping
MAX_NOISE_FLOOR_DB = -30.0   # Background noise ceiling
MAX_SILENCE_RATIO = 0.60     # >60% silence = too many pauses / dead air
MIN_SILENCE_RATIO = 0.05     # <5% silence = no natural pauses at all
SILENCE_THRESHOLD_DB = -50.0 # Frame RMS below this = silence


# ---------------------------------------------------------------------------
# Warning types
# ---------------------------------------------------------------------------

class WarnCode(str, Enum):
    TOO_SHORT        = "too_short"
    TOO_LONG         = "too_long"
    TOO_QUIET        = "too_quiet"
    CLIPPING         = "clipping"
    NOISY_BACKGROUND = "noisy_background"
    TOO_MUCH_SILENCE = "too_much_silence"
    TOO_LITTLE_PAUSE = "too_little_pause"


WARN_MESSAGES: dict[WarnCode, str] = {
    WarnCode.TOO_SHORT:        "Recording is too short. Make sure you read the full sentence.",
    WarnCode.TOO_LONG:         "Recording seems unusually long. Did something go wrong?",
    WarnCode.TOO_QUIET:        "Your voice is too quiet. Move closer to the microphone.",
    WarnCode.CLIPPING:         "Audio is clipping (too loud). Move slightly away from the mic or lower input gain.",
    WarnCode.NOISY_BACKGROUND: "Background noise detected. Try recording in a quieter space.",
    WarnCode.TOO_MUCH_SILENCE: "Too many silent gaps. Speak more continuously and avoid long pauses.",
    WarnCode.TOO_LITTLE_PAUSE: "No natural pauses detected. Remember to pause at commas and periods.",
}


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class ValidationWarning:
    code: WarnCode
    message: str
    severity: str   # "low" | "medium" | "high"

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass
class AudioStats:
    duration_sec: float
    rms_db: float
    peak_db: float
    noise_floor_db: float
    silence_ratio: float
    sample_rate: int

    def to_dict(self) -> dict:
        return {
            "duration_sec": round(self.duration_sec, 2),
            "rms_db": round(self.rms_db, 2),
            "peak_db": round(self.peak_db, 2),
            "noise_floor_db": round(self.noise_floor_db, 2),
            "silence_ratio": round(self.silence_ratio, 3),
            "sample_rate": self.sample_rate,
        }


@dataclass
class ValidationResult:
    take_id: str
    passed: bool                          # True = no warnings at all
    warnings: list[ValidationWarning] = field(default_factory=list)
    stats: Optional[AudioStats] = None

    @property
    def has_high_severity(self) -> bool:
        return any(w.severity == "high" for w in self.warnings)

    @property
    def warning_codes(self) -> list[str]:
        return [w.code.value for w in self.warnings]

    def to_dict(self) -> dict:
        return {
            "take_id": self.take_id,
            "passed": self.passed,
            "has_high_severity": self.has_high_severity,
            "warnings": [w.to_dict() for w in self.warnings],
            "stats": self.stats.to_dict() if self.stats else None,
        }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class TakeValidator:
    """
    Validates a recorded .wav take and returns soft warnings.

    Usage:
        validator = TakeValidator()
        result = validator.validate(take_id="S001_take1", file_path="recordings/S001/S001_take1.wav")
    """

    def validate(self, take_id: str, file_path: str) -> ValidationResult:
        """Run all checks. Always returns a result — never raises."""
        path = Path(file_path)

        if not path.exists():
            return ValidationResult(
                take_id=take_id,
                passed=False,
                warnings=[
                    ValidationWarning(
                        code=WarnCode.TOO_SHORT,
                        message="Audio file not found.",
                        severity="high",
                    )
                ],
            )

        try:
            audio, sr = sf.read(str(path), dtype="float32")
        except Exception as e:
            return ValidationResult(
                take_id=take_id,
                passed=False,
                warnings=[
                    ValidationWarning(
                        code=WarnCode.TOO_SHORT,
                        message=f"Could not read audio file: {e}",
                        severity="high",
                    )
                ],
            )

        # Mono flatten
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        stats = self._compute_stats(audio, sr)
        warnings = self._run_checks(stats)

        return ValidationResult(
            take_id=take_id,
            passed=len(warnings) == 0,
            warnings=warnings,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Stats computation
    # ------------------------------------------------------------------

    def _compute_stats(self, audio: np.ndarray, sr: int) -> AudioStats:
        duration = len(audio) / sr

        # RMS over full signal
        rms = float(np.sqrt(np.mean(audio ** 2)))
        rms_db = self._to_db(rms)

        # Peak
        peak = float(np.max(np.abs(audio)))
        peak_db = self._to_db(peak)

        # Noise floor — estimate from quietest 10% of frames
        frame_size = sr // 10   # 100ms frames
        frames = [
            audio[i: i + frame_size]
            for i in range(0, len(audio) - frame_size, frame_size)
        ]
        frame_rms = [float(np.sqrt(np.mean(f ** 2))) for f in frames if len(f) == frame_size]
        frame_rms.sort()
        quiet_frames = frame_rms[: max(1, len(frame_rms) // 10)]
        noise_floor_db = self._to_db(float(np.mean(quiet_frames))) if quiet_frames else -80.0

        # Silence ratio
        silence_threshold = self._from_db(SILENCE_THRESHOLD_DB)
        silent_samples = np.sum(np.abs(audio) < silence_threshold)
        silence_ratio = float(silent_samples / len(audio))

        return AudioStats(
            duration_sec=duration,
            rms_db=rms_db,
            peak_db=peak_db,
            noise_floor_db=noise_floor_db,
            silence_ratio=silence_ratio,
            sample_rate=sr,
        )

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def _run_checks(self, stats: AudioStats) -> list[ValidationWarning]:
        warnings: list[ValidationWarning] = []

        # Duration
        if stats.duration_sec < MIN_DURATION_SEC:
            warnings.append(self._warn(WarnCode.TOO_SHORT, "high"))

        elif stats.duration_sec > MAX_DURATION_SEC:
            warnings.append(self._warn(WarnCode.TOO_LONG, "low"))

        # Volume
        if stats.rms_db < MIN_RMS_DB:
            warnings.append(self._warn(WarnCode.TOO_QUIET, "high"))

        # Clipping
        if stats.peak_db > MAX_PEAK_DB:
            warnings.append(self._warn(WarnCode.CLIPPING, "high"))

        # Background noise
        if stats.noise_floor_db > MAX_NOISE_FLOOR_DB:
            warnings.append(self._warn(WarnCode.NOISY_BACKGROUND, "medium"))

        # Silence ratio
        if stats.silence_ratio > MAX_SILENCE_RATIO:
            warnings.append(self._warn(WarnCode.TOO_MUCH_SILENCE, "medium"))

        elif stats.silence_ratio < MIN_SILENCE_RATIO:
            warnings.append(self._warn(WarnCode.TOO_LITTLE_PAUSE, "low"))

        return warnings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _warn(self, code: WarnCode, severity: str) -> ValidationWarning:
        return ValidationWarning(
            code=code,
            message=WARN_MESSAGES[code],
            severity=severity,
        )

    @staticmethod
    def _to_db(value: float) -> float:
        if value <= 0:
            return -80.0
        return float(20 * np.log10(max(value, 1e-9)))

    @staticmethod
    def _from_db(db: float) -> float:
        return float(10 ** (db / 20))


# ---------------------------------------------------------------------------
# Batch validator — validate all takes in a session
# ---------------------------------------------------------------------------

class SessionValidator:
    """
    Runs TakeValidator across all takes in a session.json file.
    Useful for post-session review before training.
    """

    def __init__(self):
        self.validator = TakeValidator()

    def validate_session(self, session_json_path: str) -> dict:
        with open(session_json_path, "r") as f:
            session_data = json.load(f)

        results = {}
        for sid, sdata in session_data.get("sessions", {}).items():
            results[sid] = []
            for take in sdata.get("takes", []):
                result = self.validator.validate(
                    take_id=take["take_id"],
                    file_path=take["file_path"],
                )
                results[sid].append(result.to_dict())

        total_takes = sum(len(v) for v in results.values())
        clean_takes = sum(
            1
            for takes in results.values()
            for t in takes
            if t["passed"]
        )

        return {
            "summary": {
                "total_takes": total_takes,
                "clean_takes": clean_takes,
                "flagged_takes": total_takes - clean_takes,
            },
            "results": results,
        }


# ---------------------------------------------------------------------------
# Quick CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python validator.py <path_to_wav>")
        sys.exit(1)

    wav_path = sys.argv[1]
    validator = TakeValidator()
    result = validator.validate(take_id="test_take", file_path=wav_path)

    print(json.dumps(result.to_dict(), indent=2))
