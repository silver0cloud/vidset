"""
recorder.py
-----------
Handles per-sentence audio recording with press-and-hold trigger,
multi-take management, and file persistence.

Dependencies:
    pip install pyaudio soundfile numpy
"""

import os
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudio
import soundfile as sf


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 22050       # Hz — matches F5-TTS expected input
CHANNELS = 1              # Mono
CHUNK = 1024              # Frames per buffer
FORMAT = pyaudio.paInt16  # 16-bit PCM
MAX_RECORD_SECONDS = 30   # Safety ceiling per take


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Take:
    take_id: str          # e.g. "S001_take1"
    sentence_id: str      # e.g. "S001"
    file_path: str        # absolute path to .wav
    duration_seconds: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "take_id": self.take_id,
            "sentence_id": self.sentence_id,
            "file_path": self.file_path,
            "duration_seconds": round(self.duration_seconds, 2),
            "timestamp": self.timestamp,
        }


@dataclass
class SentenceSession:
    sentence_id: str
    recording_name: str   # user-provided name
    takes: list[Take] = field(default_factory=list)
    selected_take_id: Optional[str] = None

    @property
    def take_count(self) -> int:
        return len(self.takes)

    def next_take_label(self) -> str:
        return f"{self.sentence_id}_take{self.take_count + 1}"

    def to_dict(self) -> dict:
        return {
            "sentence_id": self.sentence_id,
            "recording_name": self.recording_name,
            "selected_take_id": self.selected_take_id,
            "takes": [t.to_dict() for t in self.takes],
        }


# ---------------------------------------------------------------------------
# Recorder core
# ---------------------------------------------------------------------------

class Recorder:
    """
    Press-and-hold recorder.

    Usage (backend / programmatic):
        recorder = Recorder(output_dir="recordings/S001")
        recorder.start_hold()          # user presses button
        ...                            # user holds
        take = recorder.stop_hold()    # user releases
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._pa = pyaudio.PyAudio()
        self._stream: Optional[pyaudio.Stream] = None
        self._frames: list[bytes] = []
        self._recording = False
        self._lock = threading.Lock()
        self._record_thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_hold(self) -> None:
        """Call when user presses and holds the record button."""
        with self._lock:
            if self._recording:
                return
            self._frames = []
            self._recording = True
            self._start_time = time.time()

        self._stream = self._pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        self._record_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._record_thread.start()

    def stop_hold(self, take_id: str) -> Optional[Take]:
        """
        Call when user releases the record button.
        Returns a Take object with saved file path, or None if too short.
        """
        with self._lock:
            if not self._recording:
                return None
            self._recording = False

        elapsed = time.time() - self._start_time

        if self._record_thread:
            self._record_thread.join(timeout=2)

        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        if elapsed < 0.5:
            # Too short — likely accidental press
            return None

        file_path = self.output_dir / f"{take_id}.wav"
        self._save_wav(str(file_path))

        return Take(
            take_id=take_id,
            sentence_id=take_id.rsplit("_take", 1)[0],
            file_path=str(file_path),
            duration_seconds=elapsed,
        )

    def delete_take(self, take: Take) -> bool:
        """Delete a take's audio file from disk."""
        path = Path(take.file_path)
        if path.exists():
            path.unlink()
            return True
        return False

    def cleanup(self) -> None:
        """Release PyAudio resources."""
        self._pa.terminate()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Runs in a thread — reads mic chunks while _recording is True."""
        safety_limit = MAX_RECORD_SECONDS * (SAMPLE_RATE / CHUNK)
        count = 0
        while True:
            with self._lock:
                if not self._recording:
                    break
            if count > safety_limit:
                with self._lock:
                    self._recording = False
                break
            try:
                data = self._stream.read(CHUNK, exception_on_overflow=False)
                self._frames.append(data)
            except OSError:
                break
            count += 1

    def _save_wav(self, file_path: str) -> None:
        """Convert raw PCM frames to a .wav file via soundfile."""
        raw = b"".join(self._frames)
        audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        sf.write(file_path, audio_np, SAMPLE_RATE, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------

class RecordingSessionManager:
    """
    Manages the full recording session across all 35 sentences.
    Persists state to session.json inside output_dir.

    Intended to be driven by the FastAPI layer or Tauri frontend events.
    """

    def __init__(self, output_dir: str, script_path: str):
        self.base_dir = Path(output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.script = self._load_script(script_path)
        self.sessions: dict[str, SentenceSession] = {}
        self._active_recorder: Optional[Recorder] = None
        self._active_session: Optional[SentenceSession] = None
        self._session_file = self.base_dir / "session.json"

        if self._session_file.exists():
            self._restore_session()

    # ------------------------------------------------------------------
    # Script
    # ------------------------------------------------------------------

    def _load_script(self, path: str) -> list[dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["sentences"]

    def get_sentence(self, sentence_id: str) -> Optional[dict]:
        return next((s for s in self.script if s["id"] == sentence_id), None)

    def get_all_sentences(self) -> list[dict]:
        return self.script

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def begin_sentence(self, sentence_id: str, recording_name: str) -> SentenceSession:
        """Start a session for a sentence (user has named it)."""
        if sentence_id not in self.sessions:
            self.sessions[sentence_id] = SentenceSession(
                sentence_id=sentence_id,
                recording_name=recording_name,
            )
        session = self.sessions[sentence_id]
        sentence_dir = self.base_dir / sentence_id
        self._active_recorder = Recorder(output_dir=str(sentence_dir))
        self._active_session = session
        return session

    def press_hold(self) -> None:
        """User pressed the record button."""
        if self._active_recorder:
            self._active_recorder.start_hold()

    def release_hold(self) -> Optional[Take]:
        """User released the record button. Returns new Take or None."""
        if not self._active_recorder or not self._active_session:
            return None

        take_id = self._active_session.next_take_label()
        take = self._active_recorder.stop_hold(take_id)

        if take:
            self._active_session.takes.append(take)
            self._persist_session()

        return take

    def delete_take(self, sentence_id: str, take_id: str) -> bool:
        """Delete a specific take."""
        session = self.sessions.get(sentence_id)
        if not session:
            return False

        take = next((t for t in session.takes if t.take_id == take_id), None)
        if not take:
            return False

        if self._active_recorder:
            self._active_recorder.delete_take(take)

        session.takes = [t for t in session.takes if t.take_id != take_id]

        if session.selected_take_id == take_id:
            session.selected_take_id = None

        self._persist_session()
        return True

    def select_take(self, sentence_id: str, take_id: str) -> bool:
        """Mark a take as the final selected take for this sentence."""
        session = self.sessions.get(sentence_id)
        if not session:
            return False
        if not any(t.take_id == take_id for t in session.takes):
            return False
        session.selected_take_id = take_id
        self._persist_session()
        return True

    def finish_sentence(self) -> None:
        """Clean up recorder after user moves to next sentence."""
        if self._active_recorder:
            self._active_recorder.cleanup()
            self._active_recorder = None
        self._active_session = None

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    def get_progress(self) -> dict:
        total = len(self.script)
        completed = sum(
            1 for s in self.sessions.values() if s.selected_take_id is not None
        )
        return {
            "total": total,
            "completed": completed,
            "percent": round((completed / total) * 100, 1),
            "remaining": total - completed,
        }

    def get_session_summary(self) -> list[dict]:
        return [s.to_dict() for s in self.sessions.values()]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_session(self) -> None:
        data = {
            "sessions": {k: v.to_dict() for k, v in self.sessions.items()}
        }
        with open(self._session_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _restore_session(self) -> None:
        with open(self._session_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for sid, sdata in data.get("sessions", {}).items():
            takes = [Take(**t) for t in sdata["takes"]]
            self.sessions[sid] = SentenceSession(
                sentence_id=sdata["sentence_id"],
                recording_name=sdata["recording_name"],
                takes=takes,
                selected_take_id=sdata.get("selected_take_id"),
            )


# ---------------------------------------------------------------------------
# Quick CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    manager = RecordingSessionManager(
        output_dir="recordings",
        script_path="curated_script.json",
    )

    sentence = manager.get_all_sentences()[0]
    print(f"\nSentence: [{sentence['id']}] {sentence['text']}")
    name = input("Name this recording: ").strip() or "default"

    manager.begin_sentence(sentence["id"], name)

    input("\nPress ENTER and HOLD to record (release ENTER to stop)...")
    manager.press_hold()
    input()
    take = manager.release_hold()

    if take:
        print(f"\nTake saved: {take.file_path} ({take.duration_seconds:.1f}s)")
        manager.select_take(sentence["id"], take.take_id)
        print("Take selected as final.")
    else:
        print("Take too short — discarded.")

    manager.finish_sentence()
    print("\nProgress:", manager.get_progress())