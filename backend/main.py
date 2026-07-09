"""
FastAPI backend wiring the full TTS pipeline together.

REST endpoints  : script, recording session CRUD, dataset build, voice profile
WebSocket       : /ws/setup  (model download progress on first launch)
                  /ws/train  (live training progress)
                  /ws/speak  (streaming TTS generation)

Run:
    uvicorn main:app --reload --port 8000

Dependencies:
    pip install fastapi uvicorn[standard] python-multipart
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

# Anchor backend directory on sys.path so all imports work regardless
# of where uvicorn is launched from
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# Import setup first — applies HF warning suppressions as a side effect
from setup import ModelSetup, is_model_cached, SetupProgress  # noqa: F401

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from recorder import RecordingSessionManager
from validator import TakeValidator
from cleaner import SessionCleaner
from segmenter import DatasetSegmenter, ManifestValidator
from voice_encoder import VoiceEncoder, VoiceVerifier
from trainer import F5TTSTrainer, TrainingConfig, TrainingProgress
from inference import TTSInferenceEngine, detect_device


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Occano API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tauri webview origin — tighten in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Paths — single user, local-first app (no multi-tenant auth needed)
# ---------------------------------------------------------------------------

BASE_DIR        = Path("workspace")
SCRIPT_PATH     = Path("curated_script.json")
RECORDINGS_DIR  = BASE_DIR / "recordings"
CLEANED_DIR     = BASE_DIR / "recordings" / "cleaned"
DATASET_DIR     = BASE_DIR / "dataset"
VOICE_PROFILE_DIR = BASE_DIR / "voice_profile"
MODEL_DIR       = BASE_DIR / "model"
BASE_MODEL_PATH = Path("models/f5_tts_base.pt")

for d in [RECORDINGS_DIR, CLEANED_DIR, DATASET_DIR, VOICE_PROFILE_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Single shared session manager instance (single-user local app)
session_manager = RecordingSessionManager(
    output_dir=str(RECORDINGS_DIR), script_path=str(SCRIPT_PATH)
)
validator = TakeValidator()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class BeginSentenceRequest(BaseModel):
    sentence_id: str
    recording_name: str


class SelectTakeRequest(BaseModel):
    sentence_id: str
    take_id: str


class DeleteTakeRequest(BaseModel):
    sentence_id: str
    take_id: str


class SpeakRequest(BaseModel):
    text: str


# ===========================================================================
# REST — Script
# ===========================================================================

@app.get("/api/script")
def get_script():
    """Returns all 35 curated sentences."""
    return {"sentences": session_manager.get_all_sentences()}


@app.get("/api/script/{sentence_id}")
def get_sentence(sentence_id: str):
    sentence = session_manager.get_sentence(sentence_id)
    if not sentence:
        raise HTTPException(404, f"Sentence {sentence_id} not found")
    return sentence


# ===========================================================================
# REST — Recording session
# ===========================================================================

@app.post("/api/session/begin")
def begin_sentence(req: BeginSentenceRequest):
    """Start a recording session for a given sentence."""
    sentence = session_manager.get_sentence(req.sentence_id)
    if not sentence:
        raise HTTPException(404, f"Sentence {req.sentence_id} not found")

    session = session_manager.begin_sentence(req.sentence_id, req.recording_name)
    return session.to_dict()


@app.post("/api/session/press")
def press_hold():
    """User pressed the record button."""
    session_manager.press_hold()
    return {"status": "recording"}


@app.post("/api/session/release")
def release_hold():
    """
    User released the record button.
    Saves the take and auto-runs validation (soft warnings only).
    """
    take = session_manager.release_hold()
    if not take:
        raise HTTPException(400, "Recording too short — discarded. Try again.")

    result = validator.validate(take_id=take.take_id, file_path=take.file_path)

    return {
        "take": take.to_dict(),
        "validation": result.to_dict(),
    }


@app.post("/api/session/select")
def select_take(req: SelectTakeRequest):
    ok = session_manager.select_take(req.sentence_id, req.take_id)
    if not ok:
        raise HTTPException(404, "Take or sentence not found")
    return {"status": "selected"}


@app.post("/api/session/delete")
def delete_take(req: DeleteTakeRequest):
    ok = session_manager.delete_take(req.sentence_id, req.take_id)
    if not ok:
        raise HTTPException(404, "Take or sentence not found")
    return {"status": "deleted"}


@app.post("/api/session/finish")
def finish_sentence():
    """Call when moving to the next sentence — releases mic resources."""
    session_manager.finish_sentence()
    return {"status": "finished"}


@app.get("/api/session/progress")
def get_progress():
    return session_manager.get_progress()


@app.get("/api/session/summary")
def get_summary():
    return {"sessions": session_manager.get_session_summary()}


# ===========================================================================
# REST — Dataset pipeline (clean + segment)
# ===========================================================================

@app.post("/api/dataset/build")
def build_dataset():
    """
    Runs cleaning (on selected takes only) + segmentation into the
    final training dataset with F5-TTS ready manifests.
    """
    session_json = RECORDINGS_DIR / "session.json"
    if not session_json.exists():
        raise HTTPException(400, "No recording session found yet.")

    # 1. Clean selected takes
    cleaner = SessionCleaner(
        session_json=str(session_json), cleaned_dir=str(CLEANED_DIR)
    )
    clean_report = cleaner.run()

    # 2. Segment into dataset + manifests
    segmenter = DatasetSegmenter(
        session_json=str(session_json),
        cleaning_report=str(CLEANED_DIR / "cleaning_report.json"),
        script_json=str(SCRIPT_PATH),
        dataset_dir=str(DATASET_DIR),
    )
    seg_report = segmenter.build()

    # 3. Validate manifest
    mv = ManifestValidator()
    manifest_check = mv.validate_filelist(str(DATASET_DIR / "filelist.txt"))

    return {
        "cleaning": clean_report["summary"],
        "segmentation": seg_report.to_dict()["summary"],
        "manifest_check": manifest_check,
    }


# ===========================================================================
# REST — Voice profile
# ===========================================================================

@app.post("/api/voice-profile/build")
def build_voice_profile():
    """Extracts speaker embedding + consistency check from the built dataset."""
    if not (DATASET_DIR / "wavs").exists():
        raise HTTPException(400, "Dataset not built yet. Call /api/dataset/build first.")

    encoder = VoiceEncoder(output_dir=str(VOICE_PROFILE_DIR))
    try:
        profile = encoder.encode_from_dataset(dataset_dir=str(DATASET_DIR))
    except ValueError as e:
        raise HTTPException(400, str(e))

    return profile.to_dict()


@app.get("/api/voice-profile")
def get_voice_profile():
    path = VOICE_PROFILE_DIR / "voice_profile.json"
    if not path.exists():
        raise HTTPException(404, "Voice profile not built yet.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# REST — Engine info
# ===========================================================================

@app.get("/api/engine-info")
def engine_info():
    """Returns which compute device will be used for inference."""
    device = detect_device()
    return {
        "device": device,
        "engine": "f5tts_lora_cuda" if device == "cuda" else "f5tts_lora_cpu",
        "reason": (
            "CUDA GPU detected — using GPU for fast inference"
            if device == "cuda"
            else "No GPU detected — using CPU inference (slower but works)"
        ),
    }


# ===========================================================================
# WebSocket — Training (live progress)
# ===========================================================================

@app.websocket("/ws/train")
async def ws_train(websocket: WebSocket):
    """
    Starts LoRA fine-tuning and streams live progress.
    Client sends: {"mode": "lora" | "full", "epochs": 100}
    Server sends: TrainingProgress dicts, then a final "complete" message.
    """
    await websocket.accept()

    try:
        start_msg = await websocket.receive_json()
        mode = start_msg.get("mode", "lora")
        epochs = start_msg.get("epochs", 100)

        filelist_path = DATASET_DIR / "filelist.txt"

        if not filelist_path.exists():
            await websocket.send_json({
                "status": "error",
                "message": "Dataset not ready. Build it first via /api/dataset/build.",
            })
            await websocket.close()
            return

        config = TrainingConfig(
            filelist_path=str(filelist_path),
            output_dir=str(MODEL_DIR),
            epochs=epochs,
            use_lora=True,
        )
        trainer = F5TTSTrainer(config)

        if trainer.resume_check():
            await websocket.send_json({
                "status": "resuming",
                "message": "Found existing checkpoint — resuming training.",
            })

        # Send immediate message — model load takes 10-30s and WS times out without this
        await websocket.send_json({
            "status": "loading",
            "message": "Loading F5-TTS model into memory...",
        })

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        # Periodic heartbeat keeps WS alive during slow model load
        async def heartbeat():
            count = 0
            while True:
                await asyncio.sleep(5)
                count += 1
                try:
                    await websocket.send_json({
                        "status": "loading",
                        "message": f"Loading model... ({count * 5}s)",
                    })
                except Exception:
                    break

        heartbeat_task = asyncio.create_task(heartbeat())

        def on_progress(progress: TrainingProgress):
            asyncio.run_coroutine_threadsafe(queue.put(progress.to_dict()), loop)

        async def run_training():
            try:
                await loop.run_in_executor(None, lambda: trainer.train(on_progress=on_progress))
                await queue.put({"status": "complete"})
            except Exception as e:
                import traceback
                err = traceback.format_exc()
                print(f"[ws_train] Training crashed:\n{err}")
                await queue.put({"status": "error", "message": str(e), "traceback": err})

        training_task = asyncio.create_task(run_training())

        first_msg = True
        while True:
            msg = await queue.get()
            if first_msg:
                heartbeat_task.cancel()  # Stop heartbeat once real progress starts
                first_msg = False
            await websocket.send_json(msg)
            if msg.get("status") in ("complete", "error"):
                break

        await training_task

    except WebSocketDisconnect:
        print("Training client disconnected — training continues in background.")
    except Exception as e:
        try:
            await websocket.send_json({"status": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ===========================================================================
# WebSocket — Inference (streaming TTS)
# ===========================================================================

_inference_engine: Optional[TTSInferenceEngine] = None


def _get_ref_audio_and_text() -> tuple[str, str]:
    """
    Resolves the reference audio path and text for F5-TTS zero-shot inference.
    Uses the first selected take from the session as the voice reference.
    """
    session_json = RECORDINGS_DIR / "session.json"
    if not session_json.exists():
        raise HTTPException(400, "No recording session found.")

    with open(session_json, encoding="utf-8") as f:
        session = json.load(f)

    script = session_manager.get_all_sentences()
    script_map = {s["id"]: s["text"] for s in script}

    # Find the first session entry with a selected take in the cleaned dataset
    for sid, sdata in session.get("sessions", {}).items():
        selected_id = sdata.get("selected_take_id")
        if not selected_id:
            continue
        take = next((t for t in sdata["takes"] if t["take_id"] == selected_id), None)
        if not take:
            continue
        # Look for the cleaned version in dataset/wavs/
        cleaned_name = f"{selected_id}_cleaned.wav"
        cleaned_path = DATASET_DIR / "wavs" / cleaned_name
        if cleaned_path.exists():
            return str(cleaned_path), script_map.get(sid, "")

    raise HTTPException(400, "No cleaned takes found. Build the dataset first.")


def get_inference_engine() -> TTSInferenceEngine:
    global _inference_engine
    if _inference_engine is None:
        ref_audio, ref_text = _get_ref_audio_and_text()
        _inference_engine = TTSInferenceEngine(
            model_dir=str(MODEL_DIR),
            ref_audio_path=ref_audio,
            ref_text=ref_text,
        )
    return _inference_engine


@app.websocket("/ws/speak")
async def ws_speak(websocket: WebSocket):
    """
    Streams generated speech audio chunks as they're produced.
    Client sends: {"text": "Hello world"}
    Server sends: binary PCM16 chunks, then {"status": "done", "result": {...}}
    """
    await websocket.accept()

    try:
        msg = await websocket.receive_json()
        text = msg.get("text", "").strip()

        if not text:
            await websocket.send_json({"status": "error", "message": "Empty text."})
            await websocket.close()
            return

        engine = get_inference_engine()
        loop = asyncio.get_event_loop()

        def run_generation():
            gen = engine.generate_stream(text)
            chunks = []
            try:
                while True:
                    chunk = next(gen)
                    chunks.append(chunk)
            except StopIteration as e:
                return chunks, e.value

        chunks, result = await loop.run_in_executor(None, run_generation)

        for chunk in chunks:
            await websocket.send_bytes(chunk.to_bytes())

        await websocket.send_json({"status": "done", "result": result.to_dict()})

    except WebSocketDisconnect:
        print("Speak client disconnected.")
    except Exception as e:
        try:
            await websocket.send_json({"status": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ===========================================================================
# Health check
# ===========================================================================

@app.get("/api/health")
def health():
    return {"status": "ok"}


# ===========================================================================
# Setup — model download (must complete before recording can begin)
# ===========================================================================

@app.get("/api/setup/model-status")
def model_status():
    """Check whether the F5-TTS model is already cached locally."""
    cached = is_model_cached()
    return {
        "cached": cached,
        "message": "Model ready." if cached else "Model not downloaded yet.",
    }


@app.websocket("/ws/setup")
async def ws_setup(websocket: WebSocket):
    """
    Downloads F5-TTS + Vocos on first launch, streams progress.
    Client receives SetupProgress dicts, then {"done": true}.
    If already cached, completes immediately.

    Client usage:
        const ws = new WebSocket("ws://localhost:8000/ws/setup")
        ws.onmessage = (e) => {
            const data = JSON.parse(e.data)
            if (data.done) { proceed to welcome screen }
        }
    """
    await websocket.accept()

    try:
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def on_progress(p: SetupProgress):
            asyncio.run_coroutine_threadsafe(queue.put(p.to_dict()), loop)

        async def run_setup():
            success = await loop.run_in_executor(
                None, lambda: ModelSetup().run(on_progress=on_progress)
            )
            await queue.put({"done": True, "success": success})

        setup_task = asyncio.create_task(run_setup())

        while True:
            msg = await queue.get()
            await websocket.send_json(msg)
            if msg.get("done"):
                break

        await setup_task

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"done": True, "success": False, "error": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
