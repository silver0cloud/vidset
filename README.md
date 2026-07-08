# Open TTS Studio

**Clone your voice in three minutes. Entirely on your machine. No cloud, no APIs.**

Open TTS Studio is an open-source desktop application that lets anyone record their voice and fine-tune a personal text-to-speech model using [F5-TTS](https://github.com/SWivid/F5-TTS). The goal is to make high-quality, personalised TTS accessible to non-technical users while keeping all processing local and private.

---

## Features

- **Guided recording** — 35 curated sentences designed to cover all 44 English phonemes across 5 moods (neutral, warm, assertive, curious, reflective)
- **Real-time quality checks** — soft warnings for clipping, noise, silence, and volume issues after each take
- **LoRA fine-tuning** — lightweight adapter training on top of F5-TTS base (~16GB VRAM recommended, CPU fallback available)
- **Streaming inference** — generated speech streams back in real time as it's produced
- **Fully local** — no internet required after the initial model download; voice data never leaves your machine
- **Desktop app** — built with Tauri + React; ships as a single native installer

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Tauri Desktop App                        │
│  ┌──────────────────────┐     ┌─────────────────────────────┐ │
│  │   React Frontend      │    │   Python FastAPI Backend    │ │
│  │   (4 screens)         │◄──►│   (sidecar process)         │ │
│  │   Zustand state       │    │                             │ │
│  └──────────────────────┘     │  audio_processor/           │ │
│                               │    recorder.py              │ │
│  WebSocket streams:           │    validator.py             │ │
│    /ws/train  (live loss)     │    cleaner.py               │ │
│    /ws/speak  (audio chunks)  │    segmenter.py             │ │
│                               │  voice_encoder.py (QC)      │ │
│                               │  trainer.py (F5-TTS LoRA)   │ │
│                               │  inference.py (streaming)   │ │
│                               └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**User flow:**
```
Record 35 sentences → validate + clean → build dataset
→ voice consistency check → LoRA fine-tune → speak any text
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML core | PyTorch + F5-TTS (DiT + flow matching) |
| Fine-tuning | PEFT LoRA (`target_modules="all-linear"`) |
| Audio processing | torchaudio, librosa, scipy, soundfile |
| Voice QC | resemblyzer (speaker embeddings) |
| Backend | FastAPI + uvicorn |
| Frontend | React 18 + TypeScript + Zustand |
| Desktop shell | Tauri 2 (Rust) |
| Packaging | PyInstaller (Python sidecar binary) |

---

## Prerequisites

### System dependencies

```bash
# Linux
sudo apt install portaudio19-dev ffmpeg

# macOS
brew install portaudio ffmpeg

# Windows
# portaudio ships via pip wheel; install ffmpeg from https://ffmpeg.org
```

### Python ≥ 3.10, Node.js ≥ 18, Rust (for Tauri)

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

---

## Development setup

The fastest way to work on the project — no sidecar build required.

**Terminal 1 — Backend**
```bash
cd backend
pip install -r requirements.txt
python main.py 8000
```

**Terminal 2 — Frontend (web only)**
```bash
npm install
npm run dev
# Open http://localhost:1420
```

**Terminal 2 (alternative) — Full Tauri shell**
```bash
npm install
npm run tauri dev
# Tauri will try to spawn the sidecar; the manually started backend on :8000
# handles all requests via the browser-mode fallback in client.ts
```

> **Note:** On first run, F5-TTS will download the base model (~1.3 GB) from HuggingFace.
> Set `HF_HOME` to control the cache location.

---

## Production build

Packages the Python backend into a single executable and bundles it with the Tauri app.

```bash
# 1. Package the backend (takes 10-20 min — torch + onnxruntime are large)
pip install pyinstaller
python build_sidecar.py

# 2. Build the Tauri app
npm run tauri build
# Installer output: src-tauri/target/release/bundle/
```

The sidecar binary is automatically named `tts-backend-<target-triple>` by `build_sidecar.py`, matching Tauri's `externalBin` convention.

---

## Project structure

```
open-tts-studio/
├── backend/                        Python FastAPI backend
│   ├── main.py                       REST + WebSocket API (20 routes)
│   ├── recorder.py                   Press-hold mic recording, take management
│   ├── validator.py                  Audio quality checks (7 soft warnings)
│   ├── cleaner.py                    Audio post-processing → 24kHz output
│   ├── segmenter.py                  Dataset builder + F5-TTS manifest writer
│   ├── voice_encoder.py              Speaker embedding QC (resemblyzer)
│   ├── trainer.py                    F5-TTS LoRA fine-tuning pipeline
│   ├── inference.py                  Streaming TTS generation (F5-TTS API)
│   ├── curated_script.json           35 phoneme-rich recording sentences
│   └── requirements.txt
│
├── src/                            React frontend
│   ├── screens/
│   │   ├── WelcomeScreen.tsx         Onboarding
│   │   ├── RecordingScreen.tsx       Press-hold recording + takes management
│   │   ├── TrainingScreen.tsx        Dataset build + live training progress
│   │   └── PlaygroundScreen.tsx      Type & speak
│   ├── store/store.ts                Zustand global state
│   ├── api/
│   │   ├── client.ts                 REST + WebSocket wrappers (dynamic port)
│   │   └── audioPlayer.ts            PCM16 streaming audio (Web Audio API)
│   ├── App.tsx                       Stage-based screen switcher
│   ├── App.css / index.css           Styles
│   └── main.tsx
│
├── src-tauri/                      Rust native shell
│   ├── src/main.rs                   Sidecar spawn + port wiring
│   ├── capabilities/default.json     Shell permissions (Tauri v2)
│   ├── Cargo.toml
│   ├── build.rs
│   └── tauri.conf.json
│
├── build_sidecar.py                PyInstaller packaging script
├── WIRING.md                       Dev vs production setup details
├── package.json
├── vite.config.ts
├── tsconfig.json
└── index.html
```

---

## API reference

### REST endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/script` | All 35 curated sentences |
| GET | `/api/script/{id}` | Single sentence |
| POST | `/api/session/begin` | Start recording a sentence |
| POST | `/api/session/press` | Begin recording (hold) |
| POST | `/api/session/release` | Stop recording + auto-validate |
| POST | `/api/session/select` | Mark a take as final |
| POST | `/api/session/delete` | Delete a take |
| POST | `/api/session/finish` | Release mic, move to next sentence |
| GET | `/api/session/progress` | Overall recording progress |
| GET | `/api/session/summary` | All sessions with takes |
| POST | `/api/dataset/build` | Clean + segment → training dataset |
| POST | `/api/voice-profile/build` | Run voice consistency check |
| GET | `/api/voice-profile` | Fetch saved voice profile |
| GET | `/api/engine-info` | GPU/CPU detection |

### WebSocket endpoints

**`/ws/train`**
```
Client → {"epochs": 100}
Server → {"epoch": 1, "total_epochs": 100, "percent": 1.0, "loss": 0.42, "eta_human": "4h 12m", ...}
Server → {"status": "complete"}
```

**`/ws/speak`**
```
Client → {"text": "Hello, this is my voice."}
Server → <binary PCM16 audio chunks>
Server → {"status": "done", "result": {"total_duration_sec": 2.4, "real_time_factor": 0.18, ...}}
```

---

## Model details

**Base model:** [F5-TTS v1 Base](https://huggingface.co/SWivid/F5-TTS) — 335.8M parameters, DiT architecture, trained on 100K hours of multilingual audio.

**Fine-tuning approach:** LoRA adapters applied to all linear layers of the DiT backbone using PEFT (`target_modules="all-linear"`, rank=16). This is consistent with the PEFT-TTS research (Interspeech 2025) which validates LoRA on F5-TTS DiT layers.

**Inference approach:** Zero-shot voice cloning — a reference audio clip from the user's recordings is passed alongside the target text, and the fine-tuned model generates speech in the user's voice.

**Sample rate:** 24kHz (F5-TTS native). The recorder captures at 22050Hz; `cleaner.py` resamples to 24kHz.

**License note:** The F5-TTS pretrained weights are under **CC-BY-NC** license due to their training data (Emilia dataset). This means the weights cannot be used for commercial purposes. The code in this repository is MIT licensed.

---

## Hardware requirements

| Mode | VRAM | Training time (35 takes) |
|---|---|---|
| GPU (recommended) | 16GB+ | ~30-60 min |
| GPU (minimum) | 8GB | ~2-4 hours (reduce `batch_size`) |
| CPU only | — | ~12-24 hours |

To reduce batch size for low VRAM, edit `backend/trainer.py`:
```python
config = TrainingConfig(..., batch_size=800)  # default is 1600
```

---

## Troubleshooting

**`portaudio` not found on Linux**
```bash
sudo apt install portaudio19-dev
pip install pyaudio
```

**F5-TTS model download times out**
```bash
# Set a longer timeout or mirror
export HF_ENDPOINT=https://hf-mirror.com  # China mirror
```

**CUDA out of memory during training**
Reduce `batch_size` in `TrainingConfig` (default: 1600 frames). Try 800 or 400.

**LoRA adapter not found at inference time**
Training must complete before the playground works. The adapter is saved to `workspace/model/lora_adapter/` after training finishes.

**Sidecar fails to start in production build**
Run `python build_sidecar.py` before `npm run tauri build`. The sidecar binary must exist at `src-tauri/binaries/tts-backend-<target-triple>`.

---

## Contributing

Contributions are very welcome. A few areas where help is especially valuable:

- **Audio pipeline improvements** — better noise reduction, VAD silence detection
- **Training optimizations** — gradient checkpointing, mixed precision (fp16/bf16)
- **More curated sentences** — additional phoneme coverage, accents, languages
- **Windows testing** — the project is primarily developed on Linux/macOS
- **Frontend design polish** — the UI is functional but could use a design pass

### Adding a new language

1. Add sentences to `curated_script.json` with the new language tag
2. Find or train a multilingual F5-TTS checkpoint (see [SHARED.md](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/SHARED.md))
3. Update `VOCAB_HF` and `BASE_MODEL_HF` in `trainer.py` / `inference.py`

---

## Roadmap

- [ ] Docker Compose for contributor onboarding
- [ ] Multiple voice profiles per user
- [ ] Export to ONNX for edge deployment
- [ ] Multilingual support (starting with Hindi, Spanish)
- [ ] Longer recording sessions (10+ min for higher accuracy)
- [ ] Evaluation metrics (MOS, speaker similarity score)

---

## License

**Code:** MIT License — see `LICENSE`

**F5-TTS model weights:** CC-BY-NC — see [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)

---

## Acknowledgements

This project builds on the work of many open-source contributors:

- [F5-TTS](https://github.com/SWivid/F5-TTS) — Yushen Chen et al., the core TTS architecture
- [resemblyzer](https://github.com/resemble-ai/Resemblyzer) — speaker embedding
- [PEFT](https://github.com/huggingface/peft) — LoRA fine-tuning
- [Tauri](https://tauri.app) — desktop shell
- [FastAPI](https://fastapi.tiangolo.com) — backend framework
