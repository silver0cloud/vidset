<div align="center">

# Occano

**Clone your voice in three minutes. Entirely on your machine. No cloud, no APIs.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Node 18+](https://img.shields.io/badge/node-18%2B-green)](https://nodejs.org/)
[![Tauri 2](https://img.shields.io/badge/tauri-2-orange)](https://tauri.app)

[Features](#features) • [Architecture](#architecture) • [Quick Start](#development-setup) • [API](#api-reference) • [Contributing](#contributing)

</div>

---

Occano is an open-source desktop application that lets anyone record their voice and fine-tune a personal text-to-speech model using [F5-TTS](https://github.com/SWivid/F5-TTS). It's built for a simple goal: high-quality, personalized voice cloning that non-technical users can run themselves, with every byte of audio staying local.

## Features

| | |
|---|---|
|  **Guided recording** | 35 curated sentences covering all 44 English phonemes across 5 emotional tones |
|  **Real-time quality checks** | Instant feedback on clipping, noise, silence, and volume as you record |
|  **LoRA fine-tuning** | Lightweight adapter training on F5-TTS (~16GB VRAM recommended, CPU fallback supported) |
|  **Streaming inference** | Speech streams back as it's generated — no waiting for full synthesis |
|  **Fully local** | One-time model download, then everything runs offline |
|  **Native desktop app** | Built with Tauri + React, ships as a single installer |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Tauri Desktop App                       │
│                                                              │
│   ┌────────────────────┐          ┌───────────────────────┐  │
│   │   React Frontend   │  ◄────►  │  Python FastAPI       │  │
│   │   (4 screens)      │          │  Backend (sidecar)    │  │
│   │   Zustand state    │          │                       │  │
│   └────────────────────┘          │  audio_processor/     │  │
│                                   |  recorder.py          │  │
│   WebSocket streams:              |  validator.py         │  │
│     /ws/train  (live loss)        |  cleaner.py           │  │
│     /ws/speak  (audio chunks)     |  segmenter.py         │  │
│                                   |  voice_encoder.py (QC)│  │
│                                   |  trainer.py (LoRA)    │  │
│                                   |  inference.py (stream)│  │
│                                   └───────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

**User flow**

```
Record 35 sentences → validate & clean → build dataset
     → voice consistency check → LoRA fine-tune → speak any text
```

## Tech Stack

| Layer | Technology |
|---|---|
| ML core | PyTorch + F5-TTS (DiT + flow matching) |
| Fine-tuning | PEFT LoRA |
| Audio processing | torchaudio, librosa, scipy, soundfile |
| Backend | FastAPI + uvicorn |
| Frontend | React 18 + TypeScript + Zustand |
| Desktop shell | Tauri (Rust) |
| Packaging | PyInstaller (Python sidecar binary) |

## Dependencies 

- **Python** ≥ 3.10
- **Node.js** ≥ 18
- **Rust** (for Tauri)

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**System audio dependencies**

```bash
# Linux
sudo apt install portaudio19-dev ffmpeg

# macOS
brew install portaudio ffmpeg

# Windows
# portaudio ships via pip wheel; install ffmpeg from https://ffmpeg.org
```

## Quick Start 

Setup a virtual environment, then run two terminals and paste the following commands.

**Terminal 1 — Backend**

```bash
cd backend
pip install -r requirements.txt
python main.py 8000
```

**Terminal 2 — Frontend**

```bash
npm install
npm run dev
# → http://localhost:1420
```

**Terminal 2 (alternative) — Full Tauri shell**
**For running the desktop app-** Run the following commands in the terminal 2 in place of React frontend.

```bash
npm install
npm run tauri dev
# Tauri spawns the sidecar; the manually started backend on :8000
# handles requests via the browser-mode fallback in client.ts
```

> **Note:** On first run, F5-TTS downloads its base model (~1.3 GB) from HuggingFace.
> Set `HF_HOME` to control the cache location.

## Production build

Packages the Python backend into a single executable and bundles it with the Tauri app.

```bash
# 1. Package the backend (10–20 min — torch + onnxruntime are large)
pip install pyinstaller
python build_sidecar.py

# 2. Build the Tauri app
npm run tauri build
# → src-tauri/target/release/bundle/
```

The sidecar binary is automatically named `tts-backend-<target-triple>` by `build_sidecar.py`, matching Tauri's `externalBin` convention.

## Project structure

```
occano/
├── backend/                     Python FastAPI backend
│   ├── main.py                    REST + WebSocket API (20 routes)
│   ├── recorder.py                Press-hold mic recording, take management
│   ├── validator.py               Audio quality checks (7 soft warnings)
│   ├── cleaner.py                 Audio post-processing → 24kHz output
│   ├── segmenter.py               Dataset builder + F5-TTS manifest writer
│   ├── voice_encoder.py           Speaker embedding QC (resemblyzer)
│   ├── trainer.py                 F5-TTS LoRA fine-tuning pipeline
│   ├── inference.py               Streaming TTS generation (F5-TTS API)
│   ├── curated_script.json        35 phoneme-rich recording sentences
│   └── requirements.txt
│
├── src/                        React frontend
│   ├── screens/
│   │   ├── WelcomeScreen.tsx       Onboarding
│   │   ├── RecordingScreen.tsx     Press-hold recording + takes management
│   │   ├── TrainingScreen.tsx      Dataset build + live training progress
│   │   └── PlaygroundScreen.tsx    Type & speak
│   ├── store/store.ts              Zustand global state
│   ├── api/
│   │   ├── client.ts               REST + WebSocket wrappers (dynamic port)
│   │   └── audioPlayer.ts          PCM16 streaming audio (Web Audio API)
│   ├── App.tsx                     Stage-based screen switcher
│   ├── App.css / index.css         Styles
│   └── main.tsx
│
├── src-tauri/                  Rust native shell
│   ├── src/main.rs                 Sidecar spawn + port wiring
│   ├── capabilities/default.json   Shell permissions (Tauri v2)
│   ├── Cargo.toml
│   ├── build.rs
│   └── tauri.conf.json
│
├── build_sidecar.py            PyInstaller packaging script
├── WIRING.md                   Dev vs. production setup details
├── package.json
├── vite.config.ts
├── tsconfig.json
└── index.html
```

## API reference

### REST endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/script` | All 35 curated sentences |
| `GET` | `/api/script/{id}` | Single sentence |
| `POST` | `/api/session/begin` | Start recording a sentence |
| `POST` | `/api/session/press` | Begin recording (hold) |
| `POST` | `/api/session/release` | Stop recording + auto-validate |
| `POST` | `/api/session/select` | Mark a take as final |
| `POST` | `/api/session/delete` | Delete a take |
| `POST` | `/api/session/finish` | Release mic, move to next sentence |
| `GET` | `/api/session/progress` | Overall recording progress |
| `GET` | `/api/session/summary` | All sessions with takes |
| `POST` | `/api/dataset/build` | Clean + segment → training dataset |
| `POST` | `/api/voice-profile/build` | Run voice consistency check |
| `GET` | `/api/voice-profile` | Fetch saved voice profile |
| `GET` | `/api/engine-info` | GPU/CPU detection |

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

## Model details

| | |
|---|---|
| **Base model** | [F5-TTS v1 Base](https://huggingface.co/SWivid/F5-TTS) — 335.8M params, DiT architecture, trained on 100K hours of multilingual audio |
| **Fine-tuning** | LoRA adapters on all linear layers of the DiT backbone via PEFT (`target_modules="all-linear"`, rank=16) — consistent with PEFT-TTS research (Interspeech 2025) |
| **Inference** | Zero-shot voice cloning — a reference clip plus target text drives the fine-tuned model |
| **Sample rate** | 24kHz (F5-TTS native). Recorder captures at 22050Hz; `cleaner.py` resamples to 24kHz |

> **License note:** F5-TTS pretrained weights are **CC-BY-NC** (due to the Emilia training dataset) and cannot be used commercially. This repository's own code is MIT licensed.

## Hardware requirements

| Mode | VRAM | Training time (35 takes) |
|---|---|---|
| GPU (recommended) | 16GB+ | ~30–60 min |
| GPU (minimum) | 8GB | ~2–4 hours (reduce `batch_size`) |
| CPU only | — | ~12–24 hours |

To reduce batch size for low VRAM, edit `backend/trainer.py`:

```python
config = TrainingConfig(..., batch_size=800)  # default is 1600
```

## Troubleshooting

<details>
<summary><strong>portaudio not found on Linux</strong></summary>

```bash
sudo apt install portaudio19-dev
pip install pyaudio
```
</details>

<details>
<summary><strong>F5-TTS model download times out</strong></summary>

```bash
export HF_ENDPOINT=https://hf-mirror.com  # China mirror
```
</details>

<details>
<summary><strong>CUDA out of memory during training</strong></summary>

Reduce `batch_size` in `TrainingConfig` (default: 1600 frames). Try 800 or 400.
</details>

<details>
<summary><strong>LoRA adapter not found at inference time</strong></summary>

Training must complete before the playground works. The adapter is saved to `workspace/model/lora_adapter/` after training finishes.
</details>

<details>
<summary><strong>Sidecar fails to start in production build</strong></summary>

Run `python build_sidecar.py` before `npm run tauri build`. The sidecar binary must exist at `src-tauri/binaries/tts-backend-<target-triple>`.
</details>

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

## License

**Code:** MIT License — see [`LICENSE`](LICENSE)
**F5-TTS model weights:** CC-BY-NC — see [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)

## Acknowledgements

This project builds on the work of many open-source contributors:

- [F5-TTS](https://github.com/SWivid/F5-TTS) — Yushen Chen et al., the core TTS architecture
- [resemblyzer](https://github.com/resemble-ai/Resemblyzer) — speaker embedding
- [PEFT](https://github.com/huggingface/peft) — LoRA fine-tuning
- [Tauri](https://tauri.app) — desktop shell
- [FastAPI](https://fastapi.tiangolo.com) — backend framework

---

<div align="center">
<sub>Built with care for privacy-first voice technology.</sub>
</div>
