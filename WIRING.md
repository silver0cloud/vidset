# Open TTS Studio — Wiring Guide

## Development (fast iteration, no sidecar build needed)

In dev mode, `tauri dev` does **not** require the PyInstaller sidecar binary.
Run the backend directly with Python, and the frontend connects to it on the
default port `8000` (see the browser-mode fallback in `src/api/client.ts`).

```bash
# Terminal 1 — backend
cd backend
pip install -r requirements.txt --break-system-packages
python main.py 8000

# Terminal 2 — frontend (web only, no native shell)
npm install
npm run dev
# open http://localhost:1420
```

For the full native Tauri shell in dev mode (still talking to the
manually-started backend above), run:

```bash
npm run tauri dev
```

> Note: `main.rs` always tries to spawn the sidecar binary. For pure web-mode
> dev (`npm run dev` only, no `tauri dev`), the sidecar isn't involved at
> all — `client.ts` just hits `localhost:8000` directly, which is why
> starting the backend manually is enough.

## Production (sidecar-packaged backend)

For a real distributable app, the Python backend is compiled into a
standalone binary so end users never need Python installed.

```bash
# 1. Package the backend (slow — 10-20 min due to torch/onnxruntime/transformers)
pip install pyinstaller --break-system-packages
python build_sidecar.py

# 2. Build the Tauri app — bundles the sidecar binary automatically
npm run tauri build
```

`build_sidecar.py` outputs to `src-tauri/binaries/tts-backend-<target-triple>`,
matching Tauri's required sidecar naming convention. `tauri.conf.json`
references this via `bundle.externalBin`.

At runtime, `main.rs`:
1. Finds a free local port
2. Spawns the sidecar binary with that port as an argument
3. Polls `/api/health` until it responds
4. Emits a `backend-ready` event with the port to the frontend
5. `client.ts` listens for this event and routes all REST/WebSocket calls there

## Known gaps before a real release

- `trainer.py` / `inference.py` have placeholder F5-TTS calls — need the
  actual F5-TTS package/fork locked in before sidecar packaging will produce
  a fully functional binary (it will currently build, but training/inference
  calls will fail until those imports resolve to real code).
- The sidecar binary bundles torch + onnxruntime + transformers, so expect
  a large (~1-2GB) binary and a slow first build. This is normal for an
  ML-heavy sidecar, not a bug.
- `pyaudio` requires system `portaudio` on the build machine
  (`apt install portaudio19-dev` / `brew install portaudio`).
