/**
 * client.ts
 * ---------
 * Thin wrapper around the FastAPI backend.
 * REST calls via fetch, streaming via WebSocket.
 *
 * Backend port is dynamic in production (Tauri sidecar picks a free port
 * and emits it via the "backend-ready" event). In plain browser dev mode
 * (no Tauri runtime), it falls back to the default :8000 used by
 * `python main.py` / `uvicorn main:app`.
 */

let resolvedPort = 8000;
let portReady: Promise<number> = Promise.resolve(8000);

// Only attempt to hook into Tauri's event system if running inside Tauri.
// `window.__TAURI__` is injected by the Tauri runtime; absent in plain browser dev.
if (typeof window !== "undefined" && (window as any).__TAURI__) {
  portReady = new Promise((resolve) => {
    import("@tauri-apps/api/event").then(({ listen }) => {
      listen<number>("backend-ready", (event) => {
        resolvedPort = event.payload;
        resolve(event.payload);
      });
      listen("backend-failed", () => {
        console.error("TTS backend failed to start.");
      });
    });
  });
}

function baseUrl(): string {
  return `http://127.0.0.1:${resolvedPort}`;
}

function wsBase(): string {
  return `ws://127.0.0.1:${resolvedPort}`;
}

/** Call once on app boot (e.g. in App.tsx) to ensure the port is known
 *  before the first API call fires. No-ops instantly in browser dev mode. */
export async function waitForBackend(): Promise<void> {
  await portReady;
}

// ---------------------------------------------------------------------------
// REST helpers
// ---------------------------------------------------------------------------

async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${baseUrl()}${path}`);
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return res.json();
}

async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${baseUrl()}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `POST ${path} failed: ${res.status}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Script
// ---------------------------------------------------------------------------

export const getScript = () =>
  apiGet<{ sentences: any[] }>("/api/script");

// ---------------------------------------------------------------------------
// Recording session
// ---------------------------------------------------------------------------

export const beginSentence = (sentenceId: string, recordingName: string) =>
  apiPost("/api/session/begin", { sentence_id: sentenceId, recording_name: recordingName });

export const pressHold = () => apiPost("/api/session/press");

export const releaseHold = () => apiPost<{ take: any; validation: any }>("/api/session/release");

export const selectTake = (sentenceId: string, takeId: string) =>
  apiPost("/api/session/select", { sentence_id: sentenceId, take_id: takeId });

export const deleteTake = (sentenceId: string, takeId: string) =>
  apiPost("/api/session/delete", { sentence_id: sentenceId, take_id: takeId });

export const finishSentence = () => apiPost("/api/session/finish");

export const getProgress = () => apiGet<any>("/api/session/progress");

export const getSummary = () => apiGet<{ sessions: any[] }>("/api/session/summary");

// ---------------------------------------------------------------------------
// Dataset / voice profile
// ---------------------------------------------------------------------------

export const buildDataset = () => apiPost<any>("/api/dataset/build");

export const buildVoiceProfile = () => apiPost<any>("/api/voice-profile/build");

export const getVoiceProfile = () => apiGet<any>("/api/voice-profile");

export const getEngineInfo = () => apiGet<any>("/api/engine-info");

// ---------------------------------------------------------------------------
// Training — WebSocket
// ---------------------------------------------------------------------------

export interface TrainOptions {
  epochs?: number;
  onProgress: (data: any) => void;
  onError: (message: string) => void;
  onComplete: () => void;
}

export function startTraining(opts: TrainOptions): () => void {
  const ws = new WebSocket(`${wsBase()}/ws/train`);

  ws.onopen = () => {
    ws.send(JSON.stringify({ epochs: opts.epochs || 100 }));
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.status === "error") {
      opts.onError(data.message);
    } else if (data.status === "complete") {
      opts.onComplete();
    } else {
      opts.onProgress(data);
    }
  };

  ws.onerror = () => opts.onError("Connection to training server lost.");

  // Return a cleanup function to close the socket if user navigates away
  return () => ws.close();
}

// ---------------------------------------------------------------------------
// Inference — WebSocket streaming
// ---------------------------------------------------------------------------

export interface SpeakOptions {
  text: string;
  onAudioChunk: (chunk: ArrayBuffer) => void;
  onDone: (result: any) => void;
  onError: (message: string) => void;
}

export function speak(opts: SpeakOptions): () => void {
  const ws = new WebSocket(`${wsBase()}/ws/speak`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    ws.send(JSON.stringify({ text: opts.text }));
  };

  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      opts.onAudioChunk(event.data);
    } else {
      const data = JSON.parse(event.data);
      if (data.status === "error") {
        opts.onError(data.message);
      } else if (data.status === "done") {
        opts.onDone(data.result);
      }
    }
  };

  ws.onerror = () => opts.onError("Connection to speech server lost.");

  return () => ws.close();
}
