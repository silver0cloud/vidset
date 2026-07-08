/**
 * SetupScreen.tsx
 * ---------------
 * Shown on first launch only. Downloads F5-TTS base model (~1.3GB)
 * and Vocos vocoder before recording begins.
 * Once model is cached, this screen is skipped entirely.
 */

import { useEffect, useState } from "react";
import { useAppStore } from "../store/store";

interface SetupProgress {
  step: number;
  total_steps: number;
  percent: number;
  label: string;
  done: boolean;
  error?: string;
  success?: boolean;
}

const WS_BASE = "ws://127.0.0.1:8000";

export default function SetupScreen() {
  const setStage = useAppStore((s) => s.setStage);
  const [progress, setProgress] = useState<SetupProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ws = new WebSocket(`${WS_BASE}/ws/setup`);

    ws.onmessage = (event) => {
      const data: SetupProgress = JSON.parse(event.data);

      if (data.error) {
        setError(data.error);
        return;
      }

      setProgress(data);

      if (data.done && data.success !== false) {
        // Short pause so user sees "Model ready!" before moving on
        setTimeout(() => setStage("welcome"), 800);
      }
    };

    ws.onerror = () => {
      setError("Could not connect to backend. Is the server running?");
    };

    return () => ws.close();
  }, []);

  return (
    <div className="screen setup-screen">
      <span className="eyebrow">First time setup</span>
      <h1>Downloading voice model</h1>
      <p className="lead">
        Open TTS Studio needs to download the F5-TTS base model (~1.3 GB).
        This happens once — future launches will skip this step.
      </p>

      {!progress && !error && (
        <div className="status-line">
          <span className="spinner" />
          <span>Connecting...</span>
        </div>
      )}

      {progress && (
        <div className="setup-progress">
          <div className="progress-bar-track">
            <div
              className="progress-bar-fill"
              style={{ width: `${progress.percent}%` }}
            />
          </div>
          <div className="setup-stats">
            <span>{progress.label}</span>
            <span>{progress.percent.toFixed(0)}%</span>
          </div>
        </div>
      )}

      {error && (
        <div className="error-panel">
          <p className="error-text">{error}</p>
          <p className="hint">
            Check your internet connection, then restart the app.
            If you're behind a proxy, set the <code>HF_ENDPOINT</code> environment variable.
          </p>
        </div>
      )}

      {progress?.done && !error && (
        <p className="hint" style={{ color: "var(--success)" }}>
          ✓ {progress.label}
        </p>
      )}
    </div>
  );
}
