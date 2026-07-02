/**
 * TrainingScreen.tsx
 * ------------------
 * Pipeline: build dataset -> build voice profile -> start training (WS).
 * Shows live loss curve + ETA while training runs.
 */

import { useEffect, useRef, useState } from "react";
import { useAppStore } from "../store/store";
import * as api from "../api/client";

type Phase = "dataset" | "voice_profile" | "ready" | "training" | "done" | "error";

export default function TrainingScreen() {
  const {
    isTraining,
    setIsTraining,
    trainingProgress,
    setTrainingProgress,
    setTrainingComplete,
    voiceConsistency,
    setVoiceProfile,
    setStage,
  } = useAppStore();

  const [phase, setPhase] = useState<Phase>("dataset");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [lossPoints, setLossPoints] = useState<number[]>([]);
  const cleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    runSetup();
    return () => cleanupRef.current?.();
  }, []);

  async function runSetup() {
    try {
      setPhase("dataset");
      await api.buildDataset();

      setPhase("voice_profile");
      const profile = await api.buildVoiceProfile();
      setVoiceProfile(profile.ready_for_training, profile.consistency.mean);

      setPhase("ready");
    } catch (e: any) {
      setErrorMsg(e.message);
      setPhase("error");
    }
  }

  function handleStartTraining() {
    setPhase("training");
    setIsTraining(true);
    setLossPoints([]);

    const cleanup = api.startTraining({
      epochs: 100,
      onProgress: (data) => {
        setTrainingProgress(data);
        if (typeof data.loss === "number") {
          setLossPoints((prev) => [...prev.slice(-49), data.loss]);
        }
      },
      onError: (msg) => {
        setErrorMsg(msg);
        setPhase("error");
        setIsTraining(false);
      },
      onComplete: () => {
        setIsTraining(false);
        setTrainingComplete(true);
        setPhase("done");
      },
    });

    cleanupRef.current = cleanup;
  }

  const maxLoss = Math.max(...lossPoints, 1);

  return (
    <div className="screen training-screen">
      <h1>Training your voice</h1>

      {phase === "dataset" && <StatusLine label="Building your dataset..." />}
      {phase === "voice_profile" && <StatusLine label="Analyzing your voice..." />}

      {phase === "ready" && (
        <div className="ready-panel">
          <p>
            Voice consistency:{" "}
            <strong>
              {voiceConsistency !== null ? `${(voiceConsistency * 100).toFixed(0)}%` : "—"}
            </strong>
          </p>
          <p className="hint">
            This fine-tunes a lightweight model on your voice. It runs entirely
            on your machine and may take a while depending on your hardware.
          </p>
          <button className="primary-btn" onClick={handleStartTraining}>
            Start training
          </button>
        </div>
      )}

      {(phase === "training" || phase === "done") && trainingProgress && (
        <div className="training-panel">
          <div className="progress-bar-track">
            <div
              className="progress-bar-fill"
              style={{ width: `${trainingProgress.percent}%` }}
            />
          </div>
          <div className="training-stats">
            <span>Epoch {trainingProgress.epoch}/{trainingProgress.total_epochs}</span>
            <span>Loss: {trainingProgress.loss.toFixed(4)}</span>
            <span>ETA: {trainingProgress.eta_human}</span>
          </div>

          <svg className="loss-chart" viewBox="0 0 300 80" preserveAspectRatio="none">
            <polyline
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              points={lossPoints
                .map((l, i) => {
                  const x = (i / Math.max(lossPoints.length - 1, 1)) * 300;
                  const y = 80 - (l / maxLoss) * 70 - 5;
                  return `${x},${y}`;
                })
                .join(" ")}
            />
          </svg>

          {phase === "done" && (
            <button className="primary-btn" onClick={() => setStage("playground")}>
              Try your voice
            </button>
          )}
        </div>
      )}

      {phase === "error" && (
        <div className="error-panel">
          <p className="error-text">{errorMsg}</p>
          <button onClick={runSetup}>Retry</button>
        </div>
      )}
    </div>
  );
}

function StatusLine({ label }: { label: string }) {
  return (
    <div className="status-line">
      <span className="spinner" />
      <span>{label}</span>
    </div>
  );
}
