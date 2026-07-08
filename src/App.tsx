/**
 * App.tsx
 * -------
 * Root component. On first launch checks if the F5-TTS model is cached.
 * If not → shows SetupScreen (download). If yes → shows WelcomeScreen.
 * After setup, renders the active screen based on Zustand stage state.
 */

import { useEffect, useState } from "react";
import { useAppStore } from "./store/store";
import { waitForBackend } from "./api/client";
import SetupScreen from "./screens/SetupScreen";
import WelcomeScreen from "./screens/WelcomeScreen";
import RecordingScreen from "./screens/RecordingScreen";
import TrainingScreen from "./screens/TrainingScreen";
import PlaygroundScreen from "./screens/PlaygroundScreen";
import "./App.css";

export default function App() {
  const { stage, setStage } = useAppStore();
  const [backendReady, setBackendReady] = useState(false);

  useEffect(() => {
    waitForBackend().then(async () => {
      setBackendReady(true);

      // Check if model already cached — skip setup screen if so
      try {
        const res = await fetch("http://127.0.0.1:8000/api/setup/model-status");
        const data = await res.json();
        setStage(data.cached ? "welcome" : "setup");
      } catch {
        // Backend not reachable yet — default to setup which will show error
        setStage("setup");
      }
    });
  }, []);

  if (!backendReady) {
    return (
      <div className="app-shell">
        <div className="screen">
          <div className="status-line">
            <span className="spinner" />
            <span>Starting up...</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell">
      {stage === "setup" && <SetupScreen />}
      {stage === "welcome" && <WelcomeScreen />}
      {stage === "recording" && <RecordingScreen />}
      {stage === "training" && <TrainingScreen />}
      {stage === "playground" && <PlaygroundScreen />}
    </div>
  );
}
