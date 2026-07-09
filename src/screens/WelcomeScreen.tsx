/**
 * WelcomeScreen.tsx
 * -----------------
 * Onboarding entry point. Explains the 3-minute recording flow
 * and loads the curated script before moving to the recording studio.
 */

import { useEffect, useState } from "react";
import { useAppStore } from "../store/store";
import { getScript } from "../api/client";

export default function WelcomeScreen() {
  const setStage = useAppStore((s) => s.setStage);
  const setSentences = useAppStore((s) => s.setSentences);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Pre-warm the script load so "Get Started" feels instant
    getScript().then((r) => setSentences(r.sentences)).catch(() => {});
  }, []);

  async function handleStart() {
    setLoading(true);
    setError(null);
    try {
      const r = await getScript();
      setSentences(r.sentences);
      setStage("recording");
    } catch (e: any) {
      setError("Could not reach the backend. Is the server running?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="screen welcome-screen">
      <div className="welcome-content">
        <span className="eyebrow">Occano</span>
        <h1>Clone your voice in three minutes.</h1>
        <p className="lead">
          Record 35 short sentences in a quiet space, and we'll build a
          personalized text-to-speech model that sounds like you — entirely
          on your own machine.
        </p>

        <div className="prep-list">
          <div className="prep-item">
            <span className="prep-num">1</span>
            <span>Find a quiet room and use a decent microphone.</span>
          </div>
          <div className="prep-item">
            <span className="prep-num">2</span>
            <span>Speak naturally, like reading an audiobook.</span>
          </div>
          <div className="prep-item">
            <span className="prep-num">3</span>
            <span>Hold the record button, release when you're done.</span>
          </div>
        </div>

        {error && <p className="error-text">{error}</p>}

        <button className="primary-btn" onClick={handleStart} disabled={loading}>
          {loading ? "Loading..." : "Get started"}
        </button>
      </div>
    </div>
  );
}
