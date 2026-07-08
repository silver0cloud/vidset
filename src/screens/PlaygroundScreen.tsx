/**
 * PlaygroundScreen.tsx
 * --------------------
 * Type any text, hear it spoken in the user's trained voice.
 * Streams audio chunks live via WebSocket as they're generated.
 */

import { useRef, useState } from "react";
import { useAppStore } from "../store/store";
import * as api from "../api/client";
import { StreamingAudioPlayer } from "../api/audioPlayer";

export default function PlaygroundScreen() {
  const { playgroundText, setPlaygroundText, isGenerating, setIsGenerating } =
    useAppStore();

  const [resultInfo, setResultInfo] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const playerRef = useRef<StreamingAudioPlayer | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);

  function handleSpeak() {
    if (!playgroundText.trim() || isGenerating) return;

    setError(null);
    setResultInfo(null);
    setIsGenerating(true);

    const player = new StreamingAudioPlayer();
    playerRef.current = player;

    const cleanup = api.speak({
      text: playgroundText.trim(),
      onAudioChunk: (chunk) => player.pushChunk(chunk),
      onDone: (result) => {
        player.finish();
        setIsGenerating(false);
        setResultInfo(
          `Generated ${result.total_duration_sec.toFixed(1)}s of audio in ${result.generation_time_sec.toFixed(
            1
          )}s (${result.real_time_factor.toFixed(2)}x real-time) using ${result.engine_used}.`
        );
      },
      onError: (msg) => {
        setIsGenerating(false);
        setError(msg);
      },
    });

    cleanupRef.current = cleanup;
  }

  function handleStop() {
    cleanupRef.current?.();
    playerRef.current?.stop();
    setIsGenerating(false);
  }

  return (
    <div className="screen playground-screen">
      <h1>Your voice, your words</h1>
      <p className="lead">Type anything and hear it read back in your own voice.</p>

      <textarea
        className="playground-textarea"
        placeholder="Type something to say..."
        value={playgroundText}
        onChange={(e) => setPlaygroundText(e.target.value)}
        rows={6}
        maxLength={1000}
      />

      <div className="char-count">{playgroundText.length} / 1000</div>

      <div className="playground-actions">
        {!isGenerating ? (
          <button
            className="primary-btn"
            onClick={handleSpeak}
            disabled={!playgroundText.trim()}
          >
            Speak
          </button>
        ) : (
          <button className="danger-btn" onClick={handleStop}>
            Stop
          </button>
        )}
      </div>

      {isGenerating && (
        <div className="status-line">
          <span className="spinner" />
          <span>Generating speech...</span>
        </div>
      )}

      {resultInfo && <p className="result-info">{resultInfo}</p>}
      {error && <p className="error-text">{error}</p>}
    </div>
  );
}
