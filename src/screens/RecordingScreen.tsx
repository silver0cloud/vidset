/**
 * RecordingScreen.tsx
 * -------------------
 * Core recording studio. Shows current sentence, press-and-hold record
 * button, take list with select/delete, and overall progress.
 */

import { useState } from "react";
import { useAppStore } from "../store/store";
import * as api from "../api/client";

export default function RecordingScreen() {
  const {
    sentences,
    currentSentenceIndex,
    nextSentence,
    prevSentence,
    isRecording,
    setIsRecording,
    takesBySentence,
    addTake,
    removeTake,
    selectedTakeBySentence,
    selectTake,
    lastValidationWarnings,
    setLastValidationWarnings,
    progress,
    setProgress,
    setStage,
  } = useAppStore();

  const [recordingName, setRecordingName] = useState("");
  const [sessionStarted, setSessionStarted] = useState(false);
  const [busy, setBusy] = useState(false);

  const sentence = sentences[currentSentenceIndex];
  const takes = sentence ? takesBySentence[sentence.id] || [] : [];
  const selectedTakeId = sentence ? selectedTakeBySentence[sentence.id] : null;

  if (!sentence) {
    return <div className="screen">Loading script...</div>;
  }

  async function handleBegin() {
    if (!recordingName.trim()) return;
    setBusy(true);
    try {
      await api.beginSentence(sentence.id, recordingName.trim());
      setSessionStarted(true);
    } catch (e: any) {
      alert(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function handlePressStart() {
    setIsRecording(true);
    try {
      await api.pressHold();
    } catch (e: any) {
      setIsRecording(false);
      alert(e.message);
    }
  }

  async function handlePressEnd() {
    setIsRecording(false);
    setBusy(true);
    try {
      const result = await api.releaseHold();
      addTake(sentence.id, result.take);
      setLastValidationWarnings(result.validation.warnings || []);
    } catch (e: any) {
      // Too-short recordings raise a 400 — not a crash, just inform user
      setLastValidationWarnings([
        { code: "discarded", message: e.message, severity: "low" },
      ]);
    } finally {
      setBusy(false);
    }
  }

  async function handleSelectTake(takeId: string) {
    await api.selectTake(sentence.id, takeId);
    selectTake(sentence.id, takeId);
  }

  async function handleDeleteTake(takeId: string) {
    await api.deleteTake(sentence.id, takeId);
    removeTake(sentence.id, takeId);
  }

  async function handleNext() {
    await api.finishSentence();
    const p = await api.getProgress();
    setProgress(p);
    setSessionStarted(false);
    setRecordingName("");
    setLastValidationWarnings([]);

    if (currentSentenceIndex >= sentences.length - 1) {
      setStage("training");
    } else {
      nextSentence();
    }
  }

  return (
    <div className="screen recording-screen">
      <div className="recording-header">
        <span className="sentence-counter">
          Sentence {currentSentenceIndex + 1} of {sentences.length}
        </span>
        <span className={`mood-tag mood-${sentence.mood}`}>{sentence.mood}</span>
      </div>

      <div className="sentence-display">
        <p>{sentence.text}</p>
      </div>

      {!sessionStarted ? (
        <div className="name-input-row">
          <input
            type="text"
            placeholder="Name this recording (e.g. take_1)"
            value={recordingName}
            onChange={(e) => setRecordingName(e.target.value)}
          />
          <button
            className="primary-btn"
            onClick={handleBegin}
            disabled={!recordingName.trim() || busy}
          >
            Start
          </button>
        </div>
      ) : (
        <>
          <div className="record-control">
            <button
              className={`record-btn ${isRecording ? "recording" : ""}`}
              onMouseDown={handlePressStart}
              onMouseUp={handlePressEnd}
              onTouchStart={handlePressStart}
              onTouchEnd={handlePressEnd}
              disabled={busy}
            >
              {isRecording ? "Recording..." : "Hold to record"}
            </button>
            <p className="hint">Press and hold, release when finished.</p>
          </div>

          {lastValidationWarnings.length > 0 && (
            <div className="warnings">
              {lastValidationWarnings.map((w, i) => (
                <p key={i} className={`warning warning-${w.severity}`}>
                  {w.message}
                </p>
              ))}
            </div>
          )}

          {takes.length > 0 && (
            <div className="takes-list">
              <h3>Takes ({takes.length})</h3>
              {takes.map((t) => (
                <div
                  key={t.take_id}
                  className={`take-row ${selectedTakeId === t.take_id ? "selected" : ""}`}
                >
                  <span>{t.take_id}</span>
                  <span className="take-duration">{t.duration_seconds.toFixed(1)}s</span>
                  <div className="take-actions">
                    <button onClick={() => handleSelectTake(t.take_id)}>
                      {selectedTakeId === t.take_id ? "✓ Selected" : "Select"}
                    </button>
                    <button className="danger-btn" onClick={() => handleDeleteTake(t.take_id)}>
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="nav-row">
            <button onClick={prevSentence} disabled={currentSentenceIndex === 0}>
              Back
            </button>
            <button
              className="primary-btn"
              onClick={handleNext}
              disabled={!selectedTakeId}
            >
              {currentSentenceIndex >= sentences.length - 1 ? "Finish" : "Next sentence"}
            </button>
          </div>
        </>
      )}

      {progress && (
        <div className="progress-bar-track">
          <div className="progress-bar-fill" style={{ width: `${progress.percent}%` }} />
        </div>
      )}
    </div>
  );
}
