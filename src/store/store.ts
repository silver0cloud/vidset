/**
 * store.ts
 * --------
 * Central Zustand store for the TTS Studio app.
 * Tracks: script, recording session, dataset/training status, playground state.
 */

import { create } from "zustand";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Sentence {
  id: string;
  text: string;
  mood: string;
  phonetic_focus: string;
}

export interface Take {
  take_id: string;
  sentence_id: string;
  file_path: string;
  duration_seconds: number;
}

export interface ValidationWarning {
  code: string;
  message: string;
  severity: "low" | "medium" | "high";
}

export interface Progress {
  total: number;
  completed: number;
  percent: number;
  remaining: number;
}

export interface TrainingProgress {
  epoch: number;
  total_epochs: number;
  percent: number;
  loss: number;
  avg_loss_last_10: number;
  eta_human: string;
  checkpoint_saved: boolean;
}

export type AppStage =
  | "setup"
  | "welcome"
  | "recording"
  | "training"
  | "playground";

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

interface AppState {
  // Navigation
  stage: AppStage;
  setStage: (stage: AppStage) => void;

  // Script
  sentences: Sentence[];
  currentSentenceIndex: number;
  setSentences: (s: Sentence[]) => void;
  nextSentence: () => void;
  prevSentence: () => void;
  goToSentence: (index: number) => void;

  // Recording
  isRecording: boolean;
  setIsRecording: (v: boolean) => void;
  takesBySentence: Record<string, Take[]>;
  addTake: (sentenceId: string, take: Take) => void;
  removeTake: (sentenceId: string, takeId: string) => void;
  selectedTakeBySentence: Record<string, string | null>;
  selectTake: (sentenceId: string, takeId: string) => void;
  lastValidationWarnings: ValidationWarning[];
  setLastValidationWarnings: (w: ValidationWarning[]) => void;
  progress: Progress | null;
  setProgress: (p: Progress) => void;

  // Dataset / voice profile
  datasetBuilt: boolean;
  setDatasetBuilt: (v: boolean) => void;
  voiceProfileReady: boolean;
  voiceConsistency: number | null;
  setVoiceProfile: (ready: boolean, consistency: number | null) => void;

  // Training
  isTraining: boolean;
  setIsTraining: (v: boolean) => void;
  trainingProgress: TrainingProgress | null;
  setTrainingProgress: (p: TrainingProgress) => void;
  trainingComplete: boolean;
  setTrainingComplete: (v: boolean) => void;

  // Playground
  playgroundText: string;
  setPlaygroundText: (t: string) => void;
  isGenerating: boolean;
  setIsGenerating: (v: boolean) => void;
  lastGeneratedAudioUrl: string | null;
  setLastGeneratedAudioUrl: (url: string | null) => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  // Navigation
  stage: "welcome",
  setStage: (stage) => set({ stage }),

  // Script
  sentences: [],
  currentSentenceIndex: 0,
  setSentences: (sentences) => set({ sentences }),
  nextSentence: () =>
    set((s) => ({
      currentSentenceIndex: Math.min(
        s.currentSentenceIndex + 1,
        s.sentences.length - 1
      ),
    })),
  prevSentence: () =>
    set((s) => ({
      currentSentenceIndex: Math.max(s.currentSentenceIndex - 1, 0),
    })),
  goToSentence: (index) => set({ currentSentenceIndex: index }),

  // Recording
  isRecording: false,
  setIsRecording: (v) => set({ isRecording: v }),
  takesBySentence: {},
  addTake: (sentenceId, take) =>
    set((s) => ({
      takesBySentence: {
        ...s.takesBySentence,
        [sentenceId]: [...(s.takesBySentence[sentenceId] || []), take],
      },
    })),
  removeTake: (sentenceId, takeId) =>
    set((s) => ({
      takesBySentence: {
        ...s.takesBySentence,
        [sentenceId]: (s.takesBySentence[sentenceId] || []).filter(
          (t) => t.take_id !== takeId
        ),
      },
    })),
  selectedTakeBySentence: {},
  selectTake: (sentenceId, takeId) =>
    set((s) => ({
      selectedTakeBySentence: {
        ...s.selectedTakeBySentence,
        [sentenceId]: takeId,
      },
    })),
  lastValidationWarnings: [],
  setLastValidationWarnings: (w) => set({ lastValidationWarnings: w }),
  progress: null,
  setProgress: (p) => set({ progress: p }),

  // Dataset / voice profile
  datasetBuilt: false,
  setDatasetBuilt: (v) => set({ datasetBuilt: v }),
  voiceProfileReady: false,
  voiceConsistency: null,
  setVoiceProfile: (ready, consistency) =>
    set({ voiceProfileReady: ready, voiceConsistency: consistency }),

  // Training
  isTraining: false,
  setIsTraining: (v) => set({ isTraining: v }),
  trainingProgress: null,
  setTrainingProgress: (p) => set({ trainingProgress: p }),
  trainingComplete: false,
  setTrainingComplete: (v) => set({ trainingComplete: v }),

  // Playground
  playgroundText: "",
  setPlaygroundText: (t) => set({ playgroundText: t }),
  isGenerating: false,
  setIsGenerating: (v) => set({ isGenerating: v }),
  lastGeneratedAudioUrl: null,
  setLastGeneratedAudioUrl: (url) => set({ lastGeneratedAudioUrl: url }),
}));
