"""
quick_test.py
-------------
Bypasses the recording UI entirely. Uses your existing recordings in
workspace/recordings/ to run the full pipeline and test inference output.

Usage:
    cd backend
    python quick_test.py                        # full pipeline test
    python quick_test.py --skip-training        # skip training, test inference only
    python quick_test.py --text "Hello world"   # custom TTS text
    python quick_test.py --steps clean,segment  # run specific steps only

Steps:
    clean     → runs cleaner on all selected takes
    segment   → builds dataset + F5-TTS manifests
    voice     → runs voice consistency check
    train     → LoRA fine-tuning (slow, ~30-60min on GPU)
    infer     → generates speech and saves output.wav
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure backend/ is on sys.path so all imports work
# regardless of which directory the script is run from
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Suppress all HF noise before any imports
import setup  # noqa — applies env suppressions

BASE_DIR          = Path(__file__).parent / "workspace"
RECORDINGS_DIR    = BASE_DIR / "recordings"
SESSION_JSON      = RECORDINGS_DIR / "session.json"
CLEANED_DIR       = RECORDINGS_DIR / "cleaned"
DATASET_DIR       = BASE_DIR / "dataset"
VOICE_PROFILE_DIR = BASE_DIR / "voice_profile"
MODEL_DIR         = BASE_DIR / "model"
SCRIPT_PATH       = Path(__file__).parent / "curated_script.json"
OUTPUT_WAV        = Path(__file__).parent / "output.wav"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def check_session() -> dict:
    """Load and validate session.json."""
    if not SESSION_JSON.exists():
        print(f"❌ No session.json found at {SESSION_JSON}")
        print("   Make sure you have recordings in workspace/recordings/")
        sys.exit(1)

    with open(SESSION_JSON, encoding="utf-8") as f:
        session = json.load(f)

    sessions = session.get("sessions", {})
    selected = {k: v for k, v in sessions.items() if v.get("selected_take_id")}
    total    = len(sessions)
    done     = len(selected)

    print(f"  Found {total} sentences, {done} with selected takes")

    if done == 0:
        print("❌ No selected takes found. Select a take for each sentence first.")
        sys.exit(1)

    if done < 10:
        print(f"⚠️  Only {done} selected takes — recommended minimum is 10 for decent quality")

    return session


def step_clean():
    section("STEP 1 — Cleaning selected takes")
    from cleaner import SessionCleaner

    cleaner = SessionCleaner(
        session_json=str(SESSION_JSON),
        cleaned_dir=str(CLEANED_DIR),
    )
    report = cleaner.run()
    s = report["summary"]
    print(f"\n  ✅ Cleaned: {s['successful']}/{s['total_processed']}")
    if s["failed"] > 0:
        print(f"  ⚠️  Failed: {s['failed']}")
    if s["skipped_no_selection"] > 0:
        print(f"  ⚠️  Skipped (no selection): {s['skipped_no_selection']}")
    return report


def step_segment():
    section("STEP 2 — Building dataset + F5-TTS manifests")
    from segmenter import DatasetSegmenter, ManifestValidator

    cleaning_report = CLEANED_DIR / "cleaning_report.json"
    if not cleaning_report.exists():
        print("❌ cleaning_report.json not found — run clean step first")
        sys.exit(1)

    seg = DatasetSegmenter(
        session_json=str(SESSION_JSON),
        cleaning_report=str(cleaning_report),
        script_json=str(SCRIPT_PATH),
        dataset_dir=str(DATASET_DIR),
    )
    report = seg.build()
    s = report.to_dict()["summary"]
    print(f"\n  ✅ Dataset: {s['total_entries_in_dataset']}/{s['total_sentences_in_script']} entries")
    print(f"  ⏱  Duration: {s['total_duration_min']} minutes")
    if s["missing_sentences"]:
        print(f"  ⚠️  Missing: {s['missing_sentences']}")

    # Validate manifest
    mv  = ManifestValidator()
    chk = mv.validate_filelist(str(DATASET_DIR / "filelist.txt"))
    if chk["ready_for_training"]:
        print(f"  ✅ Manifest valid — {chk['valid_entries']} entries ready for training")
    else:
        print(f"  ❌ Manifest issues: {chk['issues']}")
        sys.exit(1)

    return report


def step_voice():
    section("STEP 3 — Voice consistency check")
    from voice_encoder import VoiceEncoder, RESEMBLYZER_AVAILABLE

    if not RESEMBLYZER_AVAILABLE:
        print("  ⚠️  resemblyzer not installed — skipping consistency check")
        print("  ℹ️  This doesn't affect training or inference")
        return

    enc     = VoiceEncoder(output_dir=str(VOICE_PROFILE_DIR))
    profile = enc.encode_from_dataset(dataset_dir=str(DATASET_DIR))
    s       = profile.to_dict()

    print(f"\n  Consistency : {s['consistency']['mean']:.3f} ({s['consistency']['status']})")
    print(f"  Duration    : {s['dataset']['total_duration_min']} minutes")
    print(f"  Ready       : {'✅ Yes' if profile.ready_for_training else '❌ No'}")

    if profile.warnings:
        print(f"\n  ⚠️  {len(profile.warnings)} flagged takes:")
        for w in profile.warnings[:5]:
            print(f"     • {w.take_id}: {w.message[:60]}")

    if not profile.ready_for_training:
        print("\n❌ Voice profile not ready for training — re-record flagged takes")
        sys.exit(1)


def step_train():
    section("STEP 4 — LoRA fine-tuning")

    filelist = DATASET_DIR / "filelist.txt"
    if not filelist.exists():
        print("❌ filelist.txt not found — run segment step first")
        sys.exit(1)

    from trainer import F5TTSLoRATrainer, TrainingConfig, DEFAULT_EPOCHS

    config  = TrainingConfig(
        filelist_path=str(filelist),
        output_dir=str(MODEL_DIR),
        epochs=DEFAULT_EPOCHS,
    )
    trainer = F5TTSLoRATrainer(config)

    if trainer.resume_check():
        print("  ▶  Resuming from existing checkpoint...")

    def on_progress(p):
        bar    = "█" * int(p.percent / 5) + "░" * (20 - int(p.percent / 5))
        eta    = p.to_dict()["eta_human"]
        print(f"\r  [{bar}] {p.percent:5.1f}% | loss: {p.loss:.4f} | ETA: {eta}    ", end="", flush=True)

    start  = time.time()
    report = trainer.train(on_progress=on_progress)
    elapsed = time.time() - start

    print(f"\n\n  ✅ Training complete in {elapsed/60:.1f} min")
    print(f"  Final loss : {report.get('final_loss', 'N/A')}")
    print(f"  Best loss  : {report.get('best_loss', 'N/A')}")
    print(f"  Model saved: {report.get('final_model_path')}")


def step_infer(text: str):
    section("STEP 5 — Inference test")

    # Find ref audio from dataset
    filelist = DATASET_DIR / "filelist.txt"
    if not filelist.exists():
        print("❌ filelist.txt not found — run segment step first")
        sys.exit(1)

    with open(filelist, encoding="utf-8") as f:
        first_line = f.readline().strip()
    parts = first_line.split("|")
    if len(parts) < 2:
        print("❌ Could not read ref audio from filelist.txt")
        sys.exit(1)

    ref_audio = parts[0]
    ref_text  = parts[1]

    print(f"  Ref audio : {Path(ref_audio).name}")
    print(f"  Ref text  : {ref_text[:60]}...")
    print(f"  Gen text  : {text}")

    from inference import TTSInferenceEngine

    engine = TTSInferenceEngine(
        model_dir=str(MODEL_DIR),
        ref_audio_path=ref_audio,
        ref_text=ref_text,
    )

    print("\n  Generating...")
    start  = time.time()
    result = engine.generate_to_file(text, output_path=str(OUTPUT_WAV))
    elapsed = time.time() - start

    print(f"\n  ✅ Generated {result.total_duration_sec:.1f}s of audio in {elapsed:.1f}s")
    print(f"  RTF        : {result.real_time_factor:.3f}x real-time")
    print(f"  Engine     : {result.engine_used}")
    print(f"  Output     : {OUTPUT_WAV.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Quick pipeline test — no recording needed")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, run inference only (uses existing model)")
    parser.add_argument("--steps", default=None,
                        help="Comma-separated steps to run: clean,segment,voice,train,infer")
    parser.add_argument("--text", default="Hello, this is a test of my cloned voice. How does it sound?",
                        help="Text to synthesize in inference step")
    args = parser.parse_args()

    # Determine which steps to run
    if args.steps:
        steps = [s.strip() for s in args.steps.split(",")]
    elif args.skip_training:
        steps = ["clean", "segment", "voice", "infer"]
    else:
        steps = ["clean", "segment", "voice", "train", "infer"]

    print("\n╔══════════════════════════════════════════════════╗")
    print("║         Open TTS Studio — Quick Test             ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Steps : {', '.join(steps):<41}║")
    print(f"║  Text  : {args.text[:41]:<41}║")
    print("╚══════════════════════════════════════════════════╝")

    # Validate session exists before running anything
    check_session()

    start_total = time.time()

    if "clean"   in steps: step_clean()
    if "segment" in steps: step_segment()
    if "voice"   in steps: step_voice()
    if "train"   in steps: step_train()
    if "infer"   in steps: step_infer(args.text)

    elapsed = time.time() - start_total
    print(f"\n{'='*55}")
    print(f"  ✅ All steps complete in {elapsed:.1f}s")
    if "infer" in steps:
        print(f"  🔊 Output saved to: {OUTPUT_WAV.resolve()}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()