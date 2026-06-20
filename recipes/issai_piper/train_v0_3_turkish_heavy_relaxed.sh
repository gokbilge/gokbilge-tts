#!/usr/bin/env bash
# train_v0_3_turkish_heavy_relaxed.sh ? Prepare the provisional v0.3 relaxed Turkish-heavy pilot
#
# Creating or reviewing this script does NOT start training.
# Run it manually later only after the v0.3 spot-check package is reviewed.
#
# v0.3 relaxed is the selected candidate after aggressive/hard comparison.
# v0.1 step500k remains the benchmark to beat.
# v0.2 balanced was negative and should not be resumed.
# First milestones to review: 50k, 100k, 150k, 200k.
# Use the diagnostic eval set: eval_sets/v0_3_turkish_heavy_sentences.txt
# Do not start until the spot-check package is reviewed.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

REPO_ROOT="$( cd "$( dirname "$0" )/../.." && pwd )"
RUN_DIR="$REPO_ROOT/runs/v0_3_turkish_heavy_relaxed_001"
SOURCE_MANIFEST_DIR="$REPO_ROOT/runs/v0_1_full_001/manifests"
RELAXED_MANIFEST="$REPO_ROOT/data/manifests/train_v0_3_turkish_heavy_relaxed.jsonl"
EVAL_SET="$REPO_ROOT/eval_sets/v0_3_turkish_heavy_sentences.txt"
RUN_MANIFEST_DIR="$RUN_DIR/manifests"
PIPER_DIR="$RUN_DIR/piper"
TRAINING_DIR="$RUN_DIR/training"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"

if [[ ! -f "$RELAXED_MANIFEST" ]]; then
    echo "[v0.3-relaxed] ERROR: relaxed manifest not found: $RELAXED_MANIFEST" >&2
    echo "[v0.3-relaxed] Build and review the local relaxed manifest first." >&2
    exit 1
fi

if [[ ! -f "$EVAL_SET" ]]; then
    echo "[v0.3-relaxed] ERROR: diagnostic eval set missing: $EVAL_SET" >&2
    exit 1
fi

for path in \
    "$SOURCE_MANIFEST_DIR/val.jsonl" \
    "$SOURCE_MANIFEST_DIR/test.jsonl" \
    "$SOURCE_MANIFEST_DIR/symbols.txt" \
    "$SOURCE_MANIFEST_DIR/stats.json"
    do
    if [[ ! -f "$path" ]]; then
        echo "[v0.3-relaxed] ERROR: required source manifest artifact missing: $path" >&2
        exit 1
    fi
done

mkdir -p "$RUN_MANIFEST_DIR" "$PIPER_DIR" "$TRAINING_DIR" "$CHECKPOINT_DIR"

cp "$RELAXED_MANIFEST" "$RUN_MANIFEST_DIR/train.jsonl"
cp "$SOURCE_MANIFEST_DIR/val.jsonl" "$RUN_MANIFEST_DIR/val.jsonl"
cp "$SOURCE_MANIFEST_DIR/test.jsonl" "$RUN_MANIFEST_DIR/test.jsonl"
cp "$SOURCE_MANIFEST_DIR/symbols.txt" "$RUN_MANIFEST_DIR/symbols.txt"
cp "$SOURCE_MANIFEST_DIR/stats.json" "$RUN_MANIFEST_DIR/stats.json"

echo "[v0.3-relaxed] Using relaxed manifest:      $RUN_MANIFEST_DIR/train.jsonl"
echo "[v0.3-relaxed] Source full manifest dir:   $SOURCE_MANIFEST_DIR"
echo "[v0.3-relaxed] Run directory:              $RUN_DIR"
echo "[v0.3-relaxed] Diagnostic eval set:        $EVAL_SET"
echo "[v0.3-relaxed] Exporting relaxed manifest to Piper LJSpeech format..."

gokbilge-tts export-piper \
    --manifest-dir "$RUN_MANIFEST_DIR" \
    --out "$PIPER_DIR" \
    --sample-rate 22050 \
    --language tr

echo "[v0.3-relaxed] Piper dataset ready: $PIPER_DIR"
echo "[v0.3-relaxed] Hyperparameters remain aligned with v0.1/v0.2 for fair comparison."
echo "[v0.3-relaxed] Milestone review points: 50k, 100k, 150k, 200k"
echo "[v0.3-relaxed] Keep v0.1 step500k as the benchmark to beat."
echo "[v0.3-relaxed] Do not start unless the spot-check package has been reviewed:"
echo "               reports/v0_3_turkish_heavy/spotcheck/SPOTCHECK_README.md"
echo "[v0.3-relaxed] To start the pilot after review, run:"
echo "               bash recipes/issai_piper/train.sh \"$PIPER_DIR\" \"$TRAINING_DIR\" \"$CHECKPOINT_DIR\""
