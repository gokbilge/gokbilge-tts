#!/usr/bin/env bash
# train_v0_4_step500k_finetune_turkish_heavy.sh - Prepare the v0.4 Turkish-heavy fine-tune pilot.
#
# This script does not start training. It only prepares the run directory, copies manifest
# artifacts, exports Piper data, and prints the exact training command.
#
# v0.4 fine-tunes from v0.1 step500k.
# v0.3 scratch relaxed was negative and must not be resumed.
# Benchmark remains v0.1 step500k.
# Review milestones: 5k, 10k, 25k, 50k.
# Stop early if fragmented or cut-up audio appears.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

REPO_ROOT="$( cd "$( dirname "$0" )/../.." && pwd )"
RUN_DIR="$REPO_ROOT/runs/v0_4_step500k_finetune_turkish_heavy_001"
SOURCE_MANIFEST_DIR="$REPO_ROOT/runs/v0_1_full_001/manifests"
BASE_CHECKPOINT="$REPO_ROOT/runs/v0_1_full_001/candidates/step500k/step500k.ckpt"
SELECTED_MANIFEST="$REPO_ROOT/data/manifests/train_v0_4_finetune_turkish_heavy_conservative.jsonl"
EVAL_SET="$REPO_ROOT/eval_sets/v0_3_turkish_heavy_sentences.txt"
RUN_MANIFEST_DIR="$RUN_DIR/manifests"
PIPER_DIR="$RUN_DIR/piper"
TRAINING_DIR="$RUN_DIR/training"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"

if [[ ! -f "$BASE_CHECKPOINT" ]]; then
    echo "[v0.4-step500k] ERROR: base checkpoint not found: $BASE_CHECKPOINT" >&2
    exit 1
fi
if [[ ! -f "$SELECTED_MANIFEST" ]]; then
    echo "[v0.4-step500k] ERROR: selected manifest not found: $SELECTED_MANIFEST" >&2
    exit 1
fi
if [[ ! -f "$EVAL_SET" ]]; then
    echo "[v0.4-step500k] ERROR: diagnostic eval set missing: $EVAL_SET" >&2
    exit 1
fi

for path in \
    "$SOURCE_MANIFEST_DIR/val.jsonl" \
    "$SOURCE_MANIFEST_DIR/test.jsonl" \
    "$SOURCE_MANIFEST_DIR/symbols.txt" \
    "$SOURCE_MANIFEST_DIR/stats.json"
do
    if [[ ! -f "$path" ]]; then
        echo "[v0.4-step500k] ERROR: required source manifest artifact missing: $path" >&2
        exit 1
    fi
done

mkdir -p "$RUN_MANIFEST_DIR" "$PIPER_DIR" "$TRAINING_DIR" "$CHECKPOINT_DIR"

cp "$SELECTED_MANIFEST" "$RUN_MANIFEST_DIR/train.jsonl"
cp "$SOURCE_MANIFEST_DIR/val.jsonl" "$RUN_MANIFEST_DIR/val.jsonl"
cp "$SOURCE_MANIFEST_DIR/test.jsonl" "$RUN_MANIFEST_DIR/test.jsonl"
cp "$SOURCE_MANIFEST_DIR/symbols.txt" "$RUN_MANIFEST_DIR/symbols.txt"
cp "$SOURCE_MANIFEST_DIR/stats.json" "$RUN_MANIFEST_DIR/stats.json"

echo "[v0.4-step500k] Using selected manifest:   $RUN_MANIFEST_DIR/train.jsonl"
echo "[v0.4-step500k] Base checkpoint:            $BASE_CHECKPOINT"
echo "[v0.4-step500k] Run directory:              $RUN_DIR"
echo "[v0.4-step500k] Diagnostic eval set:        $EVAL_SET"
echo "[v0.4-step500k] Exporting selected manifest to Piper LJSpeech format..."

gokbilge-tts export-piper \
    --manifest-dir "$RUN_MANIFEST_DIR" \
    --out "$PIPER_DIR" \
    --sample-rate 22050 \
    --language tr

echo "[v0.4-step500k] Piper dataset ready: $PIPER_DIR"
echo "[v0.4-step500k] Fine-tune milestones: 5k, 10k, 25k, 50k"
echo "[v0.4-step500k] Keep v0.1 step500k as the benchmark to beat."
echo "[v0.4-step500k] Stop early if fragmented or cut-up audio appears."
echo "[v0.4-step500k] To start the pilot, run:"
echo "               bash recipes/issai_piper/train.sh \"$PIPER_DIR\" \"$TRAINING_DIR\" \"$CHECKPOINT_DIR\" \"$BASE_CHECKPOINT\""
