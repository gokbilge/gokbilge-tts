#!/usr/bin/env bash
# train_v0_2_balanced.sh ? Prepare and launch the first v0.2 balanced-manifest pilot
#
# This script is source-controlled only. Creating this file does NOT start training.
# Run it manually later, after review, if the v0.2 balanced pilot is approved.
#
# Pilot decision:
# - balanced manifest = first v0.2 pilot
# - strict manifest   = later comparison only
# - compare against v0.1 step300k fallback and step500k primary RC
# - preserve milestone evals at 50k / 100k / 150k / 200k
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

REPO_ROOT="$( cd "$( dirname "$0" )/../.." && pwd )"
RUN_DIR="$REPO_ROOT/runs/v0_2_balanced_001"
SOURCE_MANIFEST_DIR="$REPO_ROOT/runs/v0_1_full_001/manifests"
BALANCED_MANIFEST="$REPO_ROOT/data/manifests/train_clean_balanced.jsonl"
RUN_MANIFEST_DIR="$RUN_DIR/manifests"
PIPER_DIR="$RUN_DIR/piper"
TRAINING_DIR="$RUN_DIR/training"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"

if [[ ! -f "$BALANCED_MANIFEST" ]]; then
    echo "[v0.2] ERROR: balanced manifest not found: $BALANCED_MANIFEST" >&2
    echo "[v0.2] Run the dataset cleaning audit/build step first and keep outputs local-only." >&2
    exit 1
fi

for path in \
    "$SOURCE_MANIFEST_DIR/val.jsonl" \
    "$SOURCE_MANIFEST_DIR/test.jsonl" \
    "$SOURCE_MANIFEST_DIR/symbols.txt" \
    "$SOURCE_MANIFEST_DIR/stats.json"
    do
    if [[ ! -f "$path" ]]; then
        echo "[v0.2] ERROR: required source manifest artifact missing: $path" >&2
        exit 1
    fi
done

mkdir -p "$RUN_MANIFEST_DIR" "$PIPER_DIR" "$TRAINING_DIR" "$CHECKPOINT_DIR"

cp "$BALANCED_MANIFEST" "$RUN_MANIFEST_DIR/train.jsonl"
cp "$SOURCE_MANIFEST_DIR/val.jsonl" "$RUN_MANIFEST_DIR/val.jsonl"
cp "$SOURCE_MANIFEST_DIR/test.jsonl" "$RUN_MANIFEST_DIR/test.jsonl"
cp "$SOURCE_MANIFEST_DIR/symbols.txt" "$RUN_MANIFEST_DIR/symbols.txt"
cp "$SOURCE_MANIFEST_DIR/stats.json" "$RUN_MANIFEST_DIR/stats.json"

echo "[v0.2] Using balanced pilot manifest: $RUN_MANIFEST_DIR/train.jsonl"
echo "[v0.2] Source full manifest:          $SOURCE_MANIFEST_DIR/train.jsonl"
echo "[v0.2] Run directory:                 $RUN_DIR"
echo "[v0.2] Exporting balanced manifest to Piper LJSpeech format..."

gokbilge-tts export-piper \
    --manifest-dir "$RUN_MANIFEST_DIR" \
    --out "$PIPER_DIR" \
    --sample-rate 22050 \
    --language tr

echo "[v0.2] Piper dataset ready: $PIPER_DIR"
echo "[v0.2] Next step starts training with unchanged v0.1 hyperparameters."
echo "[v0.2] Do not compare against smoke; compare milestone samples against:"
echo "        - v0.1 step300k fallback"
echo "        - v0.1 step500k primary RC"
echo "[v0.2] Watch especially: s3_cocuklar.wav and Turkish-heavy words"
echo "        cocuklar / cicek / seker / uzum / olcum / ogrenciler / buyume"
echo "[v0.2] Planned eval milestones: 50k, 100k, 150k, 200k"
echo "[v0.2] To start the pilot after review, run:"
echo "        bash recipes/issai_piper/train.sh \"$PIPER_DIR\" \"$TRAINING_DIR\" \"$CHECKPOINT_DIR\""
