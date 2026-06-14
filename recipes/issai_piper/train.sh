#!/usr/bin/env bash
# train.sh — Preprocess Piper LJSpeech dataset and train VITS model
# Usage: bash train.sh <piper_dir> <training_dir> <checkpoint_dir>
#
# piper_dir     — output of prepare.sh (wavs/ + metadata.csv + config.json)
# training_dir  — output of piper_train.preprocess (phoneme-processed dataset)
# checkpoint_dir — directory for .ckpt files and train.log
set -euo pipefail

PIPER_DIR="${1:?Usage: train.sh <piper_dir> <training_dir> <checkpoint_dir>}"
TRAINING_DIR="${2:?Usage: train.sh <piper_dir> <training_dir> <checkpoint_dir>}"
CHECKPOINT_DIR="${3:?Usage: train.sh <piper_dir> <training_dir> <checkpoint_dir>}"

mkdir -p "$TRAINING_DIR" "$CHECKPOINT_DIR"

echo "[train] Preprocessing dataset (espeak-ng phonemization)..."
python3 -m piper_train.preprocess \
    --language tr \
    --input-dir "$PIPER_DIR" \
    --output-dir "$TRAINING_DIR" \
    --dataset-format ljspeech \
    --single-speaker \
    --sample-rate 22050

echo "[train] Starting VITS training..."
python3 -m piper_train \
    --dataset-dir "$TRAINING_DIR" \
    --accelerator gpu \
    --devices 1 \
    --batch-size 16 \
    --validation-split 0.0 \
    --num-test-examples 0 \
    --max_epochs 10000 \
    --resume_from_checkpoint latest \
    --checkpoint-epochs 100 \
    --precision 32 \
    2>&1 | tee "$CHECKPOINT_DIR/train.log"

echo "[train] Done. Checkpoints in $CHECKPOINT_DIR"
echo "[train] Next: bash export_onnx.sh <checkpoint.ckpt> $PIPER_DIR <output_name>"
