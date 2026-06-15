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
# --max_epochs 10000 is a ceiling, not the actual target.
# With 178k utterances and batch_size=16, each epoch takes ~3h; 10k epochs = ~3.4 years.
# Real stop criterion: perceptual quality plateau around 300k–800k steps (~3–8 days).
# Evaluate at step milestones with: bash tools/eval_step.sh <sample_dir>
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
    --default_root_dir "$CHECKPOINT_DIR" \
    2>&1 | tee "$CHECKPOINT_DIR/train.log"

echo "[train] Done. Checkpoints in $CHECKPOINT_DIR"
echo "[train] Next: bash recipes/issai_piper/export_onnx.sh <checkpoint.ckpt> $TRAINING_DIR <output_name>"
