#!/usr/bin/env bash
# train.sh — Train VITS on ISSAI manifest using Piper training fork
# Usage: bash train.sh <data_dir> <checkpoint_dir>
set -euo pipefail

DATA_DIR="${1:?Usage: train.sh <data_dir> <checkpoint_dir>}"
CHECKPOINT_DIR="${2:?Usage: train.sh <data_dir> <checkpoint_dir>}"
CONFIG="$(dirname "$0")/../../configs/vits_tr_v0_1.yaml"

mkdir -p "$CHECKPOINT_DIR"

echo "[train] Starting VITS training..."
python -m piper_train \
    --dataset-dir "$DATA_DIR" \
    --accelerator gpu \
    --devices 1 \
    --batch-size 16 \
    --validation-split 0.01 \
    --num-test-examples 5 \
    --max-epochs 1000 \
    --resume_from_checkpoint latest \
    --checkpoint-epochs 100 \
    --quality medium \
    --config "$CONFIG" \
    2>&1 | tee "$CHECKPOINT_DIR/train.log"

echo "[train] Done. Checkpoints in $CHECKPOINT_DIR"
