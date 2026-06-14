#!/usr/bin/env bash
# train.sh — Train VITS on ISSAI corpus
# Usage: bash train.sh <data_dir> <checkpoint_dir>
set -euo pipefail

DATA_DIR="${1:?Usage: train.sh <data_dir> <checkpoint_dir>}"
CHECKPOINT_DIR="${2:?Usage: train.sh <data_dir> <checkpoint_dir>}"
VITS_SRC="${VITS_SRC:-vits_src}"
CONFIG="$(dirname "$0")/../../configs/vits_tr_v0_1.yaml"

mkdir -p "$CHECKPOINT_DIR"

echo "[train] Starting VITS training with $CONFIG ..."
python "$VITS_SRC/train.py" \
    -c "$CONFIG" \
    -m "$CHECKPOINT_DIR" \
    2>&1 | tee "$CHECKPOINT_DIR/train.log"

echo "[train] Done."
