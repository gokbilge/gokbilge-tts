#!/usr/bin/env bash
# infer.sh — Synthesize speech from a VITS checkpoint
# Usage: bash infer.sh <checkpoint.pth> "text" <output.wav>
set -euo pipefail

CHECKPOINT="${1:?Usage: infer.sh <checkpoint.pth> <text> <output.wav>}"
TEXT="${2:?Usage: infer.sh <checkpoint.pth> <text> <output.wav>}"
OUTPUT="${3:?Usage: infer.sh <checkpoint.pth> <text> <output.wav>}"
CONFIG="$(dirname "$0")/../../configs/vits_tr_v0_1.yaml"

echo "[infer] Synthesizing: $TEXT"
python -m gokbilge_tts.infer.vits_infer \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --text "$TEXT" \
    --output "$OUTPUT"

echo "[infer] Saved: $OUTPUT"
