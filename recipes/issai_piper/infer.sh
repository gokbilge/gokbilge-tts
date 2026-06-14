#!/usr/bin/env bash
# infer.sh — Synthesize speech with a trained Piper model
# Usage: bash infer.sh <model_dir> "text" <output.wav>
set -euo pipefail

MODEL_DIR="${1:?Usage: infer.sh <model_dir> <text> <output.wav>}"
TEXT="${2:?Usage: infer.sh <model_dir> <text> <output.wav>}"
OUTPUT="${3:?Usage: infer.sh <model_dir> <text> <output.wav>}"

MODEL_ONNX="$MODEL_DIR.onnx"
MODEL_JSON="$MODEL_DIR.onnx.json"

echo "[infer] Synthesizing: $TEXT"
echo "$TEXT" | piper \
    --model "$MODEL_ONNX" \
    --config "$MODEL_JSON" \
    --output_file "$OUTPUT"

echo "[infer] Saved: $OUTPUT"
