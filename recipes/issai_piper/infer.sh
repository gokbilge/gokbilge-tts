#!/usr/bin/env bash
# infer.sh — Synthesize speech using a trained Piper ONNX model
# Usage: bash infer.sh <model_base> "text" <output.wav>
#
# model_base — path without extension (expects <model_base>.onnx + <model_base>.onnx.json)
# text       — Turkish text to synthesize
# output.wav — output WAV file path
set -euo pipefail

MODEL_BASE="${1:?Usage: infer.sh <model_base> <text> <output.wav>}"
TEXT="${2:?Usage: infer.sh <model_base> <text> <output.wav>}"
OUTPUT="${3:?Usage: infer.sh <model_base> <text> <output.wav>}"

MODEL_ONNX="${MODEL_BASE}.onnx"
MODEL_JSON="${MODEL_BASE}.onnx.json"

echo "[infer] Synthesizing: $TEXT"
echo "$TEXT" | piper \
    --model "$MODEL_ONNX" \
    --config "$MODEL_JSON" \
    --output_file "$OUTPUT"

echo "[infer] Saved: $OUTPUT"
