#!/usr/bin/env bash
# export_onnx.sh — Export Piper VITS checkpoint to ONNX
# Usage: bash export_onnx.sh <checkpoint_path> <output_model_name>
set -euo pipefail

CHECKPOINT="${1:?Usage: export_onnx.sh <checkpoint_path> <output_name>}"
OUTPUT_NAME="${2:?Usage: export_onnx.sh <checkpoint_path> <output_name>}"

echo "[export] Exporting to ONNX: $OUTPUT_NAME"
python -m piper_train.export_onnx \
    "$CHECKPOINT" \
    "${OUTPUT_NAME}.onnx"

cp "$(dirname "$0")/../../configs/piper_tr_v0_1.json" "${OUTPUT_NAME}.onnx.json"

echo "[export] Done: ${OUTPUT_NAME}.onnx + ${OUTPUT_NAME}.onnx.json"
