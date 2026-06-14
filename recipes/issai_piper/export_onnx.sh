#!/usr/bin/env bash
# export_onnx.sh — Export Piper VITS checkpoint to ONNX for CPU inference
# Usage: bash export_onnx.sh <checkpoint.ckpt> <piper_dir> <output_name>
#
# checkpoint.ckpt — .ckpt file from train.sh (e.g. last.ckpt)
# piper_dir       — output of prepare.sh (contains config.json)
# output_name     — base name for output files (no extension)
set -euo pipefail

CHECKPOINT="${1:?Usage: export_onnx.sh <checkpoint.ckpt> <piper_dir> <output_name>}"
PIPER_DIR="${2:?Usage: export_onnx.sh <checkpoint.ckpt> <piper_dir> <output_name>}"
OUTPUT_NAME="${3:?Usage: export_onnx.sh <checkpoint.ckpt> <piper_dir> <output_name>}"

echo "[export] Exporting ONNX: $CHECKPOINT → ${OUTPUT_NAME}.onnx"
python3 -m piper_train.export_onnx \
    "$CHECKPOINT" \
    "${OUTPUT_NAME}.onnx"

cp "$PIPER_DIR/config.json" "${OUTPUT_NAME}.onnx.json"

echo "[export] Done: ${OUTPUT_NAME}.onnx + ${OUTPUT_NAME}.onnx.json"
echo "[export] Next: bash infer.sh ${OUTPUT_NAME} \"Test metni\" output.wav"
