#!/usr/bin/env bash
# export_onnx.sh — Export Piper VITS checkpoint to ONNX for CPU inference
# Usage: bash export_onnx.sh <checkpoint.ckpt> <training_dir> <output_name>
#
# checkpoint.ckpt — .ckpt file produced by train.sh (e.g. last.ckpt)
# training_dir    — output of piper_train.preprocess (contains the final config.json
#                   with phoneme_id_map filled in by espeak-ng)
# output_name     — base name for output files (no extension)
#
# Note: use training_dir/config.json (NOT piper_dir/config.json) because
# piper_train.preprocess fills phoneme_id_map into the training_dir copy.
set -euo pipefail

CHECKPOINT="${1:?Usage: export_onnx.sh <checkpoint.ckpt> <training_dir> <output_name>}"
TRAINING_DIR="${2:?Usage: export_onnx.sh <checkpoint.ckpt> <training_dir> <output_name>}"
OUTPUT_NAME="${3:?Usage: export_onnx.sh <checkpoint.ckpt> <training_dir> <output_name>}"

if [[ ! -f "$TRAINING_DIR/config.json" ]]; then
    echo "[export] ERROR: $TRAINING_DIR/config.json not found." >&2
    echo "[export] Run piper_train.preprocess first (train.sh step 1)." >&2
    exit 1
fi

echo "[export] Exporting ONNX: $CHECKPOINT → ${OUTPUT_NAME}.onnx"
python3 -m piper_train.export_onnx \
    "$CHECKPOINT" \
    "${OUTPUT_NAME}.onnx"

cp "$TRAINING_DIR/config.json" "${OUTPUT_NAME}.onnx.json"

echo "[export] Done: ${OUTPUT_NAME}.onnx + ${OUTPUT_NAME}.onnx.json"
echo "[export] Next: bash recipes/issai_piper/infer.sh ${OUTPUT_NAME} \"Test metni\" output.wav"
