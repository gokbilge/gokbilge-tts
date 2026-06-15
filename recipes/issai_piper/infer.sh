#!/usr/bin/env bash
# infer.sh — Synthesize speech using a trained Piper ONNX model
# Usage: bash infer.sh <model_base> "text" <output.wav>
#
# model_base — path without extension (expects <model_base>.onnx + <model_base>.onnx.json)
# text       — Turkish text to synthesize
# output.wav — output WAV file path
#
# Backend selection (default: infer_onnx):
#   PIPER_BACKEND=infer_onnx  — use piper_train.infer_onnx via tools/piper_infer.py (default)
#   PIPER_BACKEND=piper       — use the external piper CLI binary (must be on PATH)
#
# The infer_onnx backend requires:
#   - piper_phonemize stub deployed to site-packages (tools/piper_phonemize_stub.py)
#   - piper_train installed from source (PIPER_SRC or /home/hcfk/piper-src/src/python)
#   - espeak-ng on PATH
set -euo pipefail

MODEL_BASE="${1:?Usage: infer.sh <model_base> <text> <output.wav>}"
TEXT="${2:?Usage: infer.sh <model_base> <text> <output.wav>}"
OUTPUT="${3:?Usage: infer.sh <model_base> <text> <output.wav>}"

MODEL_ONNX="${MODEL_BASE}.onnx"
MODEL_JSON="${MODEL_BASE}.onnx.json"

BACKEND="${PIPER_BACKEND:-infer_onnx}"

echo "[infer] Synthesizing: $TEXT"
echo "[infer] Model: $MODEL_ONNX"
echo "[infer] Backend: $BACKEND"

if [[ "$BACKEND" == "piper" ]]; then
    # External piper CLI — requires piper binary on PATH
    if ! command -v piper &>/dev/null; then
        echo "[infer] ERROR: PIPER_BACKEND=piper but 'piper' binary not found on PATH." >&2
        echo "[infer] Install from https://github.com/rhasspy/piper/releases or use PIPER_BACKEND=infer_onnx." >&2
        exit 1
    fi
    echo "$TEXT" | piper \
        --model "$MODEL_ONNX" \
        --config "$MODEL_JSON" \
        --output_file "$OUTPUT"
else
    # Default: piper_train.infer_onnx via tools/piper_infer.py
    # Resolve tools/piper_infer.py relative to this script's repo root
    SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
    REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
    PIPER_INFER="$REPO_ROOT/tools/piper_infer.py"

    if [[ ! -f "$PIPER_INFER" ]]; then
        echo "[infer] ERROR: $PIPER_INFER not found." >&2
        exit 1
    fi

    python3 "$PIPER_INFER" "$MODEL_ONNX" "$OUTPUT" "$TEXT"
fi

echo "[infer] Saved: $OUTPUT"
