#!/usr/bin/env bash
# prepare.sh — Build ISSAI manifests, validate, and export to Piper LJSpeech format
# Usage: bash prepare.sh <issai_corpus_dir> <manifest_dir> <piper_dir>
set -euo pipefail

CORPUS_DIR="${1:?Usage: prepare.sh <corpus_dir> <manifest_dir> <piper_dir>}"
MANIFEST_DIR="${2:?Usage: prepare.sh <corpus_dir> <manifest_dir> <piper_dir>}"
PIPER_DIR="${3:?Usage: prepare.sh <corpus_dir> <manifest_dir> <piper_dir>}"

mkdir -p "$MANIFEST_DIR" "$PIPER_DIR"

echo "[prepare] Building ISSAI manifests..."
gokbilge-tts prepare-issai \
    --dataset-dir "$CORPUS_DIR" \
    --out "$MANIFEST_DIR"

echo "[prepare] Validating train manifest..."
gokbilge-tts validate-manifest "$MANIFEST_DIR/train.jsonl"

echo "[prepare] Exporting to Piper LJSpeech format..."
gokbilge-tts export-piper \
    --manifest-dir "$MANIFEST_DIR" \
    --out "$PIPER_DIR"

echo "[prepare] Done. Piper dataset at $PIPER_DIR"
echo "[prepare] Next: bash train.sh $PIPER_DIR <training_dir> <checkpoint_dir>"
