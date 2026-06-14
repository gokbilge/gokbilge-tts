#!/usr/bin/env bash
# prepare.sh — Convert ISSAI corpus to JSONL manifest with G2P phonemes
# Usage: bash prepare.sh <issai_corpus_dir> <output_dir>
set -euo pipefail

CORPUS_DIR="${1:?Usage: prepare.sh <corpus_dir> <output_dir>}"
OUTPUT_DIR="${2:?Usage: prepare.sh <corpus_dir> <output_dir>}"

mkdir -p "$OUTPUT_DIR"

echo "[prepare] Running ISSAI manifest preparation..."
gokbilge-tts prepare-issai \
    --dataset-dir "$CORPUS_DIR" \
    --out "$OUTPUT_DIR"

echo "[prepare] Done. Manifests at $OUTPUT_DIR"
