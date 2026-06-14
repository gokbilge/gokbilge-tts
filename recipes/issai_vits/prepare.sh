#!/usr/bin/env bash
# prepare.sh — Prepare ISSAI dataset for VITS training
# Usage: bash prepare.sh <issai_corpus_dir> <output_dir>
set -euo pipefail

CORPUS_DIR="${1:?Usage: prepare.sh <corpus_dir> <output_dir>}"
OUTPUT_DIR="${2:?Usage: prepare.sh <corpus_dir> <output_dir>}"

mkdir -p "$OUTPUT_DIR"

echo "[prepare] Building manifest..."
python -m gokbilge_tts.datasets.prepare_issai \
    --corpus-dir "$CORPUS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --manifest-name issai_manifest.jsonl

echo "[prepare] Done. Manifest at $OUTPUT_DIR/issai_manifest.jsonl"
