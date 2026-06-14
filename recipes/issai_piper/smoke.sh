#!/usr/bin/env bash
# smoke.sh — 100-utterance end-to-end smoke training run (correctness check only)
# Usage: bash recipes/issai_piper/smoke.sh <issai_dir> <run_dir>
#
# issai_dir — path to ISSAI_TSC_218 corpus (or its parent directory)
# run_dir   — working directory for this run (created if absent)
#
# Produces:
#   <run_dir>/manifests/    — JSONL manifests (100-utterance training subset)
#   <run_dir>/piper/        — Piper LJSpeech export (metadata.csv, wavs/, config.json)
#   <run_dir>/training/     — piper_train.preprocess output (phoneme IDs + final config.json)
#   <run_dir>/checkpoints/  — .ckpt files and train.log
#
# Does NOT run ONNX export automatically.
# To export after a successful run:
#   bash recipes/issai_piper/export_onnx.sh \
#       <run_dir>/checkpoints/last.ckpt \
#       <run_dir>/training \
#       gokbilge_tr_smoke
set -euo pipefail

ISSAI_DIR="${1:?Usage: smoke.sh <issai_dir> <run_dir>}"
RUN_DIR="${2:?Usage: smoke.sh <issai_dir> <run_dir>}"

MANIFESTS="$RUN_DIR/manifests"
PIPER="$RUN_DIR/piper"
TRAINING="$RUN_DIR/training"
CHECKPOINTS="$RUN_DIR/checkpoints"

mkdir -p "$MANIFESTS" "$PIPER" "$TRAINING" "$CHECKPOINTS"

echo "================================================================"
echo "[smoke] Gokbilge TTS — smoke training run"
echo "[smoke] ISSAI:  $ISSAI_DIR"
echo "[smoke] Run:    $RUN_DIR"
echo "================================================================"

# ── Step 1: Prepare manifests ────────────────────────────────────────────────
echo ""
echo "[smoke] Step 1/4: Preparing ISSAI manifests..."
gokbilge-tts prepare-issai \
    --dataset-dir "$ISSAI_DIR" \
    --out "$MANIFESTS"

echo "[smoke] Validating train manifest..."
gokbilge-tts validate-manifest "$MANIFESTS/train.jsonl"

# ── Step 2: Export 100-utterance Piper subset ────────────────────────────────
echo ""
echo "[smoke] Step 2/4: Exporting 100-utterance Piper subset (--limit 100)..."
gokbilge-tts export-piper \
    --manifest-dir "$MANIFESTS" \
    --out "$PIPER" \
    --limit 100

echo "[smoke] Validating Piper export..."
gokbilge-tts validate-piper "$PIPER"

# ── Step 3: Preprocess (espeak-ng phonemization) ─────────────────────────────
echo ""
echo "[smoke] Step 3/4: Running piper_train.preprocess (espeak-ng Turkish)..."
echo "[smoke] Requires: espeak-ng installed ('sudo apt-get install espeak-ng')"
python3 -m piper_train.preprocess \
    --language tr \
    --input-dir "$PIPER" \
    --output-dir "$TRAINING" \
    --dataset-format ljspeech \
    --single-speaker \
    --sample-rate 22050

# ── Step 4: Short smoke training run ─────────────────────────────────────────
echo ""
echo "[smoke] Step 4/4: Smoke training (5 epochs, batch=4)..."
echo "[smoke] Using --accelerator gpu. Change to 'cpu' if no GPU available."
python3 -m piper_train \
    --dataset-dir "$TRAINING" \
    --accelerator gpu \
    --devices 1 \
    --batch-size 4 \
    --validation-split 0.0 \
    --num-test-examples 0 \
    --max_epochs 5 \
    --checkpoint-epochs 5 \
    --precision 32 \
    2>&1 | tee "$CHECKPOINTS/train.log"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "[smoke] Smoke run complete."
echo "  Manifests:    $MANIFESTS"
echo "  Piper export: $PIPER"
echo "  Training dir: $TRAINING"
echo "  Checkpoints:  $CHECKPOINTS"
echo "  Log:          $CHECKPOINTS/train.log"
echo ""
echo "To export ONNX (provide the checkpoint path):"
echo "  bash recipes/issai_piper/export_onnx.sh \\"
echo "      $CHECKPOINTS/last.ckpt \\"
echo "      $TRAINING \\"
echo "      gokbilge_tr_smoke"
echo "================================================================"
