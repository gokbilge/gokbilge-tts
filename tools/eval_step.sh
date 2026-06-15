#!/usr/bin/env bash
# eval_step.sh — Export ONNX from last.ckpt and generate 5 benchmark WAVs for a step milestone
#
# Usage:
#   bash tools/eval_step.sh <sample_dir> [run_dir]
#
# Arguments:
#   sample_dir — samples/ subdirectory name, e.g. 03_v0_1_full_step100k
#   run_dir    — training run directory (default: runs/v0_1_full_001)
#
# What it does:
#   1. Exports ONNX from <run_dir>/checkpoints/lightning_logs/version_0/checkpoints/last.ckpt
#   2. Saves model to <run_dir>/onnx/<sample_dir>.{onnx,onnx.json}
#   3. Synthesizes the 5 fixed benchmark sentences → samples/<sample_dir>/s{1..5}_*.wav
#
# Example (run at ~100k steps):
#   bash tools/eval_step.sh 03_v0_1_full_step100k
#
# Benchmark sentences (fixed — see CLAUDE.md § Evaluation Rule):
#   s1: "Bugün hava çok güzel."
#   s2: "Türkiye Cumhuriyeti bin dokuz yüz yirmi üç yılında kuruldu."
#   s3: "Çocuklar çiçek, şeker ve üzüm yedi."
#   s4: "Öğrenciler ölçüm sonuçlarını değerlendirdi."
#   s5: "Şirket yüzde otuz beş büyüme açıkladı."
#
# Output WAVs are committed to samples/ as training milestones.
# ONNX files live under runs/ (not committed).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

SAMPLE_DIR="${1:?Usage: eval_step.sh <sample_dir> [run_dir]   e.g. 03_v0_1_full_step100k}"
RUN_DIR="${2:-runs/v0_1_full_001}"

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

CKPT_DIR="$RUN_DIR/checkpoints/lightning_logs/version_0/checkpoints"
LAST_CKPT="$CKPT_DIR/last.ckpt"
TRAINING_DIR="$RUN_DIR/training"
ONNX_DIR="$RUN_DIR/onnx"
MODEL_BASE="$ONNX_DIR/$SAMPLE_DIR"
OUT_DIR="$REPO_ROOT/samples/$SAMPLE_DIR"

if [[ ! -f "$LAST_CKPT" ]]; then
    echo "[eval] ERROR: last.ckpt not found at $LAST_CKPT" >&2
    echo "[eval] Training may not have completed an epoch yet." >&2
    exit 1
fi

mkdir -p "$ONNX_DIR" "$OUT_DIR"

# ── 1. Export ONNX ────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "[eval] Step 1/2: Exporting ONNX"
echo "  checkpoint: $LAST_CKPT"
echo "  output:     ${MODEL_BASE}.onnx"
echo "================================================================"
bash "$REPO_ROOT/recipes/issai_piper/export_onnx.sh" \
    "$LAST_CKPT" \
    "$TRAINING_DIR" \
    "$MODEL_BASE"

# ── 2. Synthesize 5 benchmark sentences ──────────────────────────────────────
echo ""
echo "================================================================"
echo "[eval] Step 2/2: Synthesizing 5 benchmark sentences → $OUT_DIR"
echo "================================================================"

declare -a STEMS=("s1_bugun_hava" "s2_turkiye_cumh" "s3_cocuklar" "s4_ogrenciler" "s5_sirket")
declare -a TEXTS=(
    "Bugün hava çok güzel."
    "Türkiye Cumhuriyeti bin dokuz yüz yirmi üç yılında kuruldu."
    "Çocuklar çiçek, şeker ve üzüm yedi."
    "Öğrenciler ölçüm sonuçlarını değerlendirdi."
    "Şirket yüzde otuz beş büyüme açıkladı."
)

for i in 0 1 2 3 4; do
    stem="${STEMS[$i]}"
    text="${TEXTS[$i]}"
    out="$OUT_DIR/${stem}.wav"
    echo ""
    echo "[eval] Sentence $((i+1))/5: $text"
    bash "$REPO_ROOT/recipes/issai_piper/infer.sh" \
        "$MODEL_BASE" \
        "$text" \
        "$out"
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "[eval] Done: $SAMPLE_DIR"
echo "  ONNX:     ${MODEL_BASE}.onnx"
echo "  WAVs:     $OUT_DIR/"
ls -lh "$OUT_DIR/"
echo ""
echo "Next:"
echo "  1. Listen to WAVs in samples/$SAMPLE_DIR/"
echo "  2. Add metadata block to samples/README.md"
echo "  3. git add samples/$SAMPLE_DIR/ samples/README.md && git commit"
echo "================================================================"
