#!/usr/bin/env bash
# train.sh ? Preprocess Piper LJSpeech dataset and train VITS model
# Usage: bash train.sh <piper_dir> <training_dir> <checkpoint_dir> [resume_checkpoint]
#
# piper_dir          ? output of prepare.sh (wavs/ + metadata.csv + config.json)
# training_dir       ? output of piper_train.preprocess (phoneme-processed dataset)
# checkpoint_dir     ? directory for .ckpt files and train.log
# resume_checkpoint  ? optional checkpoint path or latest/last (default: latest)
set -euo pipefail

PIPER_DIR="${1:?Usage: train.sh <piper_dir> <training_dir> <checkpoint_dir> [resume_checkpoint]}"
TRAINING_DIR="${2:?Usage: train.sh <piper_dir> <training_dir> <checkpoint_dir> [resume_checkpoint]}"
CHECKPOINT_DIR="${3:?Usage: train.sh <piper_dir> <training_dir> <checkpoint_dir> [resume_checkpoint]}"
RESUME_CHECKPOINT="${4:-latest}"

mkdir -p "$TRAINING_DIR" "$CHECKPOINT_DIR"

if [[ "$RESUME_CHECKPOINT" != "latest" && "$RESUME_CHECKPOINT" != "last" && ! -f "$RESUME_CHECKPOINT" ]]; then
    echo "[train] ERROR: resume checkpoint not found: $RESUME_CHECKPOINT" >&2
    exit 1
fi

echo "[train] Resume mode: $RESUME_CHECKPOINT"
echo "[train] Preprocessing dataset (espeak-ng phonemization)..."
python3 -m piper_train.preprocess     --language tr     --input-dir "$PIPER_DIR"     --output-dir "$TRAINING_DIR"     --dataset-format ljspeech     --single-speaker     --sample-rate 22050

echo "[train] Starting VITS training..."
# --max_epochs 10000 is a ceiling, not the actual target.
# With 178k utterances and batch_size=16, each epoch takes about 3h; 10k epochs is only a ceiling.
# Real stop criterion: perceptual quality plateau around 300k-800k steps.
# Evaluate at step milestones with: bash tools/eval_step.sh <sample_dir>
python3 -m piper_train     --dataset-dir "$TRAINING_DIR"     --accelerator gpu     --devices 1     --batch-size 16     --validation-split 0.0     --num-test-examples 0     --max_epochs 10000     --resume_from_checkpoint "$RESUME_CHECKPOINT"     --checkpoint-epochs 100     --precision 32     --default_root_dir "$CHECKPOINT_DIR"     2>&1 | tee "$CHECKPOINT_DIR/train.log"

echo "[train] Done. Checkpoints in $CHECKPOINT_DIR"
echo "[train] Next: bash recipes/issai_piper/export_onnx.sh <checkpoint.ckpt> $TRAINING_DIR <output_name>"
