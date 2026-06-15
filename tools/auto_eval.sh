#!/usr/bin/env bash
# auto_eval.sh — Monitor training and auto-run eval_step.sh at step milestones
#
# Usage (launch once from repo root, runs as background daemon):
#   nohup bash tools/auto_eval.sh [run_dir] > runs/v0_1_full_001/eval_monitor.log 2>&1 &
#
# Default run_dir: runs/v0_1_full_001
# Checks every 5 minutes. Skips milestones already done (samples/ dir exists with WAVs).
# Safe to restart — already-completed milestones are detected and skipped.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

RUN_DIR="${1:-runs/v0_1_full_001}"
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
TFEVENTS_GLOB="$RUN_DIR/checkpoints/lightning_logs/version_0/events.out.tfevents.*"
CHECK_INTERVAL=300  # seconds between checks

cd "$REPO_ROOT"

# ── Milestone definitions ─────────────────────────────────────────────────────
# Format: "step_count:sample_dir_name"
# Numbers increase monotonically; NN_ prefix determines sort order in samples/.
MILESTONES=(
    "10000:02_v0_1_full_step010k"
    "20000:03_v0_1_full_step020k"
    "50000:04_v0_1_full_step050k"
    "80000:05_v0_1_full_step080k"
    "100000:06_v0_1_full_step100k"
    "120000:07_v0_1_full_step120k"
    "140000:08_v0_1_full_step140k"
    "160000:09_v0_1_full_step160k"
    "200000:10_v0_1_full_step200k"
    "250000:11_v0_1_full_step250k"
    "300000:12_v0_1_full_step300k"
    "400000:13_v0_1_full_step400k"
    "500000:14_v0_1_full_step500k"
    "600000:15_v0_1_full_step600k"
    "700000:16_v0_1_full_step700k"
    "800000:17_v0_1_full_step800k"
)

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

get_current_step() {
    local tf
    tf="$(ls $TFEVENTS_GLOB 2>/dev/null | head -1)" || true
    if [[ -z "$tf" ]]; then
        echo 0
        return
    fi
    python3 - "$tf" <<'PYEOF'
import sys
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
f = sys.argv[1]
step = 0
try:
    for e in LegacyEventFileLoader(f).Load():
        if e.step > step:
            step = e.step
except Exception:
    pass
print(step)
PYEOF
}

milestone_done() {
    local sample_dir="$1"
    local out_dir="$REPO_ROOT/samples/$sample_dir"
    # Done if at least one WAV exists in the samples dir
    [[ -d "$out_dir" ]] && ls "$out_dir"/*.wav &>/dev/null
}

# ── Main loop ─────────────────────────────────────────────────────────────────
log "auto_eval started. Run dir: $RUN_DIR"
log "Monitoring ${#MILESTONES[@]} milestones. Check interval: ${CHECK_INTERVAL}s"
log "Milestones: $(for m in "${MILESTONES[@]}"; do echo -n "${m%%:*} "; done)"

while true; do
    CURRENT_STEP="$(get_current_step)"
    log "Current step: $CURRENT_STEP"

    for entry in "${MILESTONES[@]}"; do
        TARGET_STEP="${entry%%:*}"
        SAMPLE_DIR="${entry##*:}"

        if (( CURRENT_STEP < TARGET_STEP )); then
            continue  # not reached yet
        fi

        if milestone_done "$SAMPLE_DIR"; then
            continue  # already evaluated
        fi

        log "━━━ Milestone reached: step $CURRENT_STEP >= $TARGET_STEP → $SAMPLE_DIR ━━━"
        if bash "$REPO_ROOT/tools/eval_step.sh" "$SAMPLE_DIR" "$RUN_DIR"; then
            log "✓ Eval done: samples/$SAMPLE_DIR/"
        else
            log "✗ Eval FAILED for $SAMPLE_DIR (exit $?). Will retry next check."
        fi
    done

    # Check if all milestones are done
    remaining=0
    for entry in "${MILESTONES[@]}"; do
        SAMPLE_DIR="${entry##*:}"
        milestone_done "$SAMPLE_DIR" || (( remaining++ )) || true
    done
    if (( remaining == 0 )); then
        log "All milestones completed. auto_eval exiting."
        exit 0
    fi

    log "Next check in ${CHECK_INTERVAL}s. Remaining milestones: $remaining"
    sleep "$CHECK_INTERVAL"
done
