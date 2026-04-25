#!/usr/bin/env bash
# TraFix v5 — Staged Training Orchestrator
# ==========================================
# Runs Stage 1 → Stage 2 → Stage 3 sequentially.
# Each stage writes a .done sentinel file only after fully succeeding.
# Resume skips based on the sentinel, not the checkpoint, so a crash that
# leaves a partial checkpoint will not cause the stage to be skipped.
#
# To force a stage to re-run: delete its sentinel file, e.g.
#   rm checkpoints/stage1.done
#
# Usage:
#   ./run_training.sh
#   ./run_training.sh --episodes1 300 --episodes2 300 --episodes3 2000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINTS_DIR="$SCRIPT_DIR/checkpoints"
LOGS_DIR="$SCRIPT_DIR/logs"

PYTHON="${PYTHON:-python}"

# ── Parse optional overrides ──
EPISODES1=200
EPISODES2=200
EPISODES3=1000
GUI_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --episodes1) EPISODES1="$2"; shift 2 ;;
        --episodes2) EPISODES2="$2"; shift 2 ;;
        --episodes3) EPISODES3="$2"; shift 2 ;;
        --gui)       GUI_FLAG="--gui"; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Create directories ──
mkdir -p "$CHECKPOINTS_DIR" "$LOGS_DIR"

TRAINING_START=$(date +%s)

echo "============================================================"
echo "  TraFix v5 — Staged Training Pipeline"
echo "  Checkpoints : $CHECKPOINTS_DIR"
echo "  Logs        : $LOGS_DIR"
echo "  Started     : $(date)"
echo "============================================================"

# ════════════════════════════════════════════════
#  Stage 1: GRU Pretraining
# ════════════════════════════════════════════════

STAGE1_DONE="$CHECKPOINTS_DIR/stage1.done"
STAGE1_CKP="$CHECKPOINTS_DIR/stage1_gru.pt"

if [ -f "$STAGE1_DONE" ]; then
    echo ""
    echo "Stage 1 already complete, skipping."
    echo "  (Delete $STAGE1_DONE to re-run)"
else
    echo ""
    echo "------------------------------------------------------------"
    echo "  [$(date '+%H:%M:%S')] Stage 1: GRU Temporal Encoder Pretraining"
    echo "------------------------------------------------------------"
    STAGE1_START=$(date +%s)

    "$PYTHON" "$SCRIPT_DIR/stage1_pretrain_gru.py" \
        --episodes "$EPISODES1" \
        $GUI_FLAG \
        2>&1 | tee "$LOGS_DIR/stage1.log"

    if [ ! -f "$STAGE1_CKP" ]; then
        echo "ERROR: Stage 1 checkpoint not found after training: $STAGE1_CKP" >&2
        exit 1
    fi

    date -Iseconds > "$STAGE1_DONE"
    STAGE1_END=$(date +%s)
    echo "  [$(date '+%H:%M:%S')] Stage 1 finished in $((STAGE1_END - STAGE1_START))s"
fi

# ════════════════════════════════════════════════
#  Stage 2: GATConv + Trunk Pretraining
# ════════════════════════════════════════════════

STAGE2_DONE="$CHECKPOINTS_DIR/stage2.done"
STAGE2_GATCONV_CKP="$CHECKPOINTS_DIR/stage2_gatconv.pt"
STAGE2_TRUNK_CKP="$CHECKPOINTS_DIR/stage2_trunk.pt"

if [ -f "$STAGE2_DONE" ]; then
    echo ""
    echo "Stage 2 already complete, skipping."
    echo "  (Delete $STAGE2_DONE to re-run)"
else
    echo ""
    echo "------------------------------------------------------------"
    echo "  [$(date '+%H:%M:%S')] Stage 2: GATConv + MLP Trunk Pretraining"
    echo "------------------------------------------------------------"
    STAGE2_START=$(date +%s)

    "$PYTHON" "$SCRIPT_DIR/stage2_pretrain_gatconv.py" \
        --episodes "$EPISODES2" \
        $GUI_FLAG \
        2>&1 | tee "$LOGS_DIR/stage2.log"

    if [ ! -f "$STAGE2_GATCONV_CKP" ] || [ ! -f "$STAGE2_TRUNK_CKP" ]; then
        echo "ERROR: Stage 2 checkpoints not found after training." >&2
        exit 1
    fi

    date -Iseconds > "$STAGE2_DONE"
    STAGE2_END=$(date +%s)
    echo "  [$(date '+%H:%M:%S')] Stage 2 finished in $((STAGE2_END - STAGE2_START))s"
fi

# ════════════════════════════════════════════════
#  Stage 3: Full PPO Training
# ════════════════════════════════════════════════

STAGE3_DONE="$CHECKPOINTS_DIR/stage3.done"
STAGE3_CKP="$CHECKPOINTS_DIR/trafix_v5_final.pt"

if [ -f "$STAGE3_DONE" ]; then
    echo ""
    echo "Stage 3 already complete, skipping."
    echo "  (Delete $STAGE3_DONE to re-run)"
else
    echo ""
    echo "------------------------------------------------------------"
    echo "  [$(date '+%H:%M:%S')] Stage 3: Full PPO Training"
    echo "------------------------------------------------------------"
    STAGE3_START=$(date +%s)

    "$PYTHON" "$SCRIPT_DIR/stage3_train_ppo.py" \
        --episodes "$EPISODES3" \
        $GUI_FLAG \
        2>&1 | tee "$LOGS_DIR/stage3.log"

    if [ ! -f "$STAGE3_CKP" ]; then
        echo "ERROR: Final checkpoint not found after training: $STAGE3_CKP" >&2
        exit 1
    fi

    date -Iseconds > "$STAGE3_DONE"
    STAGE3_END=$(date +%s)
    echo "  [$(date '+%H:%M:%S')] Stage 3 finished in $((STAGE3_END - STAGE3_START))s"
fi

# ════════════════════════════════════════════════
#  Summary
# ════════════════════════════════════════════════

TRAINING_END=$(date +%s)
TOTAL_ELAPSED=$((TRAINING_END - TRAINING_START))
TOTAL_MINUTES=$((TOTAL_ELAPSED / 60))
TOTAL_SECONDS=$((TOTAL_ELAPSED % 60))

echo ""
echo "============================================================"
echo "  Training pipeline complete."
echo "  Total elapsed time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo "  Final model: $STAGE3_CKP"
echo "  Logs: $LOGS_DIR"
echo "============================================================"
