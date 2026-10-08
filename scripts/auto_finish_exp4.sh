#!/usr/bin/env bash
# =============================================================================
# Automation script: Monitor Exp4 (Hybrid+DOE + UNet) training completion,
# then evaluate, download results, and shut down the remote server.
# =============================================================================
set -euo pipefail

# --- Config ---
REMOTE_HOST="connect.westb.seetacloud.com"
REMOTE_PORT=14173
REMOTE_PASS="XQCVslEXeJIG"
REMOTE_PROJECT="/root/autodl-tmp/hybrid_lens_design"
REMOTE_OUTDIR="${REMOTE_PROJECT}/results/full_8mm_20260226_172218/03_e2e_unet_hybrid"
REMOTE_LOG="${REMOTE_PROJECT}/results/full_8mm_20260226_172218/03_e2e_unet_hybrid_nohup.log"
REMOTE_LENS="${REMOTE_PROJECT}/results/full_8mm_20260226_172218/02_hybrid/hybridlens_final.json"

LOCAL_PROJECT="/Users/lilin/Desktop/hybrid_lens_design"
LOCAL_RESULTS="${LOCAL_PROJECT}/results/8mm_final_results_summary/04_unet_hybrid"

POLL_INTERVAL=120  # seconds between polls
LOGFILE="${LOCAL_PROJECT}/auto_finish_exp4.log"

SSH_CMD="sshpass -p '${REMOTE_PASS}' ssh -p ${REMOTE_PORT} -o ConnectTimeout=15 -o StrictHostKeyChecking=no root@${REMOTE_HOST}"
SCP_CMD="sshpass -p '${REMOTE_PASS}' scp -P ${REMOTE_PORT} -o ConnectTimeout=15 -o StrictHostKeyChecking=no"

# --- Helpers ---
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOGFILE"
}

remote_exec() {
    sshpass -p "${REMOTE_PASS}" ssh -p "${REMOTE_PORT}" \
        -o ConnectTimeout=15 -o StrictHostKeyChecking=no \
        "root@${REMOTE_HOST}" \
        "export PATH=/root/miniconda3/bin:\$PATH && $1"
}

notify() {
    # macOS notification
    osascript -e "display notification \"$1\" with title \"Exp4 Automation\"" 2>/dev/null || true
    # Also beep
    afplay /System/Library/Sounds/Glass.aiff 2>/dev/null || true
}

# --- Start ---
log "=========================================="
log "Exp4 automation script started"
log "Polling every ${POLL_INTERVAL}s for training completion"
log "=========================================="

# =============================================================================
# PHASE 1: Poll until training is done
# =============================================================================
log "PHASE 1: Waiting for training to complete..."

while true; do
    # Check if the training process is still running
    PROC=$(remote_exec "ps aux | grep 'train_e2e.*03_e2e_unet_hybrid' | grep -v grep | head -1" 2>/dev/null || true)

    if [ -n "$PROC" ]; then
        # Still running — grab latest epoch info
        LAST_EPOCH=$(remote_exec "grep '^Epoch [0-9]*:' ${REMOTE_LOG} | tail -1" 2>/dev/null || echo "unknown")
        VAL_MSE=$(remote_exec "grep 'Validation MSE' ${REMOTE_LOG} | tail -1" 2>/dev/null || echo "unknown")
        log "Still training... ${LAST_EPOCH} | ${VAL_MSE}"
        sleep ${POLL_INTERVAL}
        continue
    fi

    # Process not running — check if it completed successfully
    FINAL_PTH=$(remote_exec "ls -lh ${REMOTE_OUTDIR}/network_epochfinal.pth 2>/dev/null" || true)
    COMPLETION=$(remote_exec "grep -E 'End-to-end optimization complete|Early stopping at epoch' ${REMOTE_LOG} 2>/dev/null | tail -1" || true)

    if [ -n "$FINAL_PTH" ] && [ -n "$COMPLETION" ]; then
        log "Training COMPLETED: ${COMPLETION}"
        log "Final checkpoint found: ${FINAL_PTH}"
        notify "Training complete! Starting evaluation..."
        break
    fi

    # Process not running but no final checkpoint — something went wrong
    LAST_LINES=$(remote_exec "tail -10 ${REMOTE_LOG}" 2>/dev/null || echo "Could not read log")
    log "WARNING: Process not running but no final checkpoint found!"
    log "Last log lines: ${LAST_LINES}"
    log "Retrying in ${POLL_INTERVAL}s in case of transient SSH issue..."
    sleep ${POLL_INTERVAL}
done

# =============================================================================
# PHASE 2: Run evaluation with INITIAL hybrid lens (not drifted final)
# =============================================================================
log "PHASE 2: Running evaluation..."

# CRITICAL: Use initial lens (hybridlens_final.json from step 02) not the
# drifted hybridlens_epochfinal.json from training output.
# Use unet_control config to get network.type=unet in evaluator.
EVAL_CMD="cd ${REMOTE_PROJECT} && python3 scripts/evaluate_e2e.py \
    --config configs/exp_f2p8_80mm_e2e_unet_control.yaml \
    --checkpoint ${REMOTE_OUTDIR}/network_epochfinal.pth \
    --hybridlens ${REMOTE_LENS} \
    --output_dir ${REMOTE_OUTDIR}/"

log "Eval command: ${EVAL_CMD}"
EVAL_OUTPUT=$(remote_exec "$EVAL_CMD" 2>&1) || {
    log "ERROR: Evaluation failed!"
    log "$EVAL_OUTPUT"
    notify "Evaluation FAILED! Check log."
    exit 1
}
log "Evaluation output:"
log "$EVAL_OUTPUT"

# Verify eval_metrics.json was created
EVAL_METRICS=$(remote_exec "cat ${REMOTE_OUTDIR}/eval_metrics.json 2>/dev/null" || true)
if [ -z "$EVAL_METRICS" ]; then
    log "ERROR: eval_metrics.json not found after evaluation!"
    notify "Evaluation produced no metrics! Check log."
    exit 1
fi
log "Evaluation metrics: ${EVAL_METRICS}"
notify "Evaluation done! Downloading results..."

# =============================================================================
# PHASE 3: Download results to local
# =============================================================================
log "PHASE 3: Downloading results..."

mkdir -p "${LOCAL_RESULTS}/logs"

# List of files to download
FILES_TO_DOWNLOAD=(
    "eval_metrics.json"
    "network_epochfinal.pth"
    "hybridlens_epochfinal.json"
)

for f in "${FILES_TO_DOWNLOAD[@]}"; do
    log "Downloading ${f}..."
    sshpass -p "${REMOTE_PASS}" scp -P "${REMOTE_PORT}" \
        -o ConnectTimeout=15 -o StrictHostKeyChecking=no \
        "root@${REMOTE_HOST}:${REMOTE_OUTDIR}/${f}" \
        "${LOCAL_RESULTS}/" 2>&1 || log "WARNING: Failed to download ${f}"
done

# Download eval example PNGs
log "Downloading eval PNG examples..."
remote_exec "ls ${REMOTE_OUTDIR}/eval_example_*.png 2>/dev/null" | while read -r pngpath; do
    fname=$(basename "$pngpath")
    log "  ${fname}"
    sshpass -p "${REMOTE_PASS}" scp -P "${REMOTE_PORT}" \
        -o ConnectTimeout=15 -o StrictHostKeyChecking=no \
        "root@${REMOTE_HOST}:${pngpath}" \
        "${LOCAL_RESULTS}/" 2>&1 || log "WARNING: Failed to download ${fname}"
done

# Download training log
log "Downloading training log..."
sshpass -p "${REMOTE_PASS}" scp -P "${REMOTE_PORT}" \
    -o ConnectTimeout=15 -o StrictHostKeyChecking=no \
    "root@${REMOTE_HOST}:${REMOTE_LOG}" \
    "${LOCAL_RESULTS}/train.log" 2>&1 || log "WARNING: Failed to download training log"

# Download TensorBoard logs if present
log "Downloading TensorBoard logs..."
sshpass -p "${REMOTE_PASS}" scp -r -P "${REMOTE_PORT}" \
    -o ConnectTimeout=15 -o StrictHostKeyChecking=no \
    "root@${REMOTE_HOST}:${REMOTE_OUTDIR}/logs/" \
    "${LOCAL_RESULTS}/logs/" 2>&1 || log "WARNING: No TensorBoard logs found"

# Download any saved .pth checkpoints (epoch 500, 1000, etc.)
log "Downloading intermediate .pth checkpoints..."
remote_exec "ls ${REMOTE_OUTDIR}/network_epoch*.pth 2>/dev/null" 2>/dev/null | while read -r pthpath; do
    fname=$(basename "$pthpath")
    if [ "$fname" != "network_epochfinal.pth" ]; then
        log "  ${fname}"
        sshpass -p "${REMOTE_PASS}" scp -P "${REMOTE_PORT}" \
            -o ConnectTimeout=15 -o StrictHostKeyChecking=no \
            "root@${REMOTE_HOST}:${pthpath}" \
            "${LOCAL_RESULTS}/" 2>&1 || log "WARNING: Failed to download ${fname}"
    fi
done

log "Download complete. Local results:"
ls -lhR "${LOCAL_RESULTS}" >> "$LOGFILE" 2>&1
ls -lh "${LOCAL_RESULTS}"

notify "Download complete! Shutting down server..."

# =============================================================================
# PHASE 4: Shut down remote server
# =============================================================================
log "PHASE 4: Shutting down remote server..."
log "Sending shutdown command..."

remote_exec "shutdown -h now" 2>&1 || true

log "Shutdown command sent."
notify "Server shutdown initiated. All done!"

# =============================================================================
# DONE
# =============================================================================
log "=========================================="
log "ALL PHASES COMPLETE"
log "Results saved to: ${LOCAL_RESULTS}"
log "Metrics: ${EVAL_METRICS}"
log "=========================================="

# Print summary
echo ""
echo "============================================"
echo "  EXP4 AUTOMATION COMPLETE"
echo "============================================"
echo "Results: ${LOCAL_RESULTS}"
echo "Metrics: ${EVAL_METRICS}"
echo "Server: shutdown initiated"
echo "Log: ${LOGFILE}"
echo "============================================"
