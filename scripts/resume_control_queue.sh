#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/hybrid_lens_design"
OUT_DIR="${ROOT}/results/full_8mm_20260226_172218"
CONTROL_OUT_DIR="${OUT_DIR}/03_e2e_control"
RUN_LOG="${OUT_DIR}/run.log"
PY="/root/miniconda3/bin/python"

echo "==================================================" >> "${RUN_LOG}"
echo "[CONTROL] Resuming Control Ablation Study (Disk Extension Interruption)" >> "${RUN_LOG}"
echo "==================================================" >> "${RUN_LOG}"

# Find latest checkpoint
# In the log it stopped at 1800, so we should look for network_epoch1800.pth
LAST_EPOCH=1800
CHECKPOINT="${CONTROL_OUT_DIR}/network_epoch${LAST_EPOCH}.pth"
if [ ! -f "$CHECKPOINT" ]; then
    echo "[CONTROL ERROR] Checkpoint ${CHECKPOINT} not found, falling back to 1790" >> "${RUN_LOG}"
    LAST_EPOCH=1790
    CHECKPOINT="${CONTROL_OUT_DIR}/network_epoch${LAST_EPOCH}.pth"
fi

echo "[CONTROL] Resuming from checkpoint ${CHECKPOINT} (Epoch ${LAST_EPOCH})..." >> "${RUN_LOG}"

# Resume E2E Training on Pure GeoLens
"${PY}" "${ROOT}/scripts/train_e2e.py" \
  --config "${ROOT}/configs/exp_f2p8_80mm_e2e_fullsafe.yaml" \
  --hybridlens "${CONTROL_OUT_DIR}/hybridlens_epoch${LAST_EPOCH}.json" \
  --checkpoint "${CHECKPOINT}" \
  --epochs 5000 \
  --freeze_doe 1 \
  --network_type mha_unet \
  --output_dir "${CONTROL_OUT_DIR}" >> "${RUN_LOG}" 2>&1

echo "[CONTROL] E2E Training finished." >> "${RUN_LOG}"

# Evaluate Control E2E
echo "[CONTROL] Evaluating Control E2E Image Metrics" >> "${RUN_LOG}"
"${PY}" "${ROOT}/scripts/evaluate_e2e.py" \
  --config "${ROOT}/configs/exp_f2p8_80mm_e2e_fullsafe.yaml" \
  --checkpoint "${CONTROL_OUT_DIR}/network_epochfinal.pth" \
  --hybridlens "${CONTROL_OUT_DIR}/hybridlens_epochfinal.json" \
  --output_dir "${CONTROL_OUT_DIR}" >> "${RUN_LOG}" 2>&1

echo "[ALL EXPERIMENTS FINISHED]" >> "${RUN_LOG}"
touch /root/autodl-tmp/ALL_DONE.flag

# Wait 10 minutes to allow local scripts to sync before shutting down the instance
sleep 600
shutdown -h now
