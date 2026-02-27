#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/autodl-tmp/hybrid_lens_design"
OUT_DIR="${ROOT}/results/full_8mm_20260226_172218"
CONTROL_OUT_DIR="${OUT_DIR}/03_e2e_control"
RUN_LOG="${OUT_DIR}/run.log"
PY="/root/miniconda3/bin/python"
RESUME_EPOCH=1770

echo "[CONTROL] Resume v2 from epoch ${RESUME_EPOCH}" >> "${RUN_LOG}"
"${PY}" "${ROOT}/scripts/train_e2e.py" \
  --config "${ROOT}/configs/exp_f2p8_80mm_e2e_fullsafe.yaml" \
  --hybridlens "${CONTROL_OUT_DIR}/hybridlens_epoch${RESUME_EPOCH}.json" \
  --pretrained "${CONTROL_OUT_DIR}/network_epoch${RESUME_EPOCH}.pth" \
  --start_epoch ${RESUME_EPOCH} \
  --epochs 5000 \
  --freeze_doe 1 \
  --network_type mha_unet \
  --output_dir "${CONTROL_OUT_DIR}" >> "${RUN_LOG}" 2>&1

echo "[CONTROL] E2E Training finished." >> "${RUN_LOG}"
"${PY}" "${ROOT}/scripts/evaluate_e2e.py" \
  --config "${ROOT}/configs/exp_f2p8_80mm_e2e_fullsafe.yaml" \
  --checkpoint "${CONTROL_OUT_DIR}/network_epochfinal.pth" \
  --hybridlens "${CONTROL_OUT_DIR}/hybridlens_epochfinal.json" \
  --output_dir "${CONTROL_OUT_DIR}" >> "${RUN_LOG}" 2>&1

echo "[ALL EXPERIMENTS FINISHED]" >> "${RUN_LOG}"
touch /root/autodl-tmp/ALL_DONE.flag
sleep 600
shutdown -h now
