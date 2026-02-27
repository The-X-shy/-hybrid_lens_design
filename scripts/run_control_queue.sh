#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/hybrid_lens_design"
# Note: we use the hardcoded output dir from the running script to know where to find its files
OUT_DIR="${ROOT}/results/full_8mm_20260226_172218"
CONTROL_OUT_DIR="${OUT_DIR}/03_e2e_control"
RUN_LOG="${OUT_DIR}/run.log"
PY="/root/miniconda3/bin/python"

echo "[CONTROL] Daemon started. Waiting for main run_hybrid_and_e2e_v2.sh to finish..." >> "${RUN_LOG}"

# Wait until the main script finishes (by checking if the workflow completed flag is written or the process is gone)
while true; do
    if grep -q '\[DONE\] Workflow completed.' "${RUN_LOG}"; then
        echo "[CONTROL] Detected main workflow completion!" >> "${RUN_LOG}"
        break
    fi
    # If the process crashed and is no longer running, we also break out to salvage what we can
    if ! pgrep -f "run_hybrid_and_e2e_v2.sh" > /dev/null; then
         # Wait a few seconds to ensure it wasn't a temporary subshell spawn issue
         sleep 10
         if ! pgrep -f "run_hybrid_and_e2e_v2.sh" > /dev/null; then
             echo "[CONTROL] Main script no longer running. Assuming it finished or crashed. Proceeding..." >> "${RUN_LOG}"
             break
         fi
    fi
    sleep 30
done

echo "==================================================" >> "${RUN_LOG}"
echo "[CONTROL] Starting Control Ablation Study (Pure Refractive + E2E)" >> "${RUN_LOG}"
echo "==================================================" >> "${RUN_LOG}"

mkdir -p "${CONTROL_OUT_DIR}"

# E2E Training on Pure GeoLens (no DOE)
echo "[CONTROL] Starting E2E Training (5000 epochs, patience=300)..." >> "${RUN_LOG}"
"${PY}" "${ROOT}/scripts/train_e2e.py" \
  --config "${ROOT}/configs/exp_f2p8_80mm_e2e_fullsafe.yaml" \
  --hybridlens "${OUT_DIR}/01_geolens/final_lens_dummy_doe.json" \
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
