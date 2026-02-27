#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/hybrid_lens_design"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT}/results/full_8mm_${TS}"
RUN_LOG="${OUT_DIR}/run.log"
PY="/root/miniconda3/bin/python"

mkdir -p "${OUT_DIR}"

# Step 0: Wait for GeoLens to finish
echo "[INFO] Waiting for GeoLens to finish..." | tee -a "${RUN_LOG}"
while pgrep -f "train_geolens.py" > /dev/null; do
    sleep 10
done
echo "[INFO] GeoLens finished." | tee -a "${RUN_LOG}"

# Link GeoLens final to OUT_DIR for consistency
mkdir -p "${OUT_DIR}/01_geolens"
cp -r ${ROOT}/results/geolens8mm_1200/* "${OUT_DIR}/01_geolens/"

# Step 1: Hybrid Training (3000 iters)
echo "[INFO] Starting Hybrid Training (3000 iterations)..." | tee -a "${RUN_LOG}"
"${PY}" "${ROOT}/scripts/train_hybrid.py" \
  --config "${ROOT}/configs/exp_f2p8_80mm_hybrid.yaml" \
  --geolens "${OUT_DIR}/01_geolens/final_lens.json" \
  --iterations 3000 \
  --output_dir "${OUT_DIR}/02_hybrid" >> "${RUN_LOG}" 2>&1

echo "[INFO] Hybrid Training finished." | tee -a "${RUN_LOG}"

# Step 2: E2E Training (5000 epochs, early stop=300)
echo "[INFO] Starting E2E Training (5000 epochs, patience=300)..." | tee -a "${RUN_LOG}"
"${PY}" "${ROOT}/scripts/train_e2e.py" \
  --config "${ROOT}/configs/exp_f2p8_80mm_e2e_fullsafe.yaml" \
  --hybridlens "${OUT_DIR}/02_hybrid/hybrid_final.json" \
  --epochs 5000 \
  --freeze_doe 1 \
  --network_type mha_unet \
  --output_dir "${OUT_DIR}/03_e2e" >> "${RUN_LOG}" 2>&1

echo "[INFO] E2E Training finished." | tee -a "${RUN_LOG}"

# Step 3: Evaluate E2E (PSNR, SSIM)
echo "[INFO] Evaluating E2E Image Metrics" | tee -a "${RUN_LOG}"
"${PY}" "${ROOT}/scripts/evaluate_e2e.py" \
  --config "${ROOT}/configs/exp_f2p8_80mm_e2e_fullsafe.yaml" \
  --checkpoint "${OUT_DIR}/03_e2e/network_epochfinal.pth" \
  --hybridlens "${OUT_DIR}/03_e2e/hybridlens_epochfinal.json" \
  --output_dir "${OUT_DIR}/03_e2e" >> "${RUN_LOG}" 2>&1

echo "[DONE] Workflow completed." | tee -a "${RUN_LOG}"
