#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/repro/runs/smoke_${TS}}"
REPORT_PATH="${REPORT_PATH:-${ROOT_DIR}/repro/reports/repro_report_${TS}.md}"
BASELINE="${BASELINE:-${ROOT_DIR}/repro/baselines/smoke_metrics_baseline.json}"
TOLERANCE="${TOLERANCE:-${ROOT_DIR}/repro/baselines/tolerance.yaml}"

# Dataset roots (Linux+CUDA server defaults)
BSDS300_ROOT="${BSDS300_ROOT:-/root/autodl-tmp/DeepLens/datasets/BSDS300/images}"
BSDS300_SMOKE_ROOT="${BSDS300_SMOKE_ROOT:-/root/autodl-tmp/DeepLens/datasets/BSDS300_smoke/images}"

MANIFEST_BSDS300="${ROOT_DIR}/repro/datasets/bsds300_manifest.sha256"
MANIFEST_SMOKE="${ROOT_DIR}/repro/datasets/bsds300_smoke_manifest.sha256"
ALLOW_MANIFEST_BOOTSTRAP="${ALLOW_MANIFEST_BOOTSTRAP:-0}"

mkdir -p "${RUN_ROOT}" "$(dirname "${REPORT_PATH}")"
cd "${ROOT_DIR}"

echo "[INFO] Checking vendored DeepLens lock..."
"${PYTHON_BIN}" scripts/repro/check_deeplens_lock.py --strict-import

echo "[INFO] Capturing environment fingerprint..."
"${PYTHON_BIN}" scripts/repro/capture_env_fingerprint.py

echo "[INFO] Verifying dataset manifest: BSDS300"
if [[ "${ALLOW_MANIFEST_BOOTSTRAP}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/repro/check_dataset_manifest.py \
    --dataset-root "${BSDS300_ROOT}" \
    --manifest "${MANIFEST_BSDS300}" \
    --create-if-missing
else
  "${PYTHON_BIN}" scripts/repro/check_dataset_manifest.py \
    --dataset-root "${BSDS300_ROOT}" \
    --manifest "${MANIFEST_BSDS300}"
fi

echo "[INFO] Verifying dataset manifest: BSDS300_smoke"
if [[ "${ALLOW_MANIFEST_BOOTSTRAP}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/repro/check_dataset_manifest.py \
    --dataset-root "${BSDS300_SMOKE_ROOT}" \
    --manifest "${MANIFEST_SMOKE}" \
    --create-if-missing
else
  "${PYTHON_BIN}" scripts/repro/check_dataset_manifest.py \
    --dataset-root "${BSDS300_SMOKE_ROOT}" \
    --manifest "${MANIFEST_SMOKE}"
fi

echo "[INFO] Running GeoLens smoke"
"${PYTHON_BIN}" scripts/train_geolens.py \
  --config configs/exp_f2p8_80mm_geolens.yaml \
  --iterations 100 \
  --finetune_iterations 100 \
  --output_dir "${RUN_ROOT}/01_geolens"

echo "[INFO] Running Hybrid smoke"
"${PYTHON_BIN}" scripts/train_hybridlens.py \
  --config configs/exp_f2p8_80mm_hybrid.yaml \
  --geolens "${RUN_ROOT}/01_geolens/final_lens.json" \
  --iterations 100 \
  --wavelength 0.55 \
  --output_dir "${RUN_ROOT}/02_hybrid"

echo "[INFO] Running E2E smoke"
"${PYTHON_BIN}" scripts/train_e2e.py \
  --config configs/exp_f2p8_80mm_e2e_smoke.yaml \
  --hybridlens "${RUN_ROOT}/02_hybrid/hybrid_final.json" \
  --epochs 100 \
  --freeze_doe 1 \
  --network_type mha_unet \
  --output_dir "${RUN_ROOT}/03_e2e"

echo "[INFO] Running Metasurface smoke"
"${PYTHON_BIN}" scripts/train_metalens.py \
  --config configs/exp_f2p8_80mm_meta.yaml \
  --geolens "${RUN_ROOT}/01_geolens/final_lens.json" \
  --iterations 100 \
  --output_dir "${RUN_ROOT}/04_meta"

echo "[INFO] Verifying reproducibility metrics"
"${PYTHON_BIN}" scripts/repro/verify_repro_metrics.py \
  --baseline "${BASELINE}" \
  --current "${RUN_ROOT}" \
  --tolerance "${TOLERANCE}" \
  --report "${REPORT_PATH}"

echo "[DONE] Smoke reproducibility run complete"
echo "[DONE] RUN_ROOT=${RUN_ROOT}"
echo "[DONE] REPORT_PATH=${REPORT_PATH}"
