#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/hybrid_lens_design"
TS="20260225_010815"
RUN_LOG="${ROOT}/outputs/f2p8_80mm_run_${TS}_resume.log"
FULL_ROOT="${ROOT}/results/f2p8_80mm_${TS}"
SMOKE_ROOT="${ROOT}/results/f2p8_80mm_smoke_${TS}"
PY="/root/miniconda3/bin/python"

echo "[INFO] Full Training: Metasurface" | tee -a "${RUN_LOG}"
"${PY}" "${ROOT}/scripts/train_metalens.py" \
  --config "${ROOT}/configs/exp_f2p8_80mm_meta.yaml" \
  --geolens "${FULL_ROOT}/01_geolens/final_lens.json" \
  --iterations 1000 \
  --output_dir "${FULL_ROOT}/04_meta" >> "${RUN_LOG}" 2>&1

echo "[INFO] Generate markdown log document..." | tee -a "${RUN_LOG}"
"${PY}" "${ROOT}/scripts/generate_training_log.py" \
  --experiment_name "f2p8_80mm_${TS}" \
  --base_dir "${FULL_ROOT}" \
  --smoke_dir "${SMOKE_ROOT}" \
  --output "${ROOT}/outputs/training_log_f2p8_80mm_${TS}.md" >> "${RUN_LOG}" 2>&1

echo "[INFO] Generate checksum manifest..." | tee -a "${RUN_LOG}"
(
  cd "${FULL_ROOT}"
  find . -type f -print0 | sort -z | xargs -0 sha256sum > "${ROOT}/outputs/sha256_manifest_f2p8_80mm_${TS}.txt"
) >> "${RUN_LOG}" 2>&1

echo "[DONE] Workflow completed." | tee -a "${RUN_LOG}"
