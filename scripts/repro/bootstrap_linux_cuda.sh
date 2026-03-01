#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_NAME="${ENV_NAME:-hybrid_lens_repro}"
LOCK_FILE="${LOCK_FILE:-${ROOT_DIR}/repro/environment/conda-linux-cuda12.1-explicit.txt}"
PIP_LOCK_FILE="${ROOT_DIR}/repro/environment/pip-freeze.txt"

cd "${ROOT_DIR}"

echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] ENV_NAME=${ENV_NAME}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Install conda/miniconda first."
  exit 1
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "[WARN] bootstrap_linux_cuda.sh is designed for Linux+CUDA. Current OS: $(uname -s)"
fi

USE_EXPLICIT=0
if [[ -f "${LOCK_FILE}" ]] && grep -q "^@EXPLICIT" "${LOCK_FILE}"; then
  PLATFORM_LINE="$(grep '^# platform:' "${LOCK_FILE}" || true)"
  if [[ "${PLATFORM_LINE}" == *"linux"* ]]; then
    USE_EXPLICIT=1
  else
    echo "[WARN] Lock file platform mismatch: ${PLATFORM_LINE}"
  fi
else
  echo "[WARN] Explicit lock not found or placeholder: ${LOCK_FILE}"
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Removing existing env: ${ENV_NAME}"
  conda env remove -y -n "${ENV_NAME}" >/dev/null
fi

if [[ "${USE_EXPLICIT}" -eq 1 ]]; then
  echo "[INFO] Creating env from explicit lock: ${LOCK_FILE}"
  conda create -y -n "${ENV_NAME}" --file "${LOCK_FILE}"
else
  echo "[INFO] Creating env from environment.yml"
  conda env create -y -n "${ENV_NAME}" -f environment.yml
fi

echo "[INFO] Installing vendored deeplens-core"
conda run -n "${ENV_NAME}" python -m pip install -e ./third_party/deeplens-core

echo "[INFO] Installing hybrid_lens_design"
conda run -n "${ENV_NAME}" python -m pip install -e .

echo "[INFO] Verifying deeplens lock"
conda run -n "${ENV_NAME}" python scripts/repro/check_deeplens_lock.py --strict-import

echo "[INFO] Capturing env fingerprint"
conda run -n "${ENV_NAME}" python scripts/repro/capture_env_fingerprint.py --pip-freeze-out "${PIP_LOCK_FILE}"

echo "[INFO] Exporting conda explicit lock"
conda list -n "${ENV_NAME}" --explicit > "${LOCK_FILE}"

echo "[DONE] Environment bootstrap completed"
echo "[DONE] conda explicit: ${LOCK_FILE}"
echo "[DONE] pip freeze: ${PIP_LOCK_FILE}"
