#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

SMOKE_ROOT="${ROOT_DIR}/results/f2p8_80mm_smoke_${TS}"
FULL_ROOT="${ROOT_DIR}/results/f2p8_80mm_${TS}"
LOG_ROOT="${ROOT_DIR}/outputs"
RUN_LOG="${LOG_ROOT}/f2p8_80mm_run_${TS}.log"

mkdir -p "${SMOKE_ROOT}" "${FULL_ROOT}" "${LOG_ROOT}"
echo "[INFO] ROOT_DIR=${ROOT_DIR}" | tee -a "${RUN_LOG}"
echo "[INFO] TS=${TS}" | tee -a "${RUN_LOG}"
echo "[INFO] SMOKE_ROOT=${SMOKE_ROOT}" | tee -a "${RUN_LOG}"
echo "[INFO] FULL_ROOT=${FULL_ROOT}" | tee -a "${RUN_LOG}"

cd "${ROOT_DIR}"

echo "[INFO] Checking environment imports..." | tee -a "${RUN_LOG}"
"${PYTHON_BIN}" -c "import torch, torchvision, lpips, yaml, tqdm; print('deps_ok')" | tee -a "${RUN_LOG}"

echo "[INFO] Installing editable packages..." | tee -a "${RUN_LOG}"
"${PYTHON_BIN}" -m pip install -e "${ROOT_DIR}/third_party/deeplens-core" >> "${RUN_LOG}" 2>&1
"${PYTHON_BIN}" -m pip install -e "${ROOT_DIR}" >> "${RUN_LOG}" 2>&1

echo "[INFO] Ensuring BSDS300 dataset..." | tee -a "${RUN_LOG}"
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
from deeplens.network.dataset import download_bsd300
target = Path('/root/autodl-tmp/DeepLens/datasets/BSDS300/images/train/images')
if target.exists():
    print('BSDS300 exists:', target)
else:
    print('Downloading BSDS300...')
    out = download_bsd300('/root/autodl-tmp/DeepLens/datasets')
    print('Downloaded:', out)
PY

"${PYTHON_BIN}" - <<'PY' >> "${RUN_LOG}" 2>&1
from pathlib import Path
import shutil
src_train=Path('/root/autodl-tmp/DeepLens/datasets/BSDS300/images/train/images')
src_test=Path('/root/autodl-tmp/DeepLens/datasets/BSDS300/images/test/images')
dst_train=Path('/root/autodl-tmp/DeepLens/datasets/BSDS300_smoke/images/train/images')
dst_test=Path('/root/autodl-tmp/DeepLens/datasets/BSDS300_smoke/images/test/images')
dst_train.mkdir(parents=True, exist_ok=True)
dst_test.mkdir(parents=True, exist_ok=True)
for p in sorted(src_train.glob('*'))[:20]:
    q=dst_train/p.name
    if not q.exists():
        shutil.copy2(p,q)
for p in sorted(src_test.glob('*'))[:10]:
    q=dst_test/p.name
    if not q.exists():
        shutil.copy2(p,q)
print('BSDS300_smoke ready')
PY

echo "[INFO] Smoke Test: GeoLens" | tee -a "${RUN_LOG}"
"${PYTHON_BIN}" scripts/train_geolens.py --config configs/exp_f2p8_80mm_geolens.yaml --iterations 100 --finetune_iterations 100 --output_dir "${SMOKE_ROOT}/01_geolens" >> "${RUN_LOG}" 2>&1

echo "[INFO] Smoke Test: Hybrid" | tee -a "${RUN_LOG}"
"${PYTHON_BIN}" scripts/train_hybridlens.py --config configs/exp_f2p8_80mm_hybrid.yaml --geolens "${SMOKE_ROOT}/01_geolens/final_lens.json" --iterations 100 --wavelength 0.55 --output_dir "${SMOKE_ROOT}/02_hybrid" >> "${RUN_LOG}" 2>&1

echo "[INFO] Smoke Test: E2E" | tee -a "${RUN_LOG}"
"${PYTHON_BIN}" scripts/train_e2e.py --config configs/exp_f2p8_80mm_e2e_smoke.yaml --hybridlens "${SMOKE_ROOT}/02_hybrid/hybrid_final.json" --epochs 100 --freeze_doe 1 --network_type mha_unet --output_dir "${SMOKE_ROOT}/03_e2e" >> "${RUN_LOG}" 2>&1

echo "[INFO] Smoke Test: Metasurface" | tee -a "${RUN_LOG}"
for retry in 1 2 3; do
  if "${PYTHON_BIN}" scripts/train_metalens.py --config configs/exp_f2p8_80mm_meta.yaml --geolens "${SMOKE_ROOT}/01_geolens/final_lens.json" --iterations 100 --output_dir "${SMOKE_ROOT}/04_meta" >> "${RUN_LOG}" 2>&1; then
    break
  fi
  echo "[WARN] Metasurface smoke failed on attempt ${retry}, retrying..." | tee -a "${RUN_LOG}"
  sleep 20
  if [ "$retry" = "3" ]; then
    echo "[ERROR] Metasurface smoke failed after 3 attempts" | tee -a "${RUN_LOG}"
    exit 1
  fi
done

echo "[INFO] Full Training: GeoLens" | tee -a "${RUN_LOG}"
"${PYTHON_BIN}" scripts/train_geolens.py --config configs/exp_f2p8_80mm_geolens.yaml --iterations 3000 --finetune_iterations 3000 --output_dir "${FULL_ROOT}/01_geolens" >> "${RUN_LOG}" 2>&1

echo "[INFO] Full Training: Hybrid" | tee -a "${RUN_LOG}"
"${PYTHON_BIN}" scripts/train_hybridlens.py --config configs/exp_f2p8_80mm_hybrid.yaml --geolens "${FULL_ROOT}/01_geolens/final_lens.json" --iterations 5000 --wavelength 0.55 --output_dir "${FULL_ROOT}/02_hybrid" >> "${RUN_LOG}" 2>&1

echo "[INFO] Full Training: E2E" | tee -a "${RUN_LOG}"
"${PYTHON_BIN}" scripts/train_e2e.py --config configs/exp_f2p8_80mm_e2e_fullsafe.yaml --hybridlens "${FULL_ROOT}/02_hybrid/hybrid_final.json" --epochs 5000 --freeze_doe 1 --network_type mha_unet --output_dir "${FULL_ROOT}/03_e2e" >> "${RUN_LOG}" 2>&1

echo "[INFO] Full Training: Metasurface" | tee -a "${RUN_LOG}"
for retry in 1 2; do
  if "${PYTHON_BIN}" scripts/train_metalens.py --config configs/exp_f2p8_80mm_meta.yaml --geolens "${FULL_ROOT}/01_geolens/final_lens.json" --iterations 4000 --output_dir "${FULL_ROOT}/04_meta" >> "${RUN_LOG}" 2>&1; then
    break
  fi
  echo "[WARN] Metasurface full failed on attempt ${retry}, retrying..." | tee -a "${RUN_LOG}"
  sleep 30
  if [ "$retry" = "2" ]; then
    echo "[ERROR] Metasurface full failed after 2 attempts" | tee -a "${RUN_LOG}"
    exit 1
  fi
done

echo "[INFO] Generate markdown log document..." | tee -a "${RUN_LOG}"
"${PYTHON_BIN}" scripts/generate_training_log.py --experiment_name "f2p8_80mm_${TS}" --base_dir "${FULL_ROOT}" --smoke_dir "${SMOKE_ROOT}" --output "${LOG_ROOT}/training_log_f2p8_80mm_${TS}.md" >> "${RUN_LOG}" 2>&1

echo "[INFO] Generate checksum manifest..." | tee -a "${RUN_LOG}"
(
  cd "${FULL_ROOT}"
  find . -type f -print0 | sort -z | xargs -0 sha256sum > "${LOG_ROOT}/sha256_manifest_f2p8_80mm_${TS}.txt"
) >> "${RUN_LOG}" 2>&1

echo "[DONE] Workflow completed." | tee -a "${RUN_LOG}"
echo "[DONE] Smoke root: ${SMOKE_ROOT}" | tee -a "${RUN_LOG}"
echo "[DONE] Full root:  ${FULL_ROOT}" | tee -a "${RUN_LOG}"
echo "[DONE] Doc:        ${LOG_ROOT}/training_log_f2p8_80mm_${TS}.md" | tee -a "${RUN_LOG}"
echo "[DONE] Run log:    ${RUN_LOG}" | tee -a "${RUN_LOG}"
