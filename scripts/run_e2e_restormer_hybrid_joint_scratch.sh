#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MODE="${1:-full}"
RUN_ID="${2:-$(date +%Y%m%d_%H%M%S)}"

CONFIG="${PROJECT_ROOT}/configs/exp_f2p8_80mm_e2e_hybrid_joint_scratch_restormer.yaml"
INIT_LENS="${PROJECT_ROOT}/results/geolens8mm_1200_final/02_hybrid/hybridlens_final.json"
RESULT_ROOT="${PROJECT_ROOT}/results/e2e_restormer_hybrid_scratch_${RUN_ID}"
PHASE_ROOT="${RESULT_ROOT}/${MODE}"
TRAIN_DIR="${PHASE_ROOT}/train"
EVAL_DIR="${PHASE_ROOT}/eval"
RUN_LOG="${PHASE_ROOT}/runner.log"
COMPLETE_FLAG="${PHASE_ROOT}/COMPLETE.flag"

mkdir -p "${TRAIN_DIR}" "${EVAL_DIR}"
touch "${RUN_LOG}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[error] missing config: ${CONFIG}" | tee -a "${RUN_LOG}"
  exit 1
fi
if [[ ! -f "${INIT_LENS}" ]]; then
  echo "[error] missing hybrid init lens: ${INIT_LENS}" | tee -a "${RUN_LOG}"
  exit 1
fi

case "${MODE}" in
  smoke)
    EPOCHS=100
    SAVE_EVERY=50
    ;;
  full)
    EPOCHS=5000
    SAVE_EVERY=500
    ;;
  *)
    echo "Usage: bash scripts/run_e2e_restormer_hybrid_joint_scratch.sh [smoke|full] [run_id]" | tee -a "${RUN_LOG}"
    exit 1
    ;;
esac

BATCH_CANDIDATES=(2 1)

echo "[run] mode=${MODE} run_id=${RUN_ID}" | tee -a "${RUN_LOG}"
echo "[run] config=${CONFIG}" | tee -a "${RUN_LOG}"
echo "[run] init_lens=${INIT_LENS}" | tee -a "${RUN_LOG}"
echo "[run] result_root=${RESULT_ROOT}" | tee -a "${RUN_LOG}"

TRAIN_OK=0
for BATCH in "${BATCH_CANDIDATES[@]}"; do
  echo "[train] batch_size=${BATCH}" | tee -a "${RUN_LOG}"
  if "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_e2e.py" \
    --config "${CONFIG}" \
    --hybridlens "${INIT_LENS}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH}" \
    --save_network_every "${SAVE_EVERY}" \
    --network_type restormer \
    --freeze_doe 0 \
    --freeze_geolens 0 \
    --freeze_network 0 \
    --optics_grad_batches 1 \
    --output_dir "${TRAIN_DIR}" \
    >>"${RUN_LOG}" 2>&1; then
    TRAIN_OK=1
    break
  fi

  if grep -Eqi "out of memory|cuda error|cudnn|allocation" "${RUN_LOG}"; then
    echo "[train] batch_size=${BATCH} failed with memory error, retrying" | tee -a "${RUN_LOG}"
    continue
  fi

  echo "[train] failed without recoverable OOM" | tee -a "${RUN_LOG}"
  exit 1
done

if [[ "${TRAIN_OK}" -ne 1 ]]; then
  echo "[train] exhausted batch-size fallbacks" | tee -a "${RUN_LOG}"
  exit 1
fi

CHECKPOINT="${TRAIN_DIR}/network_epochfinal.pth"
EVAL_LENS="${TRAIN_DIR}/hybridlens_epochfinal.json"
if [[ ! -f "${EVAL_LENS}" ]]; then
  EVAL_LENS="${TRAIN_DIR}/hybridlens_input.json"
fi
if [[ ! -f "${EVAL_LENS}" ]]; then
  EVAL_LENS="${INIT_LENS}"
fi

echo "[eval] checkpoint=${CHECKPOINT}" | tee -a "${RUN_LOG}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_e2e.py" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --hybridlens "${EVAL_LENS}" \
  --output_dir "${EVAL_DIR}" \
  >>"${RUN_LOG}" 2>&1

if [[ -f "${EVAL_DIR}/eval_metrics.json" ]]; then
  echo "[eval] metrics:" | tee -a "${RUN_LOG}"
  cat "${EVAL_DIR}/eval_metrics.json" | tee -a "${RUN_LOG}"
fi

touch "${COMPLETE_FLAG}"
echo "[done] mode=${MODE} run_id=${RUN_ID}" | tee -a "${RUN_LOG}"
