#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RESULT_BASE="results/8mm_final_results_summary"
SOURCE_HYBRID="results/full_8mm_20260226_172218/02_hybrid/hybridlens_final.json"
VAL_DATASET="/root/autodl-tmp/DeepLens/datasets/BSDS300/images/test/images"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

run_train_eval() {
  local config="$1"
  local out_dir="$2"
  local eval_dir="${out_dir}_eval"

  log "TRAIN start: config=${config} out=${out_dir}"
  "${PYTHON_BIN}" scripts/train_e2e.py \
    --config "${config}" \
    --hybridlens "${SOURCE_HYBRID}" \
    --output_dir "${out_dir}"

  log "TRAIN done: ${out_dir}"
  if [[ -f "${out_dir}/logs/summary.json" ]]; then
    cat "${out_dir}/logs/summary.json"
  fi

  log "EVAL start: ${eval_dir}"
  "${PYTHON_BIN}" scripts/evaluate_e2e.py \
    --config "${config}" \
    --checkpoint "${out_dir}/network_epochfinal.pth" \
    --hybridlens "${out_dir}/hybridlens_epochfinal.json" \
    --dataset "${VAL_DATASET}" \
    --output_dir "${eval_dir}"

  log "EVAL done: ${eval_dir}"
  if [[ -f "${eval_dir}/eval_metrics.json" ]]; then
    cat "${eval_dir}/eval_metrics.json"
  fi
}

mkdir -p "${RESULT_BASE}"

run_train_eval \
  "configs/exp_f2p8_80mm_e2e_joint_fixed.yaml" \
  "${RESULT_BASE}/11_mha_hybrid_doe_unlock_fixed"

run_train_eval \
  "configs/exp_f2p8_80mm_e2e_joint_fixed_unet.yaml" \
  "${RESULT_BASE}/12_unet_hybrid_doe_unlock_fixed"

run_train_eval \
  "configs/exp_f2p8_80mm_e2e_refractive_tune_fixed.yaml" \
  "${RESULT_BASE}/13_mha_hybrid_refractive_tune_fixed"

run_train_eval \
  "configs/exp_f2p8_80mm_e2e_refractive_tune_fixed_unet.yaml" \
  "${RESULT_BASE}/14_unet_hybrid_refractive_tune_fixed"

log "QUEUE complete"
