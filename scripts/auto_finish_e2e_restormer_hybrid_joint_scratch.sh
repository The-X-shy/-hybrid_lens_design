#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-connect.cqa1.seetacloud.com}"
PORT="${PORT:-17580}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_PASS="${REMOTE_PASS:-zKWs7O9ujRsH}"
LOCAL_PROJECT="${LOCAL_PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/hybrid_lens_design}"
LOCAL_RESULTS="${LOCAL_RESULTS:-${LOCAL_PROJECT}/results}"
LOCAL_BACKUP="${LOCAL_BACKUP:-/Users/lilin/Desktop/deeplens backup/results}"
RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOCAL_PROJECT}/outputs"
LOG_FILE="${LOG_DIR}/auto_finish_e2e_restormer_hybrid_joint_scratch_${RUN_ID}.log"

mkdir -p "${LOG_DIR}" "${LOCAL_RESULTS}"
touch "${LOG_FILE}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG_FILE}"
}

remote_exec() {
  sshpass -p "${REMOTE_PASS}" ssh -p "${PORT}" -o ConnectTimeout=15 -o StrictHostKeyChecking=no \
    "${REMOTE_USER}@${HOST}" "$1"
}

sync_dir_to_remote() {
  local src="$1"
  local dst="$2"
  sshpass -p "${REMOTE_PASS}" rsync -az \
    --exclude '.DS_Store' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '*.pyo' \
    -e "ssh -p ${PORT} -o StrictHostKeyChecking=no" \
    "${LOCAL_PROJECT}/${src}/" "${REMOTE_USER}@${HOST}:${REMOTE_ROOT}/${dst}/"
}

sync_file_to_remote() {
  local src="$1"
  local dst="$2"
  sshpass -p "${REMOTE_PASS}" scp -P "${PORT}" -o StrictHostKeyChecking=no \
    "${LOCAL_PROJECT}/${src}" "${REMOTE_USER}@${HOST}:${REMOTE_ROOT}/${dst}"
}

launch_phase() {
  local mode="$1"
  local phase_root="${REMOTE_ROOT}/results/e2e_restormer_hybrid_scratch_${RUN_ID}/${mode}"
  local launcher_log="${REMOTE_ROOT}/outputs/e2e_restormer_hybrid_scratch_${RUN_ID}_${mode}.launcher.log"
  local pid_file="${REMOTE_ROOT}/outputs/e2e_restormer_hybrid_scratch_${RUN_ID}_${mode}.pid"
  log "Launching remote ${mode} phase"
  remote_exec "mkdir -p '${REMOTE_ROOT}/outputs' '${phase_root}' && nohup bash '${REMOTE_ROOT}/scripts/run_e2e_restormer_hybrid_joint_scratch.sh' '${mode}' '${RUN_ID}' > '${launcher_log}' 2>&1 & echo \$! > '${pid_file}'"
}

wait_for_phase() {
  local mode="$1"
  local phase_root="${REMOTE_ROOT}/results/e2e_restormer_hybrid_scratch_${RUN_ID}/${mode}"
  local complete_flag="${phase_root}/COMPLETE.flag"
  local pid_file="${REMOTE_ROOT}/outputs/e2e_restormer_hybrid_scratch_${RUN_ID}_${mode}.pid"

  log "Waiting for remote ${mode} phase to complete"
  while true; do
    if remote_exec "test -f '${complete_flag}'" >/dev/null 2>&1; then
      log "Remote ${mode} phase completed"
      return 0
    fi

    if remote_exec "test -f '${pid_file}'" >/dev/null 2>&1; then
      local pid
      pid="$(remote_exec "cat '${pid_file}'" 2>/dev/null || true)"
      if [[ -n "${pid}" ]]; then
        if remote_exec "kill -0 '${pid}'" >/dev/null 2>&1; then
          sleep 60
          continue
        fi

        log "Remote ${mode} phase stopped without completion flag"
        remote_exec "tail -n 60 '${phase_root}/runner.log' 2>/dev/null || true" || true
        return 1
      fi
    fi

    sleep 60
  done
}

sync_results_back() {
  local remote_dir="${REMOTE_ROOT}/results/e2e_restormer_hybrid_scratch_${RUN_ID}"
  local local_dir="${LOCAL_RESULTS}/e2e_restormer_hybrid_scratch_${RUN_ID}"

  log "Syncing results back to local workspace"
  mkdir -p "${local_dir}"
  sshpass -p "${REMOTE_PASS}" rsync -az --partial --inplace \
    -e "ssh -p ${PORT} -o StrictHostKeyChecking=no" \
    "${REMOTE_USER}@${HOST}:${remote_dir}/" "${local_dir}/"

  if [[ -n "${LOCAL_BACKUP}" ]]; then
    mkdir -p "${LOCAL_BACKUP}/e2e_restormer_hybrid_scratch_${RUN_ID}"
    sshpass -p "${REMOTE_PASS}" rsync -az --partial --inplace \
      -e "ssh -p ${PORT} -o StrictHostKeyChecking=no" \
      "${REMOTE_USER}@${HOST}:${remote_dir}/" \
      "${LOCAL_BACKUP}/e2e_restormer_hybrid_scratch_${RUN_ID}/"
  fi
}

log "Starting hybrid Restormer scratch automation for run_id=${RUN_ID}"

log "Syncing code to remote"
remote_exec "mkdir -p '${REMOTE_ROOT}' '${REMOTE_ROOT}/outputs' '${REMOTE_ROOT}/results' '${REMOTE_ROOT}/configs' '${REMOTE_ROOT}/scripts' '${REMOTE_ROOT}/src' '${REMOTE_ROOT}/third_party/deeplens-core' '${REMOTE_ROOT}/results/geolens8mm_1200_final/02_hybrid'"
sync_dir_to_remote "configs" "configs"
sync_dir_to_remote "scripts" "scripts"
sync_dir_to_remote "src" "src"
sync_file_to_remote "pyproject.toml" "pyproject.toml"
sync_file_to_remote "setup.py" "setup.py"
sync_file_to_remote "environment.yml" "environment.yml"
sync_file_to_remote "README.md" "README.md"
sync_dir_to_remote "third_party/deeplens-core/deeplens" "third_party/deeplens-core/deeplens"
sync_file_to_remote "third_party/deeplens-core/pyproject.toml" "third_party/deeplens-core/pyproject.toml"
sync_file_to_remote "third_party/deeplens-core/requirements.txt" "third_party/deeplens-core/requirements.txt"
sync_file_to_remote "third_party/deeplens-core/LICENSE" "third_party/deeplens-core/LICENSE"
sync_file_to_remote "results/geolens8mm_1200_final/02_hybrid/hybridlens_final.json" "results/geolens8mm_1200_final/02_hybrid/hybridlens_final.json"

log "Installing editable packages on remote"
remote_exec "cd '${REMOTE_ROOT}' && /root/miniconda3/bin/python -m pip install -e ./third_party/deeplens-core && /root/miniconda3/bin/python -m pip install -e ."

launch_phase "smoke"
if ! wait_for_phase "smoke"; then
  log "Smoke phase failed"
  sync_results_back
  remote_exec "shutdown -h now" || true
  exit 1
fi

launch_phase "full"
if ! wait_for_phase "full"; then
  log "Full phase failed"
  sync_results_back
  remote_exec "shutdown -h now" || true
  exit 1
fi

sync_results_back

log "Shutting down remote server"
remote_exec "shutdown -h now" || true

log "All phases complete"
