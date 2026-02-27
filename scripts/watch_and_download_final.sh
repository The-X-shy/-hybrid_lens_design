#!/usr/bin/env bash

HOST="connect.westb.seetacloud.com"
PORT="14173"
USER="root"
PASS="XQCVslEXeJIG"
TS="20260226_172218"
REMOTE_ROOT="/root/autodl-tmp/hybrid_lens_design"
LOCAL_DIR="/Users/lilin/Desktop/hybrid_lens_design/results/full_8mm_${TS}"
BACKUP_DIR="/Users/lilin/Desktop/deeplens backup/results/full_8mm_${TS}"

SSH_CMD="sshpass -p '${PASS}' ssh -p ${PORT} -o StrictHostKeyChecking=no"

echo "[$(date)] Final Watcher started. Waiting for ALL_DONE.flag..."

while true; do
  if eval "${SSH_CMD} ${USER}@${HOST} 'test -f /root/autodl-tmp/ALL_DONE.flag'"; then
    echo "[$(date)] Target found! Initiating precise final download..."
    break
  fi
  sleep 60
done

mkdir -p "${LOCAL_DIR}/03_e2e" "${LOCAL_DIR}/03_e2e_control" "${LOCAL_DIR}/outputs"
mkdir -p "${BACKUP_DIR}/03_e2e" "${BACKUP_DIR}/03_e2e_control" "${BACKUP_DIR}/outputs"

RSYNC_CMD="sshpass -p '${PASS}' rsync -az --partial --inplace -e \"ssh -p ${PORT} -o StrictHostKeyChecking=no\""

# 1. Download logs
eval "${RSYNC_CMD} \"${USER}@${HOST}:${REMOTE_ROOT}/results/full_8mm_${TS}/run.log\" \"${LOCAL_DIR}/outputs/\""

# 2. Download Main E2E finals only (no epoch pths/pngs)
eval "${RSYNC_CMD} --include='*final*' --include='*.png' --include='*.json' --exclude='*epoch[0-9]*.pth' --exclude='*epoch[0-9]*.json' --exclude='*epoch[0-9]*.png' \"${USER}@${HOST}:${REMOTE_ROOT}/results/full_8mm_${TS}/03_e2e/\" \"${LOCAL_DIR}/03_e2e/\""

# 3. Download Control E2E finals only
eval "${RSYNC_CMD} --include='*final*' --include='*.png' --include='*.json' --exclude='*epoch[0-9]*.pth' --exclude='*epoch[0-9]*.json' --exclude='*epoch[0-9]*.png' \"${USER}@${HOST}:${REMOTE_ROOT}/results/full_8mm_${TS}/03_e2e_control/\" \"${LOCAL_DIR}/03_e2e_control/\""

# Mirror to Backup
rsync -az "${LOCAL_DIR}/" "${BACKUP_DIR}/"

echo "[$(date)] Download and mirror complete. Putting Mac to sleep."
pmset displaysleepnow

echo "ALL DONE." > /Users/lilin/Desktop/hybrid_lens_design/outputs/final_sync.flag
