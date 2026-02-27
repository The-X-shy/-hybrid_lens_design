#!/usr/bin/env bash

HOST="connect.westb.seetacloud.com"
PORT="14173"
USER="root"
PASS="XQCVslEXeJIG"

REMOTE_DIR="/root/autodl-tmp/hybrid_lens_design/results/geolens8mm_1200"
LOCAL_DIR="/Users/lilin/Desktop/hybrid_lens_design/results/geolens8mm_1200"
BACKUP_DIR="/Users/lilin/Desktop/deeplens backup/results/geolens8mm_1200"

SSH_CMD="sshpass -p '${PASS}' ssh -p ${PORT} -o StrictHostKeyChecking=no"

echo "[$(date)] Watcher started. Waiting for final_lens.json..."

while true; do
  if eval "${SSH_CMD} ${USER}@${HOST} 'test -f ${REMOTE_DIR}/final_lens.json'"; then
    echo "[$(date)] Target found! Waiting 10 seconds for file write to complete..."
    sleep 10
    break
  fi
  sleep 30
done

echo "[$(date)] Initiating download..."

mkdir -p "${LOCAL_DIR}"
mkdir -p "${BACKUP_DIR}"

RSYNC_CMD="sshpass -p '${PASS}' rsync -az --partial --inplace -e \"ssh -p ${PORT} -o StrictHostKeyChecking=no\""

# Primary Download
eval "${RSYNC_CMD} \"${USER}@${HOST}:${REMOTE_DIR}/\" \"${LOCAL_DIR}/\""
echo "[$(date)] Download to primary dir complete."

# Backup Download
eval "${RSYNC_CMD} \"${USER}@${HOST}:${REMOTE_DIR}/\" \"${BACKUP_DIR}/\""
echo "[$(date)] Download to backup dir complete."

echo "DONE." > /Users/lilin/Desktop/hybrid_lens_design/outputs/geolens_download.flag
