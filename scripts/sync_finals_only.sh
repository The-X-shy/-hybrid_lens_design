#!/usr/bin/env bash

HOST="connect.westb.seetacloud.com"
PORT="14173"
USER="root"
PASS="XQCVslEXeJIG"
TS="20260225_010815"
REMOTE_ROOT="/root/autodl-tmp/hybrid_lens_design"
LOCAL_DIR="/Users/lilin/Desktop/hybrid_lens_design/results/f2p8_80mm_${TS}_final_only"
BACKUP_DIR="/Users/lilin/Desktop/deeplens backup/results/f2p8_80mm_${TS}_final_only"

RSYNC_CMD="sshpass -p '${PASS}' rsync -az -e \"ssh -p ${PORT} -o StrictHostKeyChecking=no\""

mkdir -p "${LOCAL_DIR}/01_geolens" "${LOCAL_DIR}/02_hybrid" "${LOCAL_DIR}/03_e2e" "${LOCAL_DIR}/04_meta" "${LOCAL_DIR}/outputs"
mkdir -p "${BACKUP_DIR}/01_geolens" "${BACKUP_DIR}/02_hybrid" "${BACKUP_DIR}/03_e2e" "${BACKUP_DIR}/04_meta" "${BACKUP_DIR}/outputs"

echo "Syncing 01_geolens..."
eval "${RSYNC_CMD} \"${USER}@${HOST}:${REMOTE_ROOT}/results/f2p8_80mm_${TS}/01_geolens/*\" \"${LOCAL_DIR}/01_geolens/\""

echo "Syncing 02_hybrid..."
eval "${RSYNC_CMD} \"${USER}@${HOST}:${REMOTE_ROOT}/results/f2p8_80mm_${TS}/02_hybrid/*\" \"${LOCAL_DIR}/02_hybrid/\""

echo "Syncing 03_e2e final checkpoints & eval results..."
# We only want *final* or non-epoch files
eval "${RSYNC_CMD} --include='*final*' --include='*.png' --include='*.json' --exclude='*epoch[0-9]*.pth' --exclude='*epoch[0-9]*.json' --exclude='*epoch[0-9]*.png' \"${USER}@${HOST}:${REMOTE_ROOT}/results/f2p8_80mm_${TS}/03_e2e/\" \"${LOCAL_DIR}/03_e2e/\""

echo "Syncing 04_meta..."
eval "${RSYNC_CMD} \"${USER}@${HOST}:${REMOTE_ROOT}/results/f2p8_80mm_${TS}/04_meta/*\" \"${LOCAL_DIR}/04_meta/\""

echo "Syncing final logs & summary documents..."
eval "${RSYNC_CMD} \"${USER}@${HOST}:${REMOTE_ROOT}/outputs/f2p8_80mm_run_${TS}_resume.log\" \"${LOCAL_DIR}/outputs/\""
eval "${RSYNC_CMD} \"${USER}@${HOST}:${REMOTE_ROOT}/outputs/training_log_f2p8_80mm_${TS}.md\" \"${LOCAL_DIR}/outputs/\""

echo "Mirroring to Backup folder..."
rsync -az "${LOCAL_DIR}/" "${BACKUP_DIR}/"

echo "DONE."
