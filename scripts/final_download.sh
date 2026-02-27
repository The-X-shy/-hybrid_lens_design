#!/usr/bin/env bash

HOST="connect.westb.seetacloud.com"
PORT="14173"
USER="root"
PASS="XQCVslEXeJIG"
TS="20260225_010815"
REMOTE_ROOT="/root/autodl-tmp/hybrid_lens_design"
LOCAL_HYBRID_ROOT="/Users/lilin/Desktop/hybrid_lens_design"
LOCAL_BACKUP_ROOT="/Users/lilin/Desktop/deeplens backup"

RSYNC_CMD="sshpass -p '${PASS}' rsync -az --partial --inplace -e \"ssh -p ${PORT} -o StrictHostKeyChecking=no\""

echo "[$(date)] Syncing FULL dir to primary..."
while ! eval "${RSYNC_CMD} \"${USER}@${HOST}:${REMOTE_ROOT}/results/f2p8_80mm_${TS}/\" \"${LOCAL_HYBRID_ROOT}/results/f2p8_80mm_${TS}/\""; do
    echo "Retry primary..."
    sleep 5
done

echo "[$(date)] Syncing FULL dir to backup..."
while ! eval "${RSYNC_CMD} \"${USER}@${HOST}:${REMOTE_ROOT}/results/f2p8_80mm_${TS}/\" \"${LOCAL_BACKUP_ROOT}/results/f2p8_80mm_${TS}/\""; do
    echo "Retry backup..."
    sleep 5
done

echo "[$(date)] Syncing logs..."
eval "${RSYNC_CMD} \"${USER}@${HOST}:${REMOTE_ROOT}/outputs/f2p8_80mm_run_${TS}_resume.log\" \"${LOCAL_HYBRID_ROOT}/outputs/\""
eval "${RSYNC_CMD} \"${USER}@${HOST}:${REMOTE_ROOT}/outputs/training_log_f2p8_80mm_${TS}.md\" \"${LOCAL_HYBRID_ROOT}/outputs/\""
eval "${RSYNC_CMD} \"${USER}@${HOST}:${REMOTE_ROOT}/outputs/sha256_manifest_f2p8_80mm_${TS}.txt\" \"${LOCAL_HYBRID_ROOT}/outputs/\""

echo "ALL SYNC DONE" > /Users/lilin/Desktop/hybrid_lens_design/outputs/download_done.flag
