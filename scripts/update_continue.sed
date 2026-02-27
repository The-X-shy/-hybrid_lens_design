/echo "\[INFO\] Full Training: Metasurface"/i \
echo "[INFO] Evaluating E2E Image Metrics (PSNR, SSIM, LPIPS)" | tee -a "${RUN_LOG}"\
"${PY}" "${ROOT}/scripts/evaluate_e2e.py" \\\
  --config "${ROOT}/configs/exp_f2p8_80mm_e2e_fullsafe.yaml" \\\
  --checkpoint "${FULL_ROOT}/03_e2e/network_epochfinal.pth" \\\
  --hybridlens "${FULL_ROOT}/02_hybrid/hybrid_final.json" \\\
  --output_dir "${FULL_ROOT}/03_e2e" >> "${RUN_LOG}" 2>\&1\
\

