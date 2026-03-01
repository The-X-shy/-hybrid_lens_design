# Environment Lock Files

- `conda-linux-cuda12.1-explicit.txt`:
  - authoritative lock file for Linux+CUDA baseline
  - if it is a placeholder, run `scripts/repro/bootstrap_linux_cuda.sh` on Linux+CUDA to regenerate
- `conda-local-osx-arm64-explicit.txt`:
  - optional local development snapshot (macOS), not used as Linux reproducibility baseline
- `pip-freeze.txt`:
  - pip package snapshot from the active reproduction environment

Note:
If this repository is prepared on non-Linux platforms, re-run bootstrap on Linux+CUDA and commit updated lock files.
