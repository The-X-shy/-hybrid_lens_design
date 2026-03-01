# DeepLens 固定与环境复现进度（2026-03-02）

## 当前状态
- 状态：已完成本地代码与文档落地，待 Linux+CUDA 机器执行完整 bootstrap + smoke 验收。
- DeepLens 固定：已完成（vendoring + 哈希锁）。
- 复现脚本：已完成（环境、数据、指标对比、报告）。
- 训练算法接口：未改动。

## 已完成项
1. DeepLens 固定
- 快照目录：`third_party/deeplens-core`
- 锁定文件：`repro/deeplens_lock.json`
- 哈希清单：`repro/deeplens_sha256_manifest.txt`

2. 环境锁定
- Conda Linux 锁文件路径已预留：`repro/environment/conda-linux-cuda12.1-explicit.txt`
- 本地开发快照：`repro/environment/conda-local-osx-arm64-explicit.txt`
- pip 锁：`repro/environment/pip-freeze.txt`

3. 复现脚本
- `scripts/repro/bootstrap_linux_cuda.sh`
- `scripts/repro/check_deeplens_lock.py`
- `scripts/repro/check_dataset_manifest.py`
- `scripts/repro/run_smoke_repro.sh`
- `scripts/repro/verify_repro_metrics.py`
- `scripts/repro/capture_env_fingerprint.py`

4. 基线与容差
- 指标基线：`repro/baselines/smoke_metrics_baseline.json`
- 容差配置：`repro/baselines/tolerance.yaml`

5. 文档
- 复现手册：`文档/过程记录/ENV_REPRO_GUIDE.md`
- README 已增加 Reproducibility 章节。

## 本地验证结果
- `python scripts/repro/check_deeplens_lock.py`：PASS
- `python scripts/repro/capture_env_fingerprint.py`：PASS
- `python scripts/repro/verify_repro_metrics.py --baseline repro/baselines/smoke_metrics_baseline.json --current results/f2p8_80mm_20260225_010815_final_only --tolerance repro/baselines/tolerance.yaml --report repro/reports/repro_report_test.md`：PASS
- 语法检查：`bash -n` 与 `python -m py_compile` 均通过。

## 待执行（Linux+CUDA）
1. `bash scripts/repro/bootstrap_linux_cuda.sh`
2. 首次生成数据清单（仅一次）：`ALLOW_MANIFEST_BOOTSTRAP=1 bash scripts/repro/run_smoke_repro.sh`
3. 固化后严格复验：`bash scripts/repro/run_smoke_repro.sh`

## 通过标准
- 输出 `repro/reports/repro_report_<timestamp>.md` 且结果 PASS。
- `e2e.doe_change_max_abs == 0`。
- 指标偏差满足容差配置。
