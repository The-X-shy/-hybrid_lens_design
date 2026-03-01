# 环境可复现操作手册（Linux + CUDA）

本手册用于固定 DeepLens 版本并验证本项目可复现性。

## 1. 目标

- 依赖固定：仅使用仓库内 `third_party/deeplens-core`
- 环境固定：Conda + pip 锁文件
- 数据固定：BSDS300/BSDS300_smoke 哈希校验
- 结果验收：按容差比较 `logs/summary.json`，生成 PASS/FAIL 报告

## 2. 关键锁定文件

- DeepLens 锁定元数据：`repro/deeplens_lock.json`
- DeepLens 文件哈希：`repro/deeplens_sha256_manifest.txt`
- Conda 显式锁：`repro/environment/conda-linux-cuda12.1-explicit.txt`
- （可选）本地开发快照：`repro/environment/conda-local-osx-arm64-explicit.txt`
- pip 锁：`repro/environment/pip-freeze.txt`
- 指标基线：`repro/baselines/smoke_metrics_baseline.json`
- 容差配置：`repro/baselines/tolerance.yaml`

## 3. 一键流程

```bash
cd /path/to/hybrid_lens_design

# 1) 构建环境并安装锁定依赖
bash scripts/repro/bootstrap_linux_cuda.sh

# 2) 跑 smoke 并自动生成验收报告
bash scripts/repro/run_smoke_repro.sh
```

说明：
- `conda-linux-cuda12.1-explicit.txt` 若为占位文件，`bootstrap_linux_cuda.sh` 会自动回退 `environment.yml` 创建环境，并在流程结束后导出真实 Linux 显式锁。
- `run_smoke_repro.sh` 默认不允许自动生成数据清单；若首次生成清单，显式执行：`ALLOW_MANIFEST_BOOTSTRAP=1 bash scripts/repro/run_smoke_repro.sh`。

## 4. 手动检查命令

```bash
# 校验 vendored DeepLens 是否被修改
python scripts/repro/check_deeplens_lock.py --strict-import

# 采集环境指纹
python scripts/repro/capture_env_fingerprint.py

# 手动验证某次 run
python scripts/repro/verify_repro_metrics.py \
  --baseline repro/baselines/smoke_metrics_baseline.json \
  --current /path/to/run_root \
  --tolerance repro/baselines/tolerance.yaml \
  --report repro/reports/repro_report_manual.md
```

## 5. 验收标准（统计一致）

- GeoLens `final_rms` 相对误差 <= 10%
- Hybrid `final_loss` 相对误差 <= 10%
- E2E `final_total_loss` 相对误差 <= 15%
- Meta `final_loss` 相对误差 <= 15%
- E2E `doe_change_max_abs` 必须等于 0

## 6. 常见失败与处理

1. `deeplens lock verification failed`
- 说明 `third_party/deeplens-core` 内容被改动。
- 处理：恢复该目录，或重新生成 `repro/deeplens_sha256_manifest.txt` 与 `repro/deeplens_lock.json`。

2. `dataset manifest mismatch`
- 说明数据文件发生变化或路径错误。
- 处理：确认数据路径，再用 `scripts/repro/check_dataset_manifest.py --create-if-missing` 重新生成清单。

3. `import deeplens failed`
- 说明环境未安装锁定版依赖。
- 处理：重新执行 `bash scripts/repro/bootstrap_linux_cuda.sh`。

4. 复现报告 FAIL
- 说明指标超出容差或 DOE 未冻结。
- 处理：先检查 `repro/reports/env_fingerprint_*.json` 与配置是否一致，再排查训练脚本参数覆盖。
