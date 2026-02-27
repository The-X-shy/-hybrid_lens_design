# Hybrid Lens Design

端到端折衍混合光学系统设计工程（GeoLens + DOE/Metasurface + 重建网络）。

本项目基于 DeepLens，提供从折射系统设计到折衍混合优化、再到端到端联合训练的完整可复现实验链路，支持：
- 折射透镜三模式训练（从零两阶段 / 现成透镜两阶段 / 现成透镜仅微调）
- 单波段或多波段 DOE 优化（默认单波段 `0.55 um`）
- 端到端阶段 `MHA + UNet` 网络，支持冻结 DOE 仅训练折射面与网络
- 全阶段结构化日志输出（`logs/metrics.jsonl` + `logs/summary.json`）
- Zemax/CODE V 导出（折射面）

---

## 1. 项目状态与目标

当前仓库聚焦毕业设计主线：

`折射设计 -> 折衍混合 -> 端到端联合 -> 超表面扩展`

重点实验配置为：
- 光学目标：`F-number = 2.8`，焦距目标 `80mm`，大视场约束
- 训练策略：分阶段训练 + 统一早停 + 结构化日志
- 中期/论文资料：与代码工程分离，放在 `文档/` 目录

---

## 2. 仓库结构

```text
hybrid_lens_design/
├── configs/                         # YAML 配置（支持 _extends 继承）
│   ├── default.yaml
│   ├── geolens.yaml
│   ├── hybrid.yaml
│   ├── meta.yaml
│   ├── e2e.yaml
│   ├── exp_f2p8_80mm_geolens.yaml
│   ├── exp_f2p8_80mm_hybrid.yaml
│   ├── exp_f2p8_80mm_e2e.yaml
│   ├── exp_f2p8_80mm_meta.yaml
│   └── exp_f2p8_80mm_e2e_smoke.yaml
├── src/
│   ├── geolens/                     # 折射设计训练与评估
│   ├── hybridlens/                  # DOE/超表面混合训练与评估
│   ├── e2e/                         # 端到端训练、MHA+UNet 网络
│   └── utils/                       # 配置、早停、日志、导出
├── scripts/
│   ├── train_geolens.py
│   ├── train_hybridlens.py
│   ├── train_metalens.py
│   ├── train_e2e.py
│   ├── run_pipeline.py
│   ├── evaluate_e2e.py
│   └── export_zemax.py
├── tutorial_e2e_hybrid_lens_design_6.ipynb  # 主实验 notebook 基线
├── 文档/
│   ├── 中期报告/
│   ├── 毕业论文/
│   └── 过程记录/
├── outputs/                         # 运行日志/临时输出（默认不建议入库）
├── results/                         # 训练结果（默认不建议入库）
├── environment.yml
├── pyproject.toml
└── README.md
```

说明：
- `datasets` 与 `deeplens` 通常是本地符号链接（见下文环境准备）。
- `文档/` 目录已与训练输出解耦，避免与 `outputs/` 混放。

---

## 3. 环境准备

### 3.1 Python / Conda

推荐：
- Python `>= 3.12`
- PyTorch `>= 2.0`
- CUDA 环境按机器安装

```bash
cd /Users/lilin/Desktop/hybrid_lens_design
conda env create -f environment.yml
conda activate hybrid_lens
```

### 3.2 安装 DeepLens 与本项目

本项目依赖 DeepLens。推荐可编辑安装：

```bash
# 1) 安装 DeepLens（按你的本地路径调整）
pip install -e /Users/lilin/Desktop/DeepLens-main

# 2) 安装当前项目
pip install -e .
```

### 3.3 数据与依赖路径（常见做法）

如需兼容现有脚本，可在仓库根目录创建符号链接：

```bash
ln -s /Users/lilin/Desktop/DeepLens-main/deeplens deeplens
ln -s /Users/lilin/Desktop/DeepLens-main/datasets datasets
```

---

## 4. 快速开始

### 4.1 折射阶段（GeoLens）

```bash
python scripts/train_geolens.py \
  --config configs/geolens.yaml \
  --mode scratch_two_stage
```

三种模式：
- `scratch_two_stage`：从零创建镜头，执行阶段1+阶段2
- `existing_two_stage`：读取现成镜头，执行阶段1+阶段2
- `existing_finetune_only`：读取现成镜头，跳过阶段1，仅阶段2微调

### 4.2 折衍混合阶段（Hybrid DOE）

```bash
python scripts/train_hybridlens.py \
  --config configs/hybrid.yaml \
  --geolens /path/to/final_lens.json \
  --wavelength 0.55
```

### 4.3 超表面扩展（Metasurface / Pixel2D）

```bash
python scripts/train_metalens.py \
  --config configs/meta.yaml \
  --geolens /path/to/final_lens.json
```

### 4.4 端到端联合训练（E2E）

```bash
python scripts/train_e2e.py \
  --config configs/e2e.yaml \
  --hybridlens /path/to/hybrid_final.json \
  --network_type mha_unet \
  --freeze_doe 1 \
  --wavelength 0.55
```

### 4.5 一键流水线

```bash
# 默认阶段：geolens,hybridlens,eval
python scripts/run_pipeline.py --stages geolens,hybridlens,eval

# 全流程：geolens,hybridlens,metalens,e2e,eval
python scripts/run_pipeline.py --full
```

---

## 5. F2.8 / 80mm 复现实验建议

仓库中已提供实验配置模板（`configs/exp_f2p8_80mm_*.yaml`）。
建议先做 smoke，再跑 full：

### 5.1 Smoke（快速连通）

```bash
python scripts/train_geolens.py --config configs/exp_f2p8_80mm_geolens.yaml --iterations 100 --finetune_iterations 100
python scripts/train_hybridlens.py --config configs/exp_f2p8_80mm_hybrid.yaml --iterations 100
python scripts/train_e2e.py --config configs/exp_f2p8_80mm_e2e_smoke.yaml
python scripts/train_metalens.py --config configs/exp_f2p8_80mm_meta.yaml --iterations 100
```

### 5.2 Full（正式训练）

```bash
python scripts/train_geolens.py --config configs/exp_f2p8_80mm_geolens.yaml --foclen 80 --fnum 2.8
python scripts/train_hybridlens.py --config configs/exp_f2p8_80mm_hybrid.yaml --iterations 5000 --wavelength 0.55
python scripts/train_e2e.py --config configs/exp_f2p8_80mm_e2e.yaml --epochs 5000 --freeze_doe 1 --network_type mha_unet
python scripts/train_metalens.py --config configs/exp_f2p8_80mm_meta.yaml --iterations 4000
```

注意：
- `exp_f2p8_80mm_geolens.yaml` 中如果焦距字段尚未更新为 80，可通过 CLI 覆盖（如上 `--foclen 80`）。
- 使用早停时，最终轮次可能小于配置上限。

---

## 6. 配置系统说明

### 6.1 继承机制

配置支持 `_extends`：
- 子配置覆盖父配置同名字段
- 常用方式：`exp_*.yaml` 继承 `geolens.yaml` / `hybrid.yaml` / `e2e.yaml` / `meta.yaml`

### 6.2 输出目录优先级

`src/utils/config.py` 逻辑：
1. 优先 `output.dir`
2. 否则使用 `output.base_dir`（可选时间戳）

### 6.3 统一早停配置

各阶段均支持：

```yaml
optimization:
  early_stop:
    enabled: true
    patience: 500
    min_delta: 1.0e-6
    mode: "min"
    monitor: "loss"  # 阶段不同可改 monitor 键
```

### 6.4 单波段优化

Hybrid 阶段：

```yaml
optimization:
  wavelengths: [0.55]
  wavelength_weights: [1.0]
```

E2E 阶段：

```yaml
optimization:
  wavelength: 0.55
  freeze_doe: true
```

---

## 7. 训练产物与日志约定

每阶段目录中统一包含：
- `logs/metrics.jsonl`：逐 step/epoch 结构化指标
- `logs/summary.json`：阶段汇总

### 7.1 GeoLens

关键文件：
- `final_lens.json`
- `final_lens.zmx`
- `final_lens_*.png`（analysis 输出，取决于 DeepLens）
- 过程快照：`iter*.json`、`fine-tune/iter*.json`

### 7.2 Hybrid DOE

关键文件：
- `hybrid_final.json`
- `hybridlens_final.json`
- `hybrid_iter*.json` / `hybridlens_iter*.json`
- `hybrid_psf_iter*.png`
- `hybrid_loss_curve.png`

### 7.3 Metasurface

关键文件：
- `metasurface_iterfinal.json`
- `phase_map_iterfinal.png` / `phase_map_iterfinal.pt`
- 过程快照：`metasurface_iter*.json`、`phase_map_iter*.png/.pt`

### 7.4 E2E

关键文件：
- `network_epoch*.pth`
- `hybridlens_epoch*.json`
- `psf_epoch*.png`
- `logs/summary.json` 中包含 `doe_change_max_abs`（用于验证 DOE 是否冻结）

---

## 8. 评估与导出

### 8.1 评估

```bash
python scripts/evaluate_e2e.py --help
python scripts/test_inference.py --help
python scripts/remote_test_inference.py --help
```

### 8.2 Zemax/CODE V 导出（折射面）

```bash
python scripts/export_zemax.py --lens /path/to/final_lens.json --all
```

说明：
- 导出主要面向折射系统；DOE/超表面通常需独立工艺或额外后处理。

---

## 9. 文档目录说明

- `文档/中期报告/`：中期报告 LaTeX/PDF/PPT 与图集
- `文档/毕业论文/`：BIT 论文模板、严格 LaTeX 模板
- `文档/过程记录/`：阶段状态文档与训练过程记录

推荐入口：
- 中期报告：`文档/中期报告/midterm_report.tex`
- 毕业论文严格模板：`文档/毕业论文/latex_strict_bit/main.tex`

---

## 10. 常见问题（FAQ）

### Q1: 数据路径不存在怎么办？
- E2E 脚本对 `BSDS300` 路径支持自动下载逻辑。
- 其他数据集请在 `configs/*.yaml` 中显式设置 `dataset.train_path` / `val_path`。

### Q2: 显存不足（OOM）？
- 降低 `batch_size`
- 降低 `psf_size`
- 降低 `spp`
- 减少 `test_per_iter` 或 `test_per_epoch` 频率

### Q3: LPIPS 不可用？
- 安装 `lpips`
- 或在 E2E 配置中将 `loss_weights.lpips` 置为 `0.0`

### Q4: 如何验证 E2E 阶段确实冻结 DOE？
- 检查 `logs/summary.json` 的 `doe_change_max_abs`
- 若接近 `0`，说明 DOE 参数基本未变化

---

## 11. 参考工作

- Yang, Fu, Heidrich. Curriculum learning for ab initio deep learned refractive optics. Nature Communications, 2024.
- Yang et al. End-to-End Hybrid Refractive-Diffractive Lens Design with Differentiable Ray-Wave Model. SIGGRAPH Asia, 2024.

---

## 12. 许可与使用

当前仓库尚未单独提供 `LICENSE` 文件。
如需公开发布或第三方协作，建议补充明确许可证（例如 MIT/Apache-2.0）后再进行二次分发。
