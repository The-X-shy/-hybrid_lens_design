#!/usr/bin/env python3
"""
Generate markdown training log document from experiment outputs.
从实验输出自动生成 Markdown 训练日志文档
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_key_files(stage_dir: Path) -> List[str]:
    patterns = [
        "final_lens.json",
        "hybrid_final.json",
        "hybridlens_final.json",
        "metalens_final.json",
        "metasurface_iterfinal.json",
        "network_epochfinal.pth",
        "network_epoch*.pth",
        "*.png",
    ]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(stage_dir.glob(pattern))
    uniq = sorted({f for f in files if f.exists()})
    return [str(p) for p in uniq[:50]]


def stage_block(stage_name: str, stage_dir: Path) -> str:
    log_dir = stage_dir / "logs"
    summary = read_json(log_dir / "summary.json")
    metrics_file = log_dir / "metrics.jsonl"
    metrics_lines = 0
    if metrics_file.exists():
        metrics_lines = sum(1 for _ in open(metrics_file, "r", encoding="utf-8"))

    lines = [f"## {stage_name}", ""]
    lines.append(f"- 目录：`{stage_dir}`")
    lines.append(f"- 日志文件：`{metrics_file}`（{metrics_lines} 行）")
    if summary:
        lines.append(f"- Summary：`{log_dir / 'summary.json'}`")
        lines.append("- 关键字段：")
        for k in sorted(summary.keys()):
            v = summary[k]
            if isinstance(v, (dict, list)):
                continue
            lines.append(f"  - `{k}`: `{v}`")
    else:
        lines.append("- Summary：未找到")

    key_files = list_key_files(stage_dir)
    lines.append("- 关键产物：")
    if key_files:
        for p in key_files[:20]:
            lines.append(f"  - `{p}`")
    else:
        lines.append("  - 无")
    lines.append("")
    return "\n".join(lines)


def build_doc(
    experiment_name: str,
    base_dir: Path,
    smoke_dir: Optional[Path],
    output_path: Path,
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append(f"# 训练日志文档：{experiment_name}")
    lines.append("")
    lines.append(f"- 生成时间：`{now}`")
    lines.append(f"- 正式训练目录：`{base_dir}`")
    if smoke_dir:
        lines.append(f"- 烟雾测试目录：`{smoke_dir}`")
    lines.append("")

    lines.append("## 实验目标与参数")
    lines.append("")
    lines.append("- 折射：foclen=80mm, fnum=2.8, FOV=70°")
    lines.append("- 轮数：GeoLens(3000+3000), Hybrid(5000), E2E(5000), Meta(4000)")
    lines.append("- 早停：patience=500, min_delta=1e-6")
    lines.append("")

    stage_map = [
        ("GeoLens", base_dir / "01_geolens"),
        ("Hybrid DOE", base_dir / "02_hybrid"),
        ("E2E", base_dir / "03_e2e"),
        ("Metasurface", base_dir / "04_meta"),
    ]
    for stage_name, stage_dir in stage_map:
        lines.append(stage_block(stage_name, stage_dir))

    if smoke_dir:
        lines.append("## 烟雾测试结果目录")
        lines.append("")
        lines.append(f"- ` {smoke_dir}`")
        lines.append("")

    lines.append("## 回传与归档")
    lines.append("")
    lines.append("- 本文档与全部结果已计划回传到以下本地目录：")
    lines.append("  - `/Users/lilin/Desktop/deeplens backup/results/`")
    lines.append("  - `/Users/lilin/Desktop/hybrid_lens_design/results/`")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate markdown training log")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--base_dir", type=str, required=True, help="Full training result root")
    parser.add_argument("--smoke_dir", type=str, default=None, help="Smoke test result root")
    parser.add_argument("--output", type=str, required=True, help="Output markdown path")
    return parser.parse_args()


def main():
    args = parse_args()
    build_doc(
        experiment_name=args.experiment_name,
        base_dir=Path(args.base_dir),
        smoke_dir=Path(args.smoke_dir) if args.smoke_dir else None,
        output_path=Path(args.output),
    )
    print(f"Training log markdown generated: {args.output}")


if __name__ == "__main__":
    main()
