#!/usr/bin/env python3
"""Compare current run metrics against reproducibility baseline.
将当前实验指标与基线进行容差比较，输出 PASS/FAIL 报告。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel_err(curr: float, base: float) -> float:
    denom = max(abs(base), 1e-12)
    return abs(curr - base) / denom


def load_current_metrics(run_root: Path) -> dict:
    files = {
        "geolens": run_root / "01_geolens/logs/summary.json",
        "hybrid": run_root / "02_hybrid/logs/summary.json",
        "e2e": run_root / "03_e2e/logs/summary.json",
        "meta": run_root / "04_meta/logs/summary.json",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing summary files: {missing}")

    g = read_json(files["geolens"])
    h = read_json(files["hybrid"])
    e = read_json(files["e2e"])
    m = read_json(files["meta"])

    return {
        "geolens": {"final_rms": float(g["final_rms"])},
        "hybrid": {"final_loss": float(h["final_loss"])},
        "e2e": {
            "final_total_loss": float(e["final_total_loss"]),
            "doe_change_max_abs": float(e["doe_change_max_abs"]),
        },
        "meta": {"final_loss": float(m["final_loss"])},
    }


def build_report(rows: list[dict], overall_pass: bool, baseline_src: str, run_root: str) -> str:
    lines = []
    lines.append("# Reproducibility Verification Report")
    lines.append("")
    lines.append(f"- Generated (UTC): `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- Baseline: `{baseline_src}`")
    lines.append(f"- Current run: `{run_root}`")
    lines.append(f"- Result: `{ 'PASS' if overall_pass else 'FAIL' }`")
    lines.append("")
    lines.append("| Stage | Metric | Baseline | Current | RelErr | Tolerance | Pass |")
    lines.append("|---|---:|---:|---:|---:|---:|:---:|")
    for r in rows:
        lines.append(
            f"| {r['stage']} | {r['metric']} | {r['baseline']:.8f} | {r['current']:.8f} | "
            f"{r['rel_err']:.6f} | {r['tol']:.6f} | {'Y' if r['pass'] else 'N'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify reproducibility metrics")
    parser.add_argument("--baseline", required=True, help="Baseline JSON path")
    parser.add_argument("--current", required=True, help="Current run root path")
    parser.add_argument(
        "--tolerance",
        default="repro/baselines/tolerance.yaml",
        help="Tolerance YAML path",
    )
    parser.add_argument("--report", required=True, help="Output markdown report path")
    args = parser.parse_args()

    baseline = read_json(Path(args.baseline))
    baseline_metrics = baseline["metrics"]
    tol = yaml.safe_load(Path(args.tolerance).read_text(encoding="utf-8"))
    curr = load_current_metrics(Path(args.current))

    rows = []
    overall_pass = True

    checks = [
        ("geolens", "final_rms"),
        ("hybrid", "final_loss"),
        ("e2e", "final_total_loss"),
        ("meta", "final_loss"),
    ]
    for stage, metric in checks:
        b = float(baseline_metrics[stage][metric])
        c = float(curr[stage][metric])
        t = float(tol["tolerances"][stage][metric])
        e = rel_err(c, b)
        p = e <= t
        overall_pass = overall_pass and p
        rows.append(
            {
                "stage": stage,
                "metric": metric,
                "baseline": b,
                "current": c,
                "rel_err": e,
                "tol": t,
                "pass": p,
            }
        )

    # Hard constraint: e2e DOE frozen
    expected_doe = float(tol["constraints"]["e2e"]["doe_change_max_abs"])
    current_doe = float(curr["e2e"]["doe_change_max_abs"])
    doe_pass = math.isclose(current_doe, expected_doe, abs_tol=1e-12)
    overall_pass = overall_pass and doe_pass
    rows.append(
        {
            "stage": "e2e",
            "metric": "doe_change_max_abs",
            "baseline": expected_doe,
            "current": current_doe,
            "rel_err": 0.0 if doe_pass else 1.0,
            "tol": 0.0,
            "pass": doe_pass,
        }
    )

    report = build_report(
        rows=rows,
        overall_pass=overall_pass,
        baseline_src=baseline.get("source", args.baseline),
        run_root=args.current,
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"[INFO] report written: {report_path}")
    print(f"[INFO] result: {'PASS' if overall_pass else 'FAIL'}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
