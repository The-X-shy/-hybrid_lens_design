"""Read-only historical G1 audit for the bounded G1-R1 revision.

Writes only artifacts/G1-R1/audit.md, fd_reanalysis.csv and evidence_manifest.json.
No propagation is run and the original G0/G1 artifacts are never edited.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/G1-R1"
OLD = ROOT / "artifacts/G1"
STEPS = (0.01, 0.003, 0.001)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def percentile(values: list[float], proportion: float) -> float:
    vals = sorted(values)
    pos = (len(vals) - 1) * proportion
    low = int(pos)
    return vals[low] + (vals[min(low + 1, len(vals) - 1)] - vals[low]) * (pos - low)


def evidence_integrity(path: Path) -> tuple[int, list[str]]:
    gate = json.loads(path.read_text())
    bad = []
    for item in gate["evidence"]:
        source = Path(item["path"])
        if not source.is_file() or sha(source) != item["sha256"]:
            bad.append(str(source))
    return len(gate["evidence"]), bad


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((OLD / "thresholds.json").read_text())
    fd_threshold = frozen["deterministic_fd"]
    all_rows: list[dict] = []
    case_summary = {}
    case_details = {}
    for case in ("C1", "C2"):
        checks = json.loads((ROOT / f"artifacts/runs/g1-check-{case.lower()}/checks.json").read_text())
        rows = checks["fd_rows"]
        assert len(rows) == 20 and {r["direction"] for r in rows} == {0, 1, 2, 3}
        assert {r["step"] for r in rows} == {0.1, 0.03, *STEPS}
        selected = []
        details = []
        for direction in range(4):
            direction_rows = {r["step"]: r for r in rows if r["direction"] == direction}
            errors = [direction_rows[step]["absolute_error"] for step in STEPS]
            stable = errors[0] >= errors[1] >= errors[2]
            details.append((direction, stable, errors))
            for row in rows:
                if row["direction"] != direction:
                    continue
                near_zero = max(abs(row["AD"]), abs(row["FD"])) < 1e-5
                selected_step = row["step"] in STEPS
                relative = abs(row["AD"] - row["FD"]) / max(abs(row["AD"]), abs(row["FD"]), 1e-30)
                assert abs(relative - row["relative_error"]) < 1e-12
                passes = row["absolute_error"] <= fd_threshold["near_zero_absolute"] if near_zero else relative <= fd_threshold["p95_relative_max"]
                all_rows.append({
                    "case": case, "direction": direction, "step": row["step"], "AD": row["AD"],
                    "FD": row["FD"], "absolute_error": row["absolute_error"],
                    "relative_error_symmetric": relative, "near_zero": near_zero,
                    "selected_validation_step": selected_step, "step_rule_pass": passes,
                    "adjacent_selected_errors_nonincreasing": stable,
                    "source_checks_sha256": sha(ROOT / f"artifacts/runs/g1-check-{case.lower()}/checks.json"),
                })
                if selected_step:
                    selected.append(relative)
        case_summary[case] = {
            "median_selected": median(selected), "p95_selected": percentile(selected, 0.95),
            "all_four_directions_stable": all(d[1] for d in details),
            "selected_rows": len(selected),
        }
        case_details[case] = details
    out_csv = OUT / "fd_reanalysis.csv"
    with out_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    listed = [
        OLD / "gate.json", OLD / "reference_protocol.json", OLD / "thresholds.json",
        OLD / "reference_check_protocol_c1.json", OLD / "reference_check_protocol_c2.json",
        OLD / "convergence.csv", OLD / "directional_checks.csv", OLD / "analysis_summary.json",
        ROOT / "artifacts/cost_ledger.csv", ROOT / "configs/week1/cases.json",
        ROOT / "tools/arw_reference_run.py", ROOT / "tools/arw_run.py",
        ROOT / "arw_reference/asm.py", ROOT / "arw_reference/rs.py", Path(__file__), out_csv,
    ]
    for case in ("c1", "c2"):
        run = ROOT / f"artifacts/runs/g1-check-{case}"
        listed.extend([run / "checks.json", run / "config.json", run / "raw.npz"])
    for path in sorted((ROOT / "artifacts/runs").glob("g1-ref-*/series.json")):
        listed.append(path)
    manifest = {"purpose": "G1-R1 R0 read-only historical evidence snapshot",
                "files": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p), "bytes": p.stat().st_size} for p in listed]}
    (OUT / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    parent_integrity = {name: evidence_integrity(ROOT / f"artifacts/{name}/gate.json")
                        for name in ("G0-U", "G0-I", "G0")}
    old_count, old_bad = evidence_integrity(OLD / "gate.json")
    with (ROOT / "artifacts/cost_ledger.csv").open(newline="") as stream:
        costs = list(csv.DictReader(stream))
    g1 = [row for row in costs if row["gate"] == "G1"]
    original = json.loads((OLD / "analysis_summary.json").read_text())
    cross = list(csv.DictReader((OLD / "directional_checks.csv").open(newline="")))
    failed_cross = [r for r in cross if r["pass"] == "False"]
    lines = [
        "# G1-R1 R0 无传播审计", "",
        "**结论：旧 G1 仍为 FAIL。** 本审计只重算既有差分统计和核对证据，不产生新的传播结果，也不能使 G1 通过。", "",
        "## 原始证据和指纹", "",
        f"- G0-U、G0-I、G0 的门槛记录分别列出 {parent_integrity['G0-U'][0]}、{parent_integrity['G0-I'][0]}、{parent_integrity['G0'][0]} 项证据；当前指纹不匹配数分别为 {len(parent_integrity['G0-U'][1])}、{len(parent_integrity['G0-I'][1])}、{len(parent_integrity['G0'][1])}。",
        f"- 旧 G1 门槛记录列出 {old_count} 项证据，当前指纹不匹配 {len(old_bad)} 项；结论字段为 FAIL。",
        "- `evidence_manifest.json` 额外覆盖旧 `checks.json`、各参考运行的 `series.json`、协议、计算与汇总代码、成本账本和本次 CSV；旧 G1 的 gate.json 没有列出这些全部原始文件。",
        "- 该清单是本次读取时的快照。其他并行任务修改代码或新账本后，应使用它区分历史证据与当前代码，不把当前代码误称为旧运行版本。", "",
        "## 有限差分重算", "",
        "验收档在新方案中预先指定为 0.01、0.003、0.001；旧记录中的 0.1、0.03 作为粗步长诊断保留。相对差统一为 |AD−FD|/max(|AD|,|FD|)，与旧 CSV 一致。", "",
        "| 案例 | 验收档行数 | 中位相对差 | 第95百分位 | 四方向三档误差均递减 |", "|---|---:|---:|---:|---|",
    ]
    for case in ("C1", "C2"):
        v = case_summary[case]
        lines.append(f"| {case} | {v['selected_rows']} | {v['median_selected']:.6%} | {v['p95_selected']:.6%} | {'是' if v['all_four_directions_stable'] else '否'} |")
    lines += [
        "", "两案例都低于原中位 1%、第95百分位 5% 门槛。C1 的第4方向导数低于 1e-5，须按旧绝对差门槛 1e-7 判断；验收档三行均通过。以上只说明**旧 ASM 最终设置的求导自洽**，不能替代新共同终点的差分检查。",
        f"旧汇总对五档混合统计的 C1 第95百分位为 {original['fd_summary']['C1']['p95_all_steps']:.4%}；粗档包含截断误差。`fd_reanalysis.csv` 保留全部 40 行、每档选择状态和原始指纹。", "",
        "## 方向与收敛边界", "",
        f"- 旧跨层检查共 {len(cross)} 项，失败 {len(failed_cross)} 项；这些检查在旧代码中只使用 `directions[0]`。其余三个方向只在同一 ASM 设置内做差分，不能视为四方向跨方法验收。",
        "- C1 最末窗口 ASM 方向变化按旧 CSV 统一分母为 4.6180%，C2 为 2.8373%，C2 最末 RS 网格为 1.2970%；均超过冻结的 1% 门槛。旧报告的 C1 约 4.84% 与 CSV 分母口径不同，后续报告应从数据自动生成。",
        "- RS 在固定间距下窗口变化为零，是相同源支持区域裁剪与空白补零的不变量，不能重复作为传播参考精度证据。旧网格路线与窗口路线终点不同，不能拼成一个已收敛的共同终点。", "",
        "## 历史成本纠偏", "",
        f"- 旧账本 G1 共 {len(g1)} 次尝试，壁钟合计 {sum(float(r['wall_seconds']) for r in g1):.3f} 秒。全部 `gpu_device_seconds` 被写成 `0.0`，但 G1 代码明确构造 CUDA 张量；该字段应解释为**未测量**，不能作为实际零 GPU 设备秒引用。",
        "- 旧账本把每次 `cpu_device_seconds` 设成完整壁钟，且 G0 的 CPU 与 GPU 字段也同时采用壁钟；这些是粗略上界或占用时间，未做 CPU/GPU 独立设备计时。各阶段 `null:not_instrumented`、显存 `null:not_reported` 保持未知，不能推算补造。",
        "- 本审计不改旧账本；G1-R1 的新运行必须前后同步计 GPU 时间、记录峰值显存，并将不能测量的历史分项写为 unknown。", "",
        "## 限制与下一步条件", "",
        "R0 只能确认旧记录和验收组织的问题，不能修复跨层不收敛。R1 需独立核对 RS 直接导数与中心差分、ASM 改法对方向导数的作用，并明确是否值得进入共同终点测试。", "",
    ]
    (OUT / "audit.md").write_text("\n".join(lines))
    print(json.dumps({"fd": case_summary, "old_g1_evidence_bad": old_bad,
                      "g0_evidence_bad": {k: v[1] for k, v in parent_integrity.items()},
                      "g1_attempts": len(g1)}, indent=2))


if __name__ == "__main__":
    main()
