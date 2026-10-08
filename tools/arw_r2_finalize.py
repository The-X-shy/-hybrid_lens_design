"""Audit the bounded G1-R2 results and derive a non-PASS verdict."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/G1-R2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def text_file(path, content):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(content)


def raw_intact(run, result_name):
    data = BASE / "runs" / run / "data"
    record = load(data / result_name)
    failures = [relative for relative, expected in record["raw_sha256"].items() if not (data / relative).is_file() or sha(data / relative) != expected]
    if failures:
        raise ValueError(f"raw field fingerprints changed: {run}: {failures[:3]}")
    return len(record["raw_sha256"])


def main():
    protocol = load(BASE / "protocol.json")
    budget = load(BASE / "budget.json")
    for relative, expected in protocol["source_hashes"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"frozen source changed: {relative}")
    for name, state in protocol["parent_state"].items():
        gate = load(ROOT / "artifacts" / name / "gate.json")
        if gate["status"] != state or any(not Path(item["path"]).is_file() or sha(item["path"]) != item["sha256"] for item in gate["evidence"]):
            raise ValueError(f"parent evidence changed: {name}")
    with (BASE / "cost_ledger.csv").open(newline="") as handle:
        costs = list(csv.DictReader(handle))
    if [row["batch"] for row in costs] != ["B1", "B2", "B3", "B4", "B4L"] or any(row["status"] != "SUCCESS" for row in costs):
        raise ValueError("bounded batch record incomplete")
    for row in costs:
        if sha(BASE / f"{row['batch']}_schedule.json") != row["schedule_sha256"]:
            raise ValueError("schedule fingerprint changed")
        status = load(BASE / "runs" / row["run_id"] / "status.json")
        if status["status"] != row["status"]:
            raise ValueError("ledger and run status mismatch")
    raw_counts = {
        "B1": raw_intact("r2-b1-01", "fine_baseline.json"),
        "B3": raw_intact("r2-b3-01", "intervention.json"),
        "B4": raw_intact("r2-b4-01", "c1_reference_check.json"),
    }
    b3 = load(BASE / "runs/r2-b3-01/data/comparison_summary.json")
    independent = load(BASE / "b3_review.json")
    if not b3["continuation_conditions_all_met"] or independent["all_claims_match"] is not True:
        raise ValueError("C2 B3 independent review failed")
    b4 = load(BASE / "runs/r2-b4-01/data/c1_comparison.json")
    b4l = load(BASE / "runs/r2-b4l-01/data/summary.json")
    if b4["all_grid_rules_met"] is not False or b4l["status"] != "SUCCESS":
        raise ValueError("C1 stop condition not established")
    failed = [item for pair in b4["adjacent"] for item in pair["direction"] if item["pass"] is not True]
    if len(failed) != 1 or failed[0]["direction"] != 3 or failed[0]["near_zero_rule"] is not True:
        raise ValueError("unexpected C1 failure pattern")
    if failed[0]["absolute_error"] <= protocol["acceptance"]["near_zero_absolute"]:
        raise ValueError("C1 failure no longer exceeds threshold")
    # Recompute selected source/sensor comparison values from the saved arrays.
    local = np.load(BASE / "runs/r2-b4l-01/data/local_raw.npz")
    def rel(a, b):
        return float(np.linalg.norm(a-b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-30))
    c1_source = [rel(local["s4_t6_field"][i], local["s6_t6_field"][i]) for i in range(5)]
    c1_sensor = [rel(local["s6_t4_dintensity"][i], local["s6_t6_dintensity"][i]) for i in range(4)]
    if max(abs(x-y) for x,y in zip(c1_source, b4l["comparison"]["source_4_6_field_rel_l2"])) > 1e-14:
        raise ValueError("C1 local source summary changed")
    if max(abs(x-y) for x,y in zip(c1_sensor, b4l["comparison"]["sensor_4_6_dintensity_rel_l2"])) > 1e-14:
        raise ValueError("C1 local sensor summary changed")
    spent_cpu = sum(float(row["cpu_process_wall_upper_seconds"]) for row in costs)
    spent_gpu = sum(float(row["gpu_process_wall_upper_seconds"]) for row in costs)
    peak_cpu = max(int(row["peak_cpu_bytes"]) for row in costs)
    if len(costs) > budget["max_research_batches"] or spent_cpu > budget["max_cpu_process_wall_seconds"] or spent_gpu > budget["max_gpu_process_seconds"] or peak_cpu > budget["max_peak_cpu_bytes"]:
        raise ValueError("revision budget exceeded")
    summary = {
        "status": "INCONCLUSIVE", "C2_B3_grid_conditions_met": True,
        "C1_B4_grid_conditions_met": False,
        "C1_failed_comparison": {"from_q": 4, "to_q": 6, "direction": 3, "absolute_error": failed[0]["absolute_error"], "near_zero_absolute_limit": protocol["acceptance"]["near_zero_absolute"]},
        "C1_B4L_local_source_4_6_max_relative_l2": max(c1_source),
        "C1_B4L_local_sensor_4_6_max_relative_l2": max(c1_sensor),
        "formal_B5_B6_B7_started": False,
        "runs": len(costs), "run_budget": budget["max_research_batches"],
        "cpu_process_wall_upper_seconds": spent_cpu,
        "gpu_process_wall_upper_seconds": spent_gpu,
        "peak_cpu_bytes": peak_cpu,
        "raw_field_fingerprints_verified": raw_counts,
    }
    save_json(BASE / "analysis_summary.json", summary)
    text_file(BASE / "b3_review.md", (
        "# B3 独立复核\n\n"
        "已重算 C2 三档的前向、功率及四方向相邻差；与原始数组和冻结门槛一致。边界面积权重在三档均还原解析圆面积，入射功率使用面积权重的一次方。\n\n"
        f"最后两档方向 0 差 {100*b3['direction0_last_grid_relative']:.3f}%，固定 25 像素四方向差的最大相对 L2 为 {100*b3['local_25pixel_dintensity_rel_l2_max']:.3f}%。B3 的继续条件经独立重算成立；这只支持进入 C1 检查，不构成 G1 资格。详见 `b3_review.json`。\n"
    ))
    text_file(BASE / "source_sensor_conventions.md", (
        "# 本轮源与传感器数值约定\n\n"
        "C2 B3 与 C1 B4 使用原硬圆孔。边界子单元以真实圆内面积比例 α 加权复场，方向变化场使用同一 α；物理入射功率为 Σα·dx²，不按 |αE|² 求和。传感器仍在原 64×64 物理像素内平均强度；相位、四方向、目标及损失权重均未改变。\n\n"
        "局部求积直接跨圆边界分段，并分别记录完整孔内与边界复场；固定 25 像素的结果不能代表整幅传感器。ASM 原法及带宽候选未在本轮正式共同终点运行，因此没有把局部结果当作 ASM/RS 交叉通过。\n"
    ))
    save_json(BASE / "references/index.json", {"status": "NO_QUALIFIED_REFERENCE", "candidates": [], "reason": "C1 B4 preregistered grid condition failed; B5-B7 formal joint-endpoint and four-direction cross-method checks were not run", "old_candidate_only_not_promoted": True})
    report = (
        "# G1-R2 有限优化实验结果\n\n"
        "**结论：仍未取得 G1 参考资格。** 原 G1=FAIL、G1-R1=INCONCLUSIVE 均保留；G2–G5 未启动。\n\n"
        "## 已取得的证据\n\n"
        "- 离线审计区分了 C2 粗档 94.70% 与细端 1.30% 的差异，并核对了旧记录的版本与指纹。\n"
        "- B1 的 C2 细网格四方向与历史同设置数据吻合；最高两档方向 0 差约 1.30%，自身差分核验通过。\n"
        "- B2 的固定 25 像素源/传感器分离检查稳定；它没有代表全图。\n"
        f"- B3 对真实圆孔边界只作一次面积权重修正。C2 最后两档方向 0 差降至 {100*b3['direction0_last_grid_relative']:.3f}%，四方向及前向两次相邻检查满足本阶段门槛；局部结果更接近独立求积。\n"
        f"- B4 将同一修正用于 C1。前向两次相邻检查通过，方向 3 在 q=4→6 的近零绝对差为 {failed[0]['absolute_error']:.9g}，超过冻结门槛 {protocol['acceptance']['near_zero_absolute']:.9g}；q=6→8 的该项通过。\n"
        f"- B4L 对 C1 相同 25 像素补查：最高两档源复场/四方向相对 L2 最大 {max(c1_source):.3g}，传感器方向变化最大 {max(c1_sensor):.3g}，局部自身稳定。这不能改写 B4 的全图失败。\n\n"
        "## 停止与成本\n\n"
        "C1 的预设三档未全部通过，因此没有运行正式五点、完整 ASM/RS 交叉和终点四方向差分。不能从已测局部结果宣布独立参考成立，也不追加更细网格或更大窗口。\n\n"
        f"本轮计费 {len(costs)} 次研究批次，CPU 子进程墙钟上界 {spent_cpu:.2f} 秒，GPU 子进程 0 秒，峰值 CPU 内存 {peak_cpu/1024**3:.2f} GiB；均在冻结预算内。账本 B2 的原始内存字段曾为 unknown，已依据同次运行的测量结果更正，旧表单独保留。\n\n"
        "本轮状态为 INCONCLUSIVE。依方案，若完整有限修订仍未取得资格，应重新评估参考计算和研究范围，不再仅换修订编号继续加密。\n"
    )
    text_file(BASE / "reference_report.md", report)
    evidence = [
        BASE / "protocol.json", BASE / "budget.json", BASE / "audit.md", BASE / "evidence_index.csv", BASE / "offline_reanalysis.json",
        BASE / "B1_schedule.json", BASE / "B2_schedule.json", BASE / "B3_schedule.json", BASE / "B4_schedule.json", BASE / "B4L_schedule.json",
        BASE / "commands_verified.md", BASE / "b2_decision.md", BASE / "b2_comparison.csv", BASE / "b3_review.json", BASE / "b3_review.md",
        BASE / "cost_ledger.csv", BASE / "cost_ledger.csv.before_b2_memory_reconciliation", BASE / "source_sensor_conventions.md", BASE / "analysis_summary.json", BASE / "reference_report.md", BASE / "references/index.json",
        ROOT / "tools/arw_r2.py", ROOT / "tools/arw_r2_fine.py", ROOT / "tools/arw_r2_local.py", ROOT / "tools/arw_r2_boundary.py", ROOT / "tools/arw_r2_c1.py", ROOT / "tools/arw_r2_local_c1.py", ROOT / "tools/arw_r2_finalize.py",
        ROOT / "artifacts/G0/gate.json", ROOT / "artifacts/G1/gate.json", ROOT / "artifacts/G1-R1/gate.json",
    ]
    by_run = {
        "r2-b1-01": ["fine_baseline.csv", "fine_fd.csv", "fine_baseline.json", "metrics.json"],
        "r2-b2-01": ["estimate.json", "nodes_weights.npz", "local_raw.npz", "local_quadrature.csv", "summary.json"],
        "r2-b3-01": ["comparison_summary.json", "intervention.csv", "intervention.json", "local_comparison.csv", "metrics.json"],
        "r2-b4-01": ["c1_comparison.json", "c1_reference_check.json", "metrics.json"],
        "r2-b4l-01": ["estimate.json", "nodes_weights.npz", "local_raw.npz", "local_quadrature.csv", "summary.json"],
    }
    for run, files in by_run.items():
        evidence += [BASE / "runs" / run / "data" / name for name in files]
        evidence += [BASE / "runs" / run / name for name in ("config.json", "status.json", "stdout.log", "stderr.log")]
    if any(not path.is_file() for path in evidence):
        raise FileNotFoundError("required R2 evidence absent")
    save_json(BASE / "gate.json", {
        "status": "INCONCLUSIVE", "protocol_version": "G1-R2/1.0", "parent_g0": "PASS", "old_g1": "FAIL", "g1_r1": "INCONCLUSIVE",
        "criteria": summary,
        "unresolved": ["C1 q4-to-q6 near-zero direction 3 exceeds frozen absolute limit", "formal common endpoint ASM/RS cross-method checks not run"],
        "evidence": [{"path": str(path), "sha256": sha(path)} for path in evidence],
        "allowed_next_action": "stop G2-G5; reassess numerical reference and study scope under a separately justified plan",
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
