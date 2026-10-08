"""Derive the bounded G1-R1 pilot verdict from immutable run evidence."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/G1-R1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    protocol_path = BASE / "pilot_protocol_v1_1.json"
    protocol = load(protocol_path)
    asm_dir = BASE / "runs/r1-pilot-asm-01-r2/data"
    rs_dir = BASE / "runs/r1-pilot-rs-01/data"
    asm_metrics = load(asm_dir / "metrics.json")
    rs = load(rs_dir / "rs_check.json")
    rejudge = load(BASE / "rs_check_rejudge.json")
    with (asm_dir / "pilot_results.csv").open(newline="") as handle:
        asm = list(csv.DictReader(handle))
    with (BASE / "cost_ledger.csv").open(newline="") as handle:
        cost = list(csv.DictReader(handle))
    if len(cost) != 3 or len(asm) != 7 or len(rs["results"]) != 3:
        raise ValueError("pilot evidence incomplete")
    if rejudge["protocol_sha256"] != sha(protocol_path) or rejudge["input_sha256"] != sha(rs_dir / "rs_check.json"):
        raise ValueError("RS rejudgment input mismatch")
    if not all(entry["status"] == "SUCCESS" for entry in cost if not entry["run_id"].endswith("asm-01")):
        raise ValueError("successful pilot run missing")
    old = load(ROOT / "artifacts/G1/gate.json")
    if old["status"] != "FAIL" or sha(ROOT / "artifacts/G1/gate.json") != protocol["old_evidence"]["g1_gate_sha256"]:
        raise ValueError("old G1 result changed")
    g0 = load(ROOT / "artifacts/G0/gate.json")
    if g0["status"] != "PASS" or sha(ROOT / "artifacts/G0/gate.json") != protocol["old_evidence"]["g0_gate_sha256"]:
        raise ValueError("G0 result changed")
    historical = load(BASE / "evidence_manifest.json")
    if any(not (ROOT / item["path"]).is_file() or sha(ROOT / item["path"]) != item["sha256"] for item in historical["files"]):
        raise ValueError("historical audit snapshot changed")
    arrays = np.load(asm_dir / "raw.npz")
    rs_row = next(row for row in asm if row["variant"] == "RS")
    reference_image = arrays["rs_normalized"]
    rows = []
    for row in asm:
        result = dict(row)
        if row["variant"] != "RS":
            n = row["n"]
            result["l1_vs_rs"] = float(np.abs(arrays[f"{row['variant']}_{n}_normalized"] - reference_image).sum())
            a, b = float(row["total"]), float(rs_row["total"])
            result["direction_relative_vs_rs"] = abs(a-b) / max(abs(a), abs(b), 1e-30)
            result["power_relative_vs_rs"] = abs(float(row["sensor_power"])-float(rs_row["sensor_power"])) / float(rs_row["sensor_power"])
        else:
            result.update(l1_vs_rs=0.0, direction_relative_vs_rs=0.0, power_relative_vs_rs=0.0)
        rows.append(result)
    top = BASE / "pilot_results.csv"
    with top.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    best = next(row for row in rows if row["variant"] == "bandlimited" and row["n"] == "1024")
    threshold = protocol["preregistered_rules"]
    forward_ok = float(best["l1_vs_rs"]) <= threshold["forward_l1_max"] and float(best["power_relative_vs_rs"]) <= threshold["power_relative_max"]
    direction_ok = float(best["direction_relative_vs_rs"]) <= threshold["direction_relative_max"]
    rs_ok = rejudge["status"] == "PASS" and rejudge["checked_steps"] == 9
    pilot_continue = rs_ok and forward_ok and direction_ok
    c2_coarse, c2_fine = [row for row in rejudge["settings"] if row["case"] == "C2"]
    c2_grid_change = abs(c2_coarse["direct_total_derivative"] - c2_fine["direct_total_derivative"]) / max(abs(c2_coarse["direct_total_derivative"]), abs(c2_fine["direct_total_derivative"]))
    summary = {
        "pilot_continue": pilot_continue,
        "rs_direct_fd_all9": rs_ok,
        "asm_best_forward_cross_pass": forward_ok,
        "asm_best_direction_cross_pass": direction_ok,
        "asm_best_l1_vs_rs": float(best["l1_vs_rs"]),
        "asm_best_direction_relative_vs_rs": float(best["direction_relative_vs_rs"]),
        "asm_best_discarded_source_spectrum": float(best["discarded_spectral_energy_fraction"]),
        "asm_best_discarded_tangent_spectrum": float(best["discarded_tangent_spectral_energy_fraction"]),
        "c2_coarse_fine_rs_direction_relative_change": c2_grid_change,
        "c2_coarse_cancellation_factor": c2_coarse["component_cancellation_factor"],
        "attempts": len(cost), "successful_diagnostic_batches": sum(row["status"] == "SUCCESS" for row in cost),
        "implementation_retries": sum(row["run_id"].endswith("-r2") for row in cost),
        "cpu_wall_seconds_upper_bound": sum(float(row["cpu_wall_seconds_upper_bound"]) for row in cost),
        "gpu_device_seconds_upper_bound": sum(float(row["gpu_device_seconds_upper_bound"]) for row in cost),
        "peak_gpu_bytes_observed": asm_metrics["timing"]["peak_gpu_bytes"],
    }
    save(BASE / "analysis_summary.json", summary)
    refs = {"status": "NO_QUALIFIED_REFERENCE", "candidates": [], "reason": "joint endpoint and four-direction formal checks not run after frozen pilot stop rule", "old_candidate_only_files": ["artifacts/G1/references/C1_candidate_only.npz", "artifacts/G1/references/C2_candidate_only.npz"]}
    save(BASE / "references/index.json", refs)
    report = (
        "# G1-R1 有限修订结果\n\n"
        "**结论：仍未取得 G1 参考资格。** 本轮按预先写明的停止条件结束于诊断阶段，没有启动共同终点五点验收或 G2–G5。原 G1 的 FAIL 保留。\n\n"
        "## 复核与诊断\n\n"
        "- 旧结果核对：四方向小步长差分符合原门槛，但旧跨层检查只覆盖方向 0，10/18 项失败。详见 `audit.md`。\n"
        "- RS 直接方向计算在 3 个设置、9 个差分点与自身中心差分相符；这只证明当前离散计算的求导自洽。\n"
        f"- C2 的两档诊断网格方向结果相差 {100*c2_grid_change:.2f}%；粗网格图像项与功率项抵消约 {c2_coarse['component_cancellation_factor']:.1f} 倍，功率项是变化的主要来源。尚未证明细网格收敛。\n"
        f"- C1 的带宽诊断在最大测试窗口与 RS 的图像差为 {float(best['l1_vs_rs']):.6f}（门槛 0.003），方向差为 {100*float(best['direction_relative_vs_rs']):.3f}%（门槛 1%）。方向仍超限。该变体删去 {100*float(best['discarded_tangent_spectral_energy_fraction']):.1f}% 的输入方向变化频谱，不能据此替代原传播方法。\n\n"
        "## 实验用量与记录\n\n"
        f"- 两个诊断批次完成；首次 ASM 运行因 NumPy 记录代码错误失败，计费后按唯一一次实现错误重试成功。总计 {len(cost)} 次尝试。\n"
        f"- 本轮 GPU 子进程占用上界 {summary['gpu_device_seconds_upper_bound']:.2f} 秒，CPU 墙钟占用上界 {summary['cpu_wall_seconds_upper_bound']:.2f} 秒；GPU 实测峰值 {summary['peak_gpu_bytes_observed']/1024**2:.1f} MiB。历史 G1 的 GPU 时间曾被记为 0，但实际为未测量，不能引用为零成本。\n"
        "- 原始图像、分项损失与导数、所有步长、版本指纹和失败日志均保留。旧 G1 结果及旧账本未覆盖。\n\n"
        "- RS 运行时脚本已保存快照；运行后对近零判据的修正只作离线重判，没有重新传播或修改原始数值。\n\n"
        "## 停止原因\n\n"
        "新变体只把前向比较降到门槛内，方向比较仍未通过，并大幅删除了方向输入的频谱。按冻结的诊断继续条件，不进入正式五点验收；本轮资格状态为 INCONCLUSIVE。若继续研究，需另定有根据的数值方案、共同终点和预算，重新取得 G1 资格。\n"
    )
    (BASE / "reference_report.md").write_text(report)
    evidence_paths = [
        ROOT / "artifacts/G0/gate.json", ROOT / "artifacts/G1/gate.json",
        BASE / "pilot_protocol.json", protocol_path, BASE / "audit.md", BASE / "fd_reanalysis.csv", BASE / "evidence_manifest.json",
        BASE / "cost_ledger.csv", BASE / "rs_check_rejudge.json", BASE / "pilot_results.csv", BASE / "analysis_summary.json", BASE / "reference_report.md", BASE / "references/index.json",
        asm_dir / "pilot_results.csv", asm_dir / "raw.npz", asm_dir / "metrics.json",
        rs_dir / "rs_check.json", rs_dir / "rs_check.npz",
        BASE / "runs/r1-pilot-rs-01/rs_script_executed.py",
        ROOT / "tools/arw_r1.py", ROOT / "tools/arw_r1_asm_pilot.py", ROOT / "tools/arw_r1_rscheck.py", ROOT / "tools/arw_r1_rs_rejudge.py", ROOT / "tools/arw_r1_finalize.py",
        ROOT / "arw_reference/asm.py", ROOT / "arw_reference/asm_variants.py", ROOT / "arw_reference/rs.py",
    ]
    for run_id in ("r1-pilot-asm-01", "r1-pilot-asm-01-r2", "r1-pilot-rs-01"):
        evidence_paths += [BASE / "runs" / run_id / name for name in ("config.json", "status.json", "stdout.log", "stderr.log")]
    if any(not path.is_file() for path in evidence_paths):
        raise FileNotFoundError("required G1-R1 evidence absent")
    verdict = {
        "status": "INCONCLUSIVE", "protocol_version": "G1-R1/1.1", "parent_g0": "PASS", "old_g1": "FAIL",
        "criteria": summary,
        "formal_joint_endpoint_completed": False,
        "formal_four_direction_completed": False,
        "evidence": [{"path": str(path), "sha256": sha(path)} for path in evidence_paths],
        "unresolved": ["C1 pilot direction mismatch exceeds 1%", "bandlimit removes most tangent spectrum", "C2 RS grid derivative not converged", "joint endpoint and four-direction formal checks not run"],
        "allowed_next_action": "stop G2-G5; define a separate numerical protocol and budget before any further G1 qualification",
    }
    save(BASE / "gate.json", verdict)
    print(json.dumps({"status": verdict["status"], "criteria": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
