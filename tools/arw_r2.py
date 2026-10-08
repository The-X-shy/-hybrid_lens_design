"""Bounded execution and append-only cost ledger for the G1-R2 revision."""

import argparse
import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/G1-R2"
FIELDS = ["run_id", "batch", "status", "started_at", "finished_at", "wall_seconds", "cpu_process_wall_upper_seconds", "gpu_process_wall_upper_seconds", "peak_cpu_bytes", "peak_gpu_bytes", "exit_reason", "protocol_sha256", "budget_sha256", "schedule_sha256", "script_sha256", "output_path"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def stamp():
    return datetime.now(timezone.utc).isoformat()


def ledger():
    path = BASE / "cost_ledger.csv"
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append(row):
    path = BASE / "cost_ledger.csv"
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def preflight(batch, protocol, budget, schedule):
    if protocol["frozen_before_propagation"] is not True or schedule["parent_protocol_sha256"] != sha(BASE / "protocol.json"):
        raise ValueError("protocol or schedule is not frozen")
    if batch not in budget["initial_authorized_batches"]:
        if batch == "B3":
            if any(row["batch"] not in ("B1", "B2") or row["status"] != "SUCCESS" for row in ledger()) or {row["batch"] for row in ledger()} != {"B1", "B2"}:
                raise ValueError("B3 requires completed B1 and B2")
            for relative, expected in schedule["decision_evidence_sha256"].items():
                if sha(BASE / relative) != expected:
                    raise ValueError(f"B3 decision evidence changed: {relative}")
        elif batch == "B4":
            previous = {row["batch"]: row for row in ledger()}
            if set(previous) != {"B1", "B2", "B3"} or any(row["status"] != "SUCCESS" for row in previous.values()):
                raise ValueError("B4 requires completed B1-B3")
            summary_path = BASE / "runs/r2-b3-01/data/comparison_summary.json"
            if sha(summary_path) != schedule["b3_decision_sha256"] or load(summary_path)["continuation_conditions_all_met"] is not True:
                raise ValueError("B3 did not authorize C1 transfer")
        elif batch == "B4L":
            previous = {row["batch"]: row for row in ledger()}
            if set(previous) != {"B1", "B2", "B3", "B4"} or any(row["status"] != "SUCCESS" for row in previous.values()):
                raise ValueError("B4L requires completed B1-B4")
            result_path = BASE / "runs/r2-b4-01/data/c1_comparison.json"
            result = load(result_path)
            if sha(result_path) != schedule["b4_failure_sha256"] or result["all_grid_rules_met"] is not False or result["local_assessment"]["needed_if_C1_grid_fails"] is not True:
                raise ValueError("B4 did not trigger a local C1 check")
            if sha(ROOT / "tools/arw_r2_local_c1.py") != schedule["script_sha256"]:
                raise ValueError("C1 local script changed after freeze")
        else:
            raise ValueError("conditional batch has no frozen prerequisite decision")
    for path, expected in protocol["source_hashes"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"frozen source changed: {path}")
    for name, state in protocol["parent_state"].items():
        record = load(ROOT / "artifacts" / name / "gate.json")
        if record["status"] != state or any(not Path(item["path"]).is_file() or sha(item["path"]) != item["sha256"] for item in record["evidence"]):
            raise ValueError(f"parent evidence invalid: {name}")
    old = ledger()
    if any(row["batch"] == batch for row in old):
        raise ValueError(f"batch already attempted: {batch}")
    if len(old) >= budget["max_total_attempts"] or sum(not row["run_id"].endswith("-r2") for row in old) >= budget["max_research_batches"]:
        raise ValueError("attempt budget exhausted")
    reserve = budget["max_local_case_seconds"] if batch in ("B2", "B4L") else budget["max_batch_seconds"]
    if sum(float(row["cpu_process_wall_upper_seconds"]) for row in old) + reserve > budget["max_cpu_process_wall_seconds"]:
        raise ValueError("CPU budget cannot reserve this batch")
    if sum(float(row["gpu_process_wall_upper_seconds"]) for row in old) > budget["max_gpu_process_seconds"]:
        raise ValueError("GPU budget exhausted")


def main():
    parser = argparse.ArgumentParser(description="G1-R2 bounded batch runner")
    parser.add_argument("--batch", choices=("B1", "B2", "B3", "B4", "B4L"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    batch = args.batch
    protocol_path, budget_path, schedule_path = BASE / "protocol.json", BASE / "budget.json", BASE / f"{batch}_schedule.json"
    protocol, budget, schedule = load(protocol_path), load(budget_path), load(schedule_path)
    preflight(batch, protocol, budget, schedule)
    script = ROOT / {"B1": "tools/arw_r2_fine.py", "B2": "tools/arw_r2_local.py", "B3": "tools/arw_r2_boundary.py", "B4": "tools/arw_r2_c1.py", "B4L": "tools/arw_r2_local_c1.py"}[batch]
    if not script.is_file():
        raise FileNotFoundError(script)
    run_id = f"r2-{batch.lower()}-01"
    run_dir = BASE / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(run_dir)
    command = ["/home/ysl/miniconda3/envs/nami-pt22/bin/python", str(script), "--protocol", str(protocol_path), "--schedule", str(schedule_path), "--output", str(run_dir / "data")]
    if args.dry_run:
        print(json.dumps({"run_id": run_id, "command": command, "protocol_sha256": sha(protocol_path), "schedule_sha256": sha(schedule_path), "script_sha256": sha(script), "reserve_seconds": budget["max_local_case_seconds"] if batch in ("B2", "B4L") else budget["max_batch_seconds"]}, ensure_ascii=False, indent=2))
        return 0
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"command": command, "protocol_sha256": sha(protocol_path), "budget_sha256": sha(budget_path), "schedule_sha256": sha(schedule_path), "script_sha256": sha(script), "host_commit": protocol["host_commit"], "upstream_commit": protocol["upstream_commit"]}, indent=2) + "\n")
    started = stamp()
    tic = time.monotonic()
    timeout = budget["max_local_case_seconds"] if batch in ("B2", "B4L") else budget["max_batch_seconds"]
    with (run_dir / "stdout.log").open("w") as stdout, (run_dir / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = 124
    elapsed = time.monotonic() - tic
    status = "SUCCESS" if code == 0 else "INCONCLUSIVE" if code == 4 or code == 124 else "FAILED"
    reason = "completed" if code == 0 else "cost_estimate_stop" if code == 4 else "timeout" if code == 124 else f"exit_{code}"
    metrics_file = run_dir / "data/metrics.json"
    metrics = load(metrics_file) if metrics_file.is_file() else {}
    peak_cpu = metrics.get("peak_cpu_bytes", "unknown:not_instrumented")
    peak_gpu = metrics.get("peak_gpu_bytes", "unknown:not_instrumented")
    if peak_cpu == "unknown:not_instrumented" and (run_dir / "data/summary.json").is_file():
        peak_cpu = int(load(run_dir / "data/summary.json")["peak_process_rss_mib"] * 1024**2)
    if peak_gpu == "unknown:not_instrumented" and batch in ("B1", "B2", "B3", "B4", "B4L"):
        peak_gpu = 0
    finished = stamp()
    (run_dir / "status.json").write_text(json.dumps({"status": status, "exit_code": code, "reason": reason, "started_at": started, "finished_at": finished}, indent=2) + "\n")
    append({"run_id": run_id, "batch": batch, "status": status, "started_at": started, "finished_at": finished, "wall_seconds": elapsed, "cpu_process_wall_upper_seconds": elapsed, "gpu_process_wall_upper_seconds": 0.0, "peak_cpu_bytes": peak_cpu, "peak_gpu_bytes": peak_gpu, "exit_reason": reason, "protocol_sha256": sha(protocol_path), "budget_sha256": sha(budget_path), "schedule_sha256": sha(schedule_path), "script_sha256": sha(script), "output_path": str(run_dir)})
    print(json.dumps({"run_id": run_id, "status": status, "wall_seconds": elapsed, "reason": reason}))
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(error, file=sys.stderr)
        sys.exit(3)
