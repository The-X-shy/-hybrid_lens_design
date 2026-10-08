"""Append-only, bounded execution for the G1-R1 revision package."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/G1-R1"
DEFAULT_PROTOCOL = BASE / "pilot_protocol.json"
ROWS = ("run_id", "stage", "status", "started_at", "finished_at", "wall_seconds", "cpu_wall_seconds_upper_bound", "gpu_device_seconds_upper_bound", "peak_gpu_bytes", "exit_reason", "protocol_sha256", "script_sha256", "output_path")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def stamp():
    return datetime.now(timezone.utc).isoformat()


def append(row):
    path = BASE / "cost_ledger.csv"
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROWS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def prior():
    path = BASE / "cost_ledger.csv"
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def verify_parent(protocol):
    for file, expected in ((ROOT / "artifacts/G0/gate.json", protocol["old_evidence"]["g0_gate_sha256"]), (ROOT / "artifacts/G1/gate.json", protocol["old_evidence"]["g1_gate_sha256"]), (ROOT / "configs/week1/cases.json", protocol["old_evidence"]["cases_sha256"]), (ROOT / "artifacts/G1/thresholds.json", protocol["old_evidence"]["thresholds_sha256"])):
        if sha(file) != expected:
            raise ValueError(f"frozen parent changed: {file}")
    for gate in ("G0-U", "G0-I", "G0"):
        value = read_json(ROOT / "artifacts" / gate / "gate.json")
        if value["status"] != "PASS" or any(not Path(e["path"]).is_file() or sha(e["path"]) != e["sha256"] for e in value["evidence"]):
            raise ValueError(f"parent evidence invalid: {gate}")
    for path, expected in protocol["source_hashes"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"pilot source changed after freeze: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", choices=("r1-pilot-asm-01", "r1-pilot-asm-01-r2", "r1-pilot-rs-01"))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    args = parser.parse_args()
    protocol_path = Path(args.protocol)
    protocol = read_json(protocol_path)
    if not protocol["frozen_before_pilot"]:
        raise ValueError("protocol not frozen")
    verify_parent(protocol)
    old = prior()
    budget = protocol["budget"]
    ordinary_count = sum(not row["run_id"].endswith("-r2") for row in old)
    retry_count = len(old) - ordinary_count
    if ordinary_count >= budget["max_pilot_batches"] and not args.run_id.endswith("-r2"):
        raise ValueError("diagnostic batch budget exhausted")
    if retry_count >= budget["max_implementation_retry"] and args.run_id.endswith("-r2"):
        raise ValueError("implementation retry budget exhausted")
    if len(old) >= budget["max_total_batches"]:
        raise ValueError("diagnostic batch budget exhausted")
    if args.run_id.endswith("-r2") and not any(row["run_id"] == "r1-pilot-asm-01" and row["status"] == "FAILED" for row in old):
        raise ValueError("retry requires failed original ASM pilot")
    if sum(float(x["gpu_device_seconds_upper_bound"]) for x in old) + budget["max_batch_seconds"] > budget["max_gpu_device_seconds"]:
        raise ValueError("GPU reservation exceeds revision budget")
    if sum(float(x["cpu_wall_seconds_upper_bound"]) for x in old) + budget["max_batch_seconds"] > budget["max_cpu_wall_seconds"]:
        raise ValueError("CPU reservation exceeds revision budget")
    run_dir = BASE / "runs" / args.run_id
    if run_dir.exists() or any(row["run_id"] == args.run_id for row in old):
        raise FileExistsError(args.run_id)
    script = ROOT / ("tools/arw_r1_asm_pilot.py" if "asm" in args.run_id else "tools/arw_r1_rscheck.py")
    if not script.is_file():
        raise FileNotFoundError(script)
    run_dir.mkdir(parents=True)
    command = ["/home/ysl/miniconda3/envs/nami-pt22/bin/python", str(script), "--protocol", str(protocol_path), "--output", str(run_dir / "data")]
    if args.run_id == "r1-pilot-rs-01":
        command += ["--case", "BOTH", "--direction", "0"]
    (run_dir / "config.json").write_text(json.dumps({"command": command, "protocol_sha256": sha(protocol_path), "script_sha256": sha(script), "parent_g0_sha256": protocol["old_evidence"]["g0_gate_sha256"]}, indent=2) + "\n")
    started = stamp()
    tic = time.monotonic()
    status, reason, exit_code = "FAILED", "unknown", 1
    with (run_dir / "stdout.log").open("w") as stdout, (run_dir / "stderr.log").open("w") as stderr:
        try:
            completed = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}, stdout=stdout, stderr=stderr, timeout=budget["max_batch_seconds"])
            exit_code = completed.returncode
            status = "SUCCESS" if exit_code == 0 else "FAILED"
            reason = "completed" if exit_code == 0 else f"exit_{exit_code}"
        except subprocess.TimeoutExpired:
            status, reason, exit_code = "TIMEOUT", "timeout", 124
    elapsed = time.monotonic() - tic
    peak = "unknown:not_recorded"
    metrics = run_dir / "data/metrics.json"
    if metrics.is_file():
        payload = read_json(metrics)
        peak = payload.get("timing", {}).get("peak_gpu_bytes", peak)
    (run_dir / "status.json").write_text(json.dumps({"status": status, "exit_code": exit_code, "reason": reason, "started_at": started, "finished_at": stamp()}, indent=2) + "\n")
    # GPU time is bounded by whole-process time for the CUDA ASM diagnostic.
    # The NumPy RS diagnostic is CPU-only and records GPU time as zero.
    append({"run_id": args.run_id, "stage": "R1", "status": status, "started_at": started, "finished_at": stamp(), "wall_seconds": elapsed, "cpu_wall_seconds_upper_bound": elapsed, "gpu_device_seconds_upper_bound": elapsed if "asm" in args.run_id else 0.0, "peak_gpu_bytes": peak, "exit_reason": reason, "protocol_sha256": sha(protocol_path), "script_sha256": sha(script), "output_path": str(run_dir)})
    print(json.dumps({"run_id": args.run_id, "status": status, "wall_seconds": elapsed, "exit_reason": reason}))
    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(3)
