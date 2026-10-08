"""Bounded, append-only runner for the ARW-V3 week-one qualification.

This runner deliberately knows only the frozen upstream micro experiment. Other
groups must acquire an explicit implementation and protocol before execution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
RUNS = ARTIFACTS / "runs"
MANIFEST_FIELDS = [
    "run_id", "gate", "experiment", "case", "config_hash", "host_commit",
    "upstream_commit", "integration_commit", "adapter_hash", "backend_mode",
    "pair_id", "input_hash", "sample_hash", "seed", "parent_run", "status",
    "started_at", "finished_at", "output_path", "worktree_patch_hash",
    "reference_source", "reference_version",
]
LEDGER_FIELDS = [
    "run_id", "gate", "status", "preprocess_seconds", "forward_seconds",
    "gradient_seconds", "check_seconds", "recompute_seconds",
    "wall_seconds", "cpu_device_seconds", "gpu_device_seconds",
    "peak_gpu_bytes", "exit_reason",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def git_patch_hash(path: Path) -> str:
    result = subprocess.check_output(["git", "-C", str(path), "diff", "--binary"], stderr=subprocess.DEVNULL)
    return hashlib.sha256(result).hexdigest()


def append_csv(path: Path, fields: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def validate(data: dict, *, require_frozen: bool = True) -> None:
    for key in ("spec_version", "project", "upstream", "python", "host_commit", "upstream_commit", "budget", "groups"):
        if key not in data:
            raise ValueError(f"missing {key}")
    project = Path(data["project"])
    upstream = Path(data["upstream"])
    python = Path(data["python"])
    if project.resolve() != ROOT or not project.is_dir() or not upstream.is_dir() or not python.is_file():
        raise ValueError("project/upstream/python path is missing or mismatched")
    if git_head(project) != data["host_commit"] or git_head(upstream) != data["upstream_commit"]:
        raise ValueError("version changed; freeze a new manifest and requalify")
    budget = data["budget"]
    if any(float(budget[k]) <= 0 for k in ("max_gpu_device_hours", "max_cpu_wall_hours", "max_run_minutes", "max_attempts")):
        raise ValueError("budget must be positive")
    if int(budget["max_attempts"]) > 24 or float(budget["max_gpu_device_hours"]) > 8 or float(budget["max_cpu_wall_hours"]) > 4 or float(budget["max_run_minutes"]) > 30:
        raise ValueError("week-one budget exceeds SPEC")
    ids = []
    for group, entries in data["groups"].items():
        if not isinstance(entries, list):
            raise ValueError(f"group {group} is not a finite list")
        for entry in entries:
            ids.append(entry["run_id"])
            if entry.get("group") != group or entry.get("gate") not in ("G0-U", "G0-I", "G1"):
                raise ValueError("group or gate mismatch")
            if entry.get("backend_mode") not in ("upstream", "integrated", "reference"):
                raise ValueError("invalid backend_mode")
            if entry.get("max_minutes", 31) > budget["max_run_minutes"]:
                raise ValueError("run timeout exceeds budget")
            if group == "W0-U":
                for name in ("optics_config", "optimization_config", "seed", "case"):
                    if name not in entry:
                        raise ValueError(f"missing {name}")
                if entry["backend_mode"] != "upstream":
                    raise ValueError("W0-U must use upstream")
                for name in ("optics_config", "optimization_config"):
                    path = ROOT / entry[name]
                    if not path.is_file() or "TODO" in path.read_text():
                        raise ValueError(f"missing or placeholder {name}")
    if len(ids) != len(set(ids)) or len(ids) > budget["max_attempts"]:
        raise ValueError("duplicate run_id or attempt budget exceeded")
    if require_frozen:
        if any(group not in ("W0-U", "W0-I", "W1-REF", "W1-CHECK") and entries for group, entries in data["groups"].items()):
            raise ValueError("later groups require frozen protocols and implemented adapters")
        if data["groups"].get("W0-I"):
            protocol = ARTIFACTS / "G0-I" / "integration_protocol.json"
            if not protocol.is_file() or digest(protocol) != data.get("integration_protocol_sha256"):
                raise ValueError("integration protocol absent or changed")
            if load_json(protocol).get("frozen_before_integrated_comparison") is not True:
                raise ValueError("integration protocol not frozen")
            for entry in data["groups"]["W0-I"]:
                if entry.get("variant") != "integration_pair" or not entry.get("pair_id"):
                    raise ValueError("integration pair metadata missing")
                if entry["backend_mode"] == "integrated" and not entry.get("sample_source_run"):
                    raise ValueError("integrated backend requires upstream sample run")
        if data["groups"].get("W1-REF") or data["groups"].get("W1-CHECK"):
            for path, key in ((ARTIFACTS / "G1/reference_protocol.json", "reference_protocol_sha256"), (ARTIFACTS / "G1/thresholds.json", "thresholds_sha256")):
                if not path.is_file() or digest(path) != data.get(key):
                    raise ValueError(f"reference protocol changed or absent: {path}")
            for group in ("W1-REF", "W1-CHECK"):
                for entry in data["groups"].get(group, []):
                    ordinary = entry.get("variant") == "reference" and entry.get("mode") == ("ref" if group == "W1-REF" else "check")
                    targeted = group == "W1-REF" and entry.get("variant") == "reference_targeted" and entry.get("mode") in ("targeted_window", "targeted_grid")
                    if not (ordinary or targeted) or entry["backend_mode"] != "reference":
                        raise ValueError("reference entry metadata missing")
                    if entry.get("protocol_file"):
                        extension = ROOT / entry["protocol_file"]
                        expected_hash = data.get("reference_extension_02_sha256") if targeted else data.get("reference_extension_sha256")
                        if not extension.is_file() or digest(extension) != expected_hash:
                            raise ValueError("reference extension absent or changed")


def current_cost() -> tuple[int, float, float]:
    path = ARTIFACTS / "cost_ledger.csv"
    if not path.exists():
        return 0, 0.0, 0.0
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows), sum(float(r["gpu_device_seconds"]) for r in rows), sum(float(r["cpu_device_seconds"]) for r in rows)


def run_upstream(entry: dict, data: dict) -> int:
    run_id = entry["run_id"]
    output = RUNS / run_id
    if output.exists():
        raise FileExistsError(f"run_id already exists: {run_id}")
    attempts, gpu_seconds, cpu_seconds = current_cost()
    budget = data["budget"]
    limit_seconds = 60 * float(entry["max_minutes"])
    if attempts >= int(budget["max_attempts"]) or gpu_seconds + limit_seconds > 3600 * float(budget["max_gpu_device_hours"]) or cpu_seconds + limit_seconds > 3600 * float(budget["max_cpu_wall_hours"]):
        raise ValueError("insufficient shared week-one budget")
    output.mkdir(parents=True)
    optics = ROOT / entry["optics_config"]
    optimization = ROOT / entry["optimization_config"]
    config_hash = hashlib.sha256(optics.read_bytes() + optimization.read_bytes()).hexdigest()
    save_json(output / "config.json", {"entry": entry, "optics_sha256": digest(optics), "optimization_sha256": digest(optimization)})
    variant = entry.get("variant", "fig6")
    environment = os.environ.copy()
    if variant == "reference_targeted":
        if load_json(ARTIFACTS / "G0" / "gate.json")["status"] != "PASS":
            raise ValueError("G0 has not passed")
        command = [data["python"], str(ROOT / "tools/arw_reference_targeted.py"), "--case", entry["case"], "--protocol", str(ROOT / entry["protocol_file"]), "--output", str(output)]
        environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    elif variant == "reference":
        if load_json(ARTIFACTS / "G0" / "gate.json")["status"] != "PASS":
            raise ValueError("G0 has not passed")
        command = [data["python"], str(ROOT / "tools/arw_reference_run.py"), "--mode", entry["mode"], "--case", entry["case"], "--output", str(output)]
        if entry.get("protocol_file"):
            command += ["--protocol", str(ROOT / entry["protocol_file"])]
        environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    elif variant == "integration_pair":
        if load_json(ARTIFACTS / "G0-U" / "gate.json")["status"] != "PASS":
            raise ValueError("G0-U has not passed")
        backend = entry["backend_mode"]
        command = [data["python"], str(ROOT / "tools/arw_pair.py"), "--backend", backend, "--config", str(optics), "--output", str(output)]
        environment["PYTHONPATH"] = (data["upstream"] if backend == "upstream" else str(ROOT / "third_party/deeplens-core") + os.pathsep + str(ROOT)) + os.pathsep + environment.get("PYTHONPATH", "")
        if backend == "integrated":
            sample_path = RUNS / entry["sample_source_run"] / "samples.npz"
            if not sample_path.is_file():
                raise FileNotFoundError(f"paired upstream samples absent: {sample_path}")
            command += ["--sample-file", str(sample_path)]
    elif variant == "native_fig5":
        command = [data["python"], str(ROOT / "tools/arw_upstream_native.py"), "--config", str(optics), "--output", str(output)]
        environment["PYTHONPATH"] = data["upstream"] + os.pathsep + environment.get("PYTHONPATH", "")
    else:
        # The author code remains untouched. Only a copied optimization config has its output path changed.
        import yaml
        opt = yaml.safe_load(optimization.read_text())
        opt["results_dir"] = str(output / "author_output")
        local_opt = output / "optimization.yaml"
        local_opt.write_text(yaml.safe_dump(opt, sort_keys=False))
        command = [data["python"], "-m", "fig6_hybrid_system.optim", "--optical-config", str(optics), "--optimization-config", str(local_opt), "--record-runtime"]
    start = now()
    tic = time.monotonic()
    status = "FAILED"
    reason = "unknown"
    code = 1
    with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
        try:
            completed = subprocess.run(command, cwd=data["upstream"], env=environment, stdout=stdout, stderr=stderr, timeout=limit_seconds, check=False)
            code = completed.returncode
            status = "SUCCESS" if code == 0 else "FAILED"
            reason = "completed" if code == 0 else f"exit_{code}"
        except subprocess.TimeoutExpired:
            status, reason, code = "TIMEOUT", "timeout", 124
        except Exception as exc:
            status, reason, code = "FAILED", type(exc).__name__, 1
            stderr.write(f"\nrunner error: {exc}\n")
    elapsed = time.monotonic() - tic
    metrics = {"status": status, "exit_code": code, "wall_seconds": elapsed, "loss": None, "peak_gpu_bytes": None}
    if status == "SUCCESS":
        if variant in ("reference", "reference_targeted"):
            ref = load_json(output / "ref_metrics.json")
            metrics.update(ref)
            metrics["loss"] = []
            metrics["reference_source"] = "arw_reference.asm + arw_reference.rs"
            metrics["reference_version"] = digest(ROOT / "arw_reference/asm.py") + ":" + digest(ROOT / "arw_reference/rs.py")
        elif variant == "integration_pair":
            pair = load_json(output / "pair_metrics.json")
            metrics.update(pair)
            metrics["loss"] = [pair["loss"]]
        elif variant == "native_fig5":
            native = load_json(output / "native_metrics.json")
            metrics.update(native)
            metrics["loss"] = [native["loss"]]
        else:
            import numpy as np
            author_output = output / "author_output"
            losses = np.load(author_output / "history" / "loss.npy")
            metrics["loss"] = [float(x) for x in losses]
            np.savez_compressed(output / "raw.npz", loss=losses, phase=np.load(author_output / "phase" / f"phase_{len(losses)-1:04d}.npy"))
            runtime_file = author_output / "runtime_metrics.json"
            if runtime_file.exists():
                runtime = load_json(runtime_file)
                metrics["author_runtime"] = runtime
                metrics["peak_gpu_bytes"] = runtime.get("peak_gpu_memory_allocated_bytes")
    save_json(output / "metrics.json", metrics)
    save_json(output / "status.json", {"status": status, "reason": reason, "started_at": start, "finished_at": now()})
    append_csv(ARTIFACTS / "run_manifest.csv", MANIFEST_FIELDS, {
        "run_id": run_id, "gate": entry["gate"], "experiment": "independent_reference_" + entry["mode"] if variant in ("reference", "reference_targeted") else "integration_pair" if variant == "integration_pair" else "author_fig5_native_micro" if variant == "native_fig5" else "author_fig6_micro", "case": entry["case"],
        "config_hash": config_hash, "host_commit": data["host_commit"], "upstream_commit": data["upstream_commit"],
        "integration_commit": "null:uncommitted_adapter" if variant == "integration_pair" else "null:independent_reference" if variant in ("reference", "reference_targeted") else "null:no_adapter_yet", "adapter_hash": hashlib.sha256((ROOT / "arw_integration/doe_plane.py").read_bytes() + (ROOT / "arw_integration/raywave.py").read_bytes()).hexdigest() if variant == "integration_pair" else "null:reference_implementation" if variant in ("reference", "reference_targeted") else "null:no_adapter_yet", "backend_mode": entry["backend_mode"],
        "pair_id": entry.get("pair_id", "null:upstream_repeat"), "input_hash": digest(optics),
        "sample_hash": digest(output / "samples.npz") if variant == "integration_pair" and (output / "samples.npz").is_file() else "null:deterministic_reference" if variant in ("reference", "reference_targeted") else "null:author_native_sampling", "seed": entry["seed"], "parent_run": entry.get("sample_source_run") or "null:first_attempt",
        "status": status, "started_at": start, "finished_at": now(), "output_path": str(output),
        "worktree_patch_hash": git_patch_hash(ROOT),
        "reference_source": "arw_reference.asm + arw_reference.rs" if variant in ("reference", "reference_targeted") else "null:not_reference",
        "reference_version": digest(ROOT / "arw_reference/asm.py") + ":" + digest(ROOT / "arw_reference/rs.py") if variant in ("reference", "reference_targeted") else "null:not_reference",
    })
    append_csv(ARTIFACTS / "cost_ledger.csv", LEDGER_FIELDS, {
        "run_id": run_id, "gate": entry["gate"], "status": status, "preprocess_seconds": "null:not_instrumented",
        "forward_seconds": "null:not_instrumented", "gradient_seconds": "null:not_instrumented",
        "check_seconds": "null:not_instrumented", "recompute_seconds": "null:not_instrumented",
        "wall_seconds": elapsed, "cpu_device_seconds": elapsed,
        "gpu_device_seconds": 0.0 if variant in ("reference", "reference_targeted") else elapsed,
        "peak_gpu_bytes": metrics["peak_gpu_bytes"] if metrics["peak_gpu_bytes"] is not None else "null:not_reported",
        "exit_reason": reason,
    })
    print(f"{run_id}: {status}; wall={elapsed:.1f}s; loss={metrics['loss']}")
    return code


def gate(gate_id: str) -> int:
    target = ARTIFACTS / gate_id / "gate.json"
    if target.exists():
        print(target.read_text())
        return 0 if load_json(target)["status"] == "PASS" else 2
    if gate_id == "G0-U":
        required = [ARTIFACTS / "G0-U" / n for n in ("inputs.json", "versions.json", "environment.txt", "entrypoints.md", "budget.json", "scope_lock.json", "native_protocol.json", "protocol_revision_02.json", "protocol_revision_03.json", "upstream_report.md")]
        ids = ("w0u-03", "w0u-04", "w0u-07", "w0u-08")
        files = [RUNS / run_id / name for run_id in ids for name in ("config.json", "metrics.json", "raw.npz", "stdout.log", "stderr.log", "status.json")]
        if any(not p.is_file() for p in required + files):
            status = "INCONCLUSIVE"
            details = {"reason": "required input, run or report evidence missing"}
        else:
            import numpy as np
            versions = load_json(ARTIFACTS / "G0-U" / "versions.json")
            upstream = Path(load_json(ROOT / "configs/week1/manifest_v4.json")["upstream"])
            upstream_dirty = subprocess.check_output(["git", "-c", "core.filemode=false", "-C", str(upstream), "status", "--short"], text=True).strip()
            m = {run_id: load_json(RUNS / run_id / "metrics.json") for run_id in ids}
            same_configs = all(digest(RUNS / left / "config.json") != "" and load_json(RUNS / left / "config.json")["optics_sha256"] == load_json(RUNS / right / "config.json")["optics_sha256"] and load_json(RUNS / left / "config.json")["optimization_sha256"] == load_json(RUNS / right / "config.json")["optimization_sha256"] for left, right in (("w0u-03", "w0u-04"), ("w0u-07", "w0u-08")))
            fig6_delta = abs(m["w0u-03"]["loss"][-1] - m["w0u-04"]["loss"][-1])
            a = np.load(RUNS / "w0u-07" / "raw.npz")
            b = np.load(RUNS / "w0u-08" / "raw.npz")
            field_delta = float(np.linalg.norm(a["field"] - b["field"]) / max(float(np.linalg.norm(a["field"])), 1e-30))
            gradient_delta = float(np.linalg.norm(a["gradient"] - b["gradient"]) / max(float(np.linalg.norm(a["gradient"])), 1e-30))
            dv_a, dv_b = m["w0u-07"]["directional_derivative"], m["w0u-08"]["directional_derivative"]
            dv_delta = abs(dv_a - dv_b)
            checks = {
                "source_clean": not upstream_dirty,
                "version_locked": versions["upstream_commit"] == git_head(upstream) and versions["host_commit"] == git_head(ROOT),
                "all_runs_succeeded": all(m[x]["status"] == "SUCCESS" for x in ids),
                "same_inputs": same_configs,
                "fig6_loss_repeat": fig6_delta <= 1e-6 * max(abs(m["w0u-03"]["loss"][-1]), 1e-30),
                "native_field_repeat": field_delta <= 1e-6,
                "native_gradient_repeat": gradient_delta <= 1e-5,
                "native_direction_repeat": dv_delta <= max(1e-7, 1e-5 * abs(dv_a)),
                "nonzero_direction": abs(dv_a) > 1e-7,
                "finite_raw": all(np.isfinite(a[k]).all() and np.isfinite(b[k]).all() for k in ("field", "intensity", "gradient")),
                "positive_power": m["w0u-07"]["raw_power"] > 0 and m["w0u-08"]["raw_power"] > 0,
            }
            status = "PASS" if all(checks.values()) else "FAIL"
            details = {"checks": checks, "fig6_loss_difference": fig6_delta, "native_field_relative_l2": field_delta, "native_gradient_relative_l2": gradient_delta, "native_direction_difference": dv_delta, "scope": "bounded author micro; full paper reproduction remains pending"}
    elif gate_id == "G0-I":
        upstream_gate = ARTIFACTS / "G0-U" / "gate.json"
        required = [ARTIFACTS / "G0-I" / n for n in ("integration_map.md", "integration_protocol.json", "sampling_definitions.json", "upstream_vs_integrated.csv", "integration_gradcheck.csv", "compatibility_report.md", "lens_action_check.json", "repair_01.json")]
        pair_ids = ("w0i-free-up", "w0i-free-host-02", "w0i-lens-up", "w0i-lens-host-02")
        files = [RUNS / run_id / name for run_id in pair_ids for name in ("config.json", "metrics.json", "raw.npz", "samples.npz", "stdout.log", "stderr.log", "status.json")]
        if not upstream_gate.is_file() or any(not p.is_file() for p in required + files):
            status, details = "INCONCLUSIVE", {"reason": "G0-U or required integration evidence missing"}
        else:
            with (ARTIFACTS / "G0-I" / "upstream_vs_integrated.csv").open(newline="") as handle:
                forward = list(csv.DictReader(handle))
            with (ARTIFACTS / "G0-I" / "integration_gradcheck.csv").open(newline="") as handle:
                gradient = list(csv.DictReader(handle))
            with (ARTIFACTS / "run_manifest.csv").open(newline="") as handle:
                manifest_rows = {row["run_id"]: row for row in csv.DictReader(handle)}
            lens = load_json(RUNS / "w0i-lens-host-02" / "pair_metrics.json")
            protocol = ARTIFACTS / "G0-I" / "integration_protocol.json"
            manifest = load_json(ROOT / "configs/week1/manifest_g0i.json")
            checks = {
                "parent_pass": load_json(upstream_gate)["status"] == "PASS",
                "parent_evidence_intact": all(Path(item["path"]).is_file() and digest(Path(item["path"])) == item["sha256"] for item in load_json(upstream_gate)["evidence"]),
                "protocol_frozen": digest(protocol) == manifest["integration_protocol_sha256"],
                "all_pair_runs_succeeded": all(load_json(RUNS / run_id / "metrics.json")["status"] == "SUCCESS" for run_id in pair_ids),
                "forward_rows_pass": len(forward) == 8 and all(row["pass"] == "True" for row in forward),
                "direction_rows_pass": len(gradient) == 2 and all(row["pass"] == "True" for row in gradient),
                "samples_paired": all(manifest_rows[u]["sample_hash"] == manifest_rows[h]["sample_hash"] != "" for u, h in (("w0i-free-up", "w0i-free-host-02"), ("w0i-lens-up", "w0i-lens-host-02"))),
                "inputs_paired": all(manifest_rows[u]["input_hash"] == manifest_rows[h]["input_hash"] for u, h in (("w0i-free-up", "w0i-free-host-02"), ("w0i-lens-up", "w0i-lens-host-02"))),
                "host_geometry_loaded": str(ROOT / "third_party/deeplens-core") in lens["geometry_module"],
                "lens_has_optical_action": lens.get("lens_direction_change", 0) > 1e-8,
                "host_no_doe_regression": lens.get("host_no_doe_regression_max_difference", 1) <= 1e-12,
            }
            status = "PASS" if all(checks.values()) else "FAIL"
            details = {"checks": checks, "lens_direction_change": lens.get("lens_direction_change"), "host_no_doe_regression_max_difference": lens.get("host_no_doe_regression_max_difference")}
    elif gate_id == "G0":
        required = [ARTIFACTS / child / "gate.json" for child in ("G0-U", "G0-I")]
        files = []
        if any(not p.is_file() for p in required):
            status, details = "BLOCKED", {"reason": "subgate verdict missing"}
        else:
            children = [load_json(p) for p in required]
            intact = all(all(Path(item["path"]).is_file() and digest(Path(item["path"])) == item["sha256"] for item in child["evidence"]) for child in children)
            status = "PASS" if all(child["status"] == "PASS" for child in children) and intact else "FAIL"
            details = {"subgate_statuses": [child["status"] for child in children], "evidence_intact": intact}
    elif gate_id == "G1":
        required = [ARTIFACTS / "G1" / name for name in (
            "reference_protocol.json", "thresholds.json", "reference_protocol_extension_01.json",
            "reference_protocol_extension_02.json", "reference_check_protocol_c1.json",
            "reference_check_protocol_c2.json", "convergence.csv", "invariants.csv",
            "directional_checks.csv", "analysis_summary.json", "reference_report.md",
            "references/index.json", "references/C1_candidate_only.npz", "references/C2_candidate_only.npz",
        )]
        run_ids = ("g1-ref-c1", "g1-ref-c2", "g1-ref-c1-ext01", "g1-ref-c2-ext01", "g1-ref-c1-ext02", "g1-ref-c1-ext02-r2", "g1-ref-c2-ext02", "g1-check-c1", "g1-check-c2")
        files = [RUNS / run_id / name for run_id in run_ids for name in ("config.json", "metrics.json", "raw.npz", "stdout.log", "stderr.log", "status.json")]
        parent = ARTIFACTS / "G0" / "gate.json"
        if not parent.is_file() or any(not path.is_file() for path in required + files):
            status, details = "INCONCLUSIVE", {"reason": "G0 or required G1 evidence missing"}
        else:
            with (ARTIFACTS / "G1" / "convergence.csv").open(newline="") as handle:
                forward = list(csv.DictReader(handle))
            with (ARTIFACTS / "G1" / "directional_checks.csv").open(newline="") as handle:
                directional = list(csv.DictReader(handle))
            summary = load_json(ARTIFACTS / "G1" / "analysis_summary.json")
            manifest = load_json(ROOT / "configs/week1/manifest_g1.json")
            checks = {
                "parent_pass": load_json(parent)["status"] == "PASS",
                "parent_evidence_intact": all(Path(item["path"]).is_file() and digest(Path(item["path"])) == item["sha256"] for item in load_json(parent)["evidence"]),
                "protocol_unchanged": digest(ARTIFACTS / "G1/reference_protocol.json") == manifest["reference_protocol_sha256"] and digest(ARTIFACTS / "G1/thresholds.json") == manifest["thresholds_sha256"],
                "all_final_runs_succeeded": all(load_json(RUNS / run_id / "metrics.json")["status"] == "SUCCESS" for run_id in run_ids if run_id != "g1-ref-c1-ext02"),
                "forward_within_threshold": len(forward) == 18 and all(row["pass"] == "True" for row in forward),
                "directional_within_threshold": len(directional) == 18 and all(row["pass"] == "True" for row in directional),
                "finite_difference_within_threshold": all(summary["fd_summary"][case]["median_all_steps"] <= .01 and summary["fd_summary"][case]["p95_all_steps"] <= .05 for case in ("C1", "C2")),
            }
            status = "PASS" if all(checks.values()) else "FAIL"
            details = {"checks": checks, "forward_pass": summary["forward_pass"], "forward_total": summary["forward_total"], "direction_pass": summary["direction_pass"], "direction_total": summary["direction_total"], "failed_forward": [{"case": r["case"], "kind": r["kind"], "method": r["method"], "setting_a": r["setting_a"], "setting_b": r["setting_b"], "l1": r["normalized_intensity_l1"]} for r in forward if r["pass"] != "True"], "failed_directional_count": sum(r["pass"] != "True" for r in directional)}
    else:
        status = "BLOCKED"
        details = {"reason": "prerequisite protocol and experimental evidence absent"}
    evidence = [{"path": str(p), "sha256": digest(p)} for p in required + files] if gate_id in ("G0-U", "G0-I", "G0", "G1") and status not in ("INCONCLUSIVE", "BLOCKED") else []
    next_action = {"G0-U": "W0-I protocol freeze", "G0-I": "G0 summary", "G0": "G1 reference qualification", "G1": "G2 protocol freeze"}.get(gate_id, "resolve prerequisite")
    report = {"status": status, "spec_version": "1.1", "criteria": details, "evidence": evidence, "unresolved": [] if status == "PASS" else [details.get("reason", "one or more criteria failed")], "allowed_next_action": next_action if status == "PASS" else "resolve prerequisite"}
    save_json(target, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


def summarize_week1() -> int:
    target = ARTIFACTS / "W1" / "gate.json"
    if target.exists():
        print(json.dumps(load_json(target), ensure_ascii=False, indent=2))
        return 0
    names = ("G0-U", "G0-I", "G0", "G1")
    paths = [ARTIFACTS / name / "gate.json" for name in names]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("week-one gate verdict missing")
    gates = {name: load_json(path) for name, path in zip(names, paths)}
    intact = {
        name: all(Path(item["path"]).is_file() and digest(Path(item["path"])) == item["sha256"] for item in value["evidence"])
        for name, value in gates.items()
    }
    with (ARTIFACTS / "run_manifest.csv").open(newline="") as handle:
        runs = list(csv.DictReader(handle))
    with (ARTIFACTS / "cost_ledger.csv").open(newline="") as handle:
        costs = list(csv.DictReader(handle))
    if len(runs) != len(costs) or {row["run_id"] for row in runs} != {row["run_id"] for row in costs}:
        raise ValueError("run manifest and cost ledger differ")
    total_wall = sum(float(row["wall_seconds"]) for row in costs)
    gpu_seconds = sum(float(row["gpu_device_seconds"]) for row in costs)
    cpu_seconds = sum(float(row["cpu_device_seconds"]) for row in costs)
    summary = load_json(ARTIFACTS / "G1" / "analysis_summary.json")
    verdict = "PASS" if all(gates[name]["status"] == "PASS" and intact[name] for name in names) else "FAIL"
    report = (
        "# ARW-V3 第一周实验总结\n\n"
        f"**结论：{verdict}。** 作者代码微型复测、两条集成路径及 G0 汇总通过；独立数值参考 G1 未达到冻结门槛。按方案停止 G2–G5，未启动可选 G2 pilot。\n\n"
        "## 各项判定\n\n"
        + "\n".join(f"- {name}：{gates[name]['status']}；证据指纹{'完整' if intact[name] else '不完整'}。" for name in names)
        + "\n\n## 未通过的检查\n\n"
        f"- G1 前向收敛和交叉检查：{summary['forward_pass']}/{summary['forward_total']} 项通过；C2 窗口变化 0.00626，门槛为 0.003。\n"
        f"- G1 方向检查：{summary['direction_pass']}/{summary['direction_total']} 项通过；C1/C2 末档窗口变化分别约 4.84% 和 2.84%，门槛为 1%。\n"
        "- 所有原始结果、失败尝试及两轮加密记录均保留；详情见 `../G1/reference_report.md`。\n\n"
        "## 用量与范围\n\n"
        f"- 尝试 {len(runs)}/24 次，其中失败 {sum(row['status'] != 'SUCCESS' for row in runs)} 次；每次运行均小于 30 分钟。\n"
        f"- 实测运行墙钟合计 {total_wall:.2f} 秒；GPU 设备占用上界 {gpu_seconds:.2f} 秒（{gpu_seconds/3600:.4f}/8 小时）；CPU 墙钟计费上界 {cpu_seconds:.2f} 秒（{cpu_seconds/3600:.4f}/4 小时）。进程阶段细分未被计时，因此账本明确记为未测。\n"
        "- GPU 上界按使用 GPU 的整个子进程墙钟估计；CPU 参考运行的 GPU 用量为零。旧账本保留为 `.original`，更正记录见现行 `cost_ledger.csv`。\n"
        "- G0-U 是作者代码的微型复测，不构成论文完整复现；G0-I 只证明所测平面配置在两端一致。\n\n"
        "## 后续条件\n\n"
        "G1 的两轮修复复测已用完。继续研究需要另立协议、预算并重新取得独立参考资格；当前候选参考数据不得用于后续方法结论。\n"
    )
    report_path = ARTIFACTS / "W1" / "week1_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        if report_path.read_text() != report:
            raise FileExistsError(f"week-one report exists with different content: {report_path}")
    else:
        report_path.write_text(report)
    evidence_paths = paths + [ARTIFACTS / name for name in ("run_manifest.csv", "run_manifest.csv.original", "cost_ledger.csv", "cost_ledger.csv.original", "G1/reference_report.md", "G1/reference_protocol_extension_01.json", "G1/repair_02.json", "W1/week1_report.md")]
    result = {
        "status": verdict, "spec_version": "1.1", "criteria": {
            "subgates": {name: gates[name]["status"] for name in names},
            "evidence_intact": intact, "attempts": len(runs), "failed_attempts": sum(row["status"] != "SUCCESS" for row in runs),
            "wall_seconds": total_wall, "gpu_device_seconds_upper_bound": gpu_seconds,
            "cpu_wall_seconds_upper_bound": cpu_seconds,
        },
        "evidence": [{"path": str(path), "sha256": digest(path)} for path in evidence_paths],
        "unresolved": [] if verdict == "PASS" else ["G1 reference qualification failed its frozen numerical thresholds"],
        "allowed_next_action": "G2 protocol freeze" if verdict == "PASS" else "stop downstream; new G1 qualification requires a new protocol and budget",
    }
    save_json(target, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="ARW-V3 bounded week-one experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--manifest", required=True)
    r = sub.add_parser("run")
    r.add_argument("--manifest", required=True)
    r.add_argument("--group", required=True)
    r.add_argument("--dry-run", action="store_true")
    g = sub.add_parser("gate")
    g.add_argument("--id", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--week", type=int, required=True)
    args = parser.parse_args()
    if args.command in ("validate", "run"):
        data = load_json(ROOT / args.manifest)
        validate(data)
        if args.command == "validate":
            print("manifest valid")
            return 0
        if args.group not in ("W0-U", "W0-I", "W1-REF", "W1-CHECK"):
            raise ValueError("group has no frozen executable protocol")
        if args.group == "W0-I" and load_json(ARTIFACTS / "G0-U" / "gate.json")["status"] != "PASS":
            raise ValueError("G0-U has not passed")
        if args.group in ("W1-REF", "W1-CHECK") and load_json(ARTIFACTS / "G0" / "gate.json")["status"] != "PASS":
            raise ValueError("G0 has not passed")
        if args.group == "W1-CHECK" and any(not (RUNS / f"g1-ref-{case}" / "series.json").is_file() for case in ("c1", "c2")):
            raise ValueError("G1 reference runs incomplete")
        entries = data["groups"].get(args.group, [])
        if args.dry_run:
            print(json.dumps({"group": args.group, "runs": entries, "max_reserved_seconds": sum(60 * e["max_minutes"] for e in entries)}, ensure_ascii=False, indent=2))
            return 0
        for entry in entries:
            code = run_upstream(entry, data)
            if code:
                return code
        return 0
    if args.command == "gate":
        return gate(args.id)
    if args.command == "summarize":
        if args.week != 1:
            raise ValueError("only week 1 is defined")
        return summarize_week1()
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        print(error, file=sys.stderr)
        sys.exit(3)
