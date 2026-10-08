"""Offline G1-R1 RS pilot re-judgment using the preregistered thresholds.

Reads the saved propagation statistics only; does not run optical propagation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError("offline re-judgment exists; never overwrite evidence")
    protocol = json.loads(protocol_path.read_text())
    saved = json.loads(input_path.read_text())
    if sha256(protocol_path) != saved["inputs"]["protocol_sha256"]:
        raise ValueError("saved run used a different pilot protocol")
    rules = protocol["preregistered_rules"]
    amp_limit = float(rules["near_zero_amplitude"])
    abs_limit = float(rules["near_zero_absolute"])
    rel_limit = float(rules["direction_relative_max"])
    steps = [float(x) for x in protocol["rs_pilot"]["steps"]]
    settings = []
    for result in saved["results"]:
        direct = float(result["direct_total_derivative"])
        rows = []
        if [float(row["h"]) for row in result["fd"]] != steps:
            raise ValueError("saved FD steps differ from preregistered steps")
        for row in result["fd"]:
            fd = float(row["fd_total_derivative"])
            absolute = abs(direct - fd)
            relative = absolute / max(abs(direct), abs(fd), 1e-30)
            near_zero = max(abs(direct), abs(fd)) < amp_limit
            passed = absolute <= abs_limit if near_zero else relative <= rel_limit
            rows.append({"h": row["h"], "direct": direct, "fd": fd,
                         "absolute_error": absolute, "relative_error": relative,
                         "rule": "near_zero_absolute" if near_zero else "relative",
                         "limit": abs_limit if near_zero else rel_limit, "pass": passed})
        image_d = float(result["direct_image_derivative"])
        power_d = float(result["direct_power_derivative"])
        cancellation = (abs(image_d) + abs(power_d)) / max(abs(direct), 1e-30)
        settings.append({"case": result["case"], "grid_n": result["grid_n"],
                         "dx_mm": result["dx_mm"], "direct_image_derivative": image_d,
                         "direct_power_derivative": power_d,
                         "direct_total_derivative": direct,
                         "component_cancellation_factor": cancellation,
                         "fd_rows": rows, "pass": all(row["pass"] for row in rows)})
    output = {"kind": "offline re-judgment; no propagation", "status": "PASS" if all(s["pass"] for s in settings) else "FAIL",
              "protocol_sha256": sha256(protocol_path), "input_sha256": sha256(input_path),
              "analysis_sha256": sha256(Path(__file__).resolve()),
              "settings": settings, "checked_steps": len(settings) * len(steps),
              "boundary": "Validates only direct derivative of the existing discrete RS operator, not grid convergence or G1 eligibility."}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "settings": len(settings),
                      "checked_steps": output["checked_steps"], "output": str(output_path)}))


if __name__ == "__main__":
    main()
