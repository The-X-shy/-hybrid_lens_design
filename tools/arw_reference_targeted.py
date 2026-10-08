"""Second, targeted precision extension after the frozen G1 convergence scan."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from arw_reference.asm import propagate as asm
from arw_reference.rs import propagate_rs1
from arw_reference.common import incident_field
from tools.arw_reference_run import numerical_field, rs_metrics


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("C1", "C2"), required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    case = next(c for c in json.loads((ROOT / "configs/week1/cases.json").read_text())["cases"] if c["id"] == args.case)
    phase = np.load(ROOT / case["initial_phase_file"]).astype(np.float64)
    target = np.load(ROOT / case["target_file"]).astype(np.float64)
    levels = protocol["C1_window"] if args.case == "C1" else protocol["C2_grid"]
    results = []
    raw = {}
    for n in levels["samples"]:
        dx = levels["dx_mm"] if args.case == "C1" else levels["window_mm"] / n
        label = f"window{n}" if args.case == "C1" else f"grid{n}"
        e0 = numerical_field(phase, n, dx, case["doe_pitch_mm"], case["aperture_radius_mm"])
        input_power = float(np.square(np.abs(e0)).sum() * dx * dx)
        field_asm = asm(torch.as_tensor(e0, device="cuda", dtype=torch.complex128), dx, case["wavelength_mm"], case["propagation_mm"]).detach().cpu().numpy()
        measured = rs_metrics(field_asm, dx, target, input_power)
        results.append({"case": args.case, "method": "ASM", "setting": label, "grid_n": n, "dx_mm": dx, "window_mm": n * dx, "input_power": input_power, "window_power": measured["power"], "loss": measured["loss"], "centroid_mm": measured["centroid"], "finite": bool(np.isfinite(field_asm).all())})
        raw[f"ASM_{label}_normalized"] = measured["normalized"]
        raw[f"ASM_{label}_image"] = measured["image"]
        if args.case == "C2":
            # RS is a zero-padded linear convolution. The central detector needs
            # only the source support plus detector support at the same dx.
            sensor_width = 64 * round(.0063 / dx)
            crop_n = 2 * sensor_width
            start = (n - crop_n) // 2
            cropped = e0[start:start + crop_n, start:start + crop_n]
            field_rs = propagate_rs1(cropped, dx, case["wavelength_mm"], case["propagation_mm"])
            measured_rs = rs_metrics(field_rs, dx, target, input_power)
            results.append({"case": args.case, "method": "RS", "setting": label, "grid_n": n, "dx_mm": dx, "window_mm": n * dx, "input_power": input_power, "window_power": measured_rs["power"], "loss": measured_rs["loss"], "centroid_mm": measured_rs["centroid"], "finite": bool(np.isfinite(field_rs).all()), "rs_source_crop_n": crop_n})
            raw[f"RS_{label}_normalized"] = measured_rs["normalized"]
            raw[f"RS_{label}_image"] = measured_rs["image"]
    out = Path(args.output)
    (out / "series.json").write_text(json.dumps(results, indent=2) + "\n")
    np.savez_compressed(out / "raw.npz", **raw)
    (out / "ref_metrics.json").write_text(json.dumps({"status": "SUCCESS", "case": args.case, "settings": len(results), "finite": all(x["finite"] for x in results)}, indent=2) + "\n")
    print(json.dumps({"case": args.case, "settings": len(results)}))


if __name__ == "__main__":
    main()
