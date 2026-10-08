"""Bounded G1-R1 C1 window and transfer sampling diagnostic."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from arw_reference.asm import propagate as original_asm
from arw_reference.asm_variants import bandlimited_propagate
from arw_reference.rs import propagate_rs1
from tools.arw_reference_run import make_directions, rs_metrics


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_source(phase, direction, n, dx, pitch, radius):
    axis = (np.arange(n) - (n - 1) / 2) * dx
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    ix = np.floor(xx / pitch + 32).astype(int)
    iy = np.floor(yy / pitch + 32).astype(int)
    mask = (xx * xx + yy * yy < radius * radius) & (ix >= 0) & (ix < 64) & (iy >= 0) & (iy < 64)
    p = phase[np.clip(iy, 0, 63), np.clip(ix, 0, 63)]
    v = direction[np.clip(iy, 0, 63), np.clip(ix, 0, 63)]
    field = mask * np.exp(1j * p)
    tangent = 1j * v * field
    return field, tangent


def measured(field, tangent, dx, target, input_power):
    value = rs_metrics(field, dx, target, input_power)
    q = round(.0063 / dx)
    width = 64 * q
    start = (field.shape[0] - width) // 2
    if start < 0 or field.shape[0] - width != 2 * start:
        raise ValueError("sensor indices not centered")
    local = field[start:start + width, start:start + width]
    local_d = tangent[start:start + width, start:start + width]
    jdot = (2 * np.real(local.conj() * local_d)).reshape(64, q, 64, q).mean(axis=(1, 3))
    image = value["image"]
    norm = float(image.sum())
    qdot = jdot / norm - image * (float(jdot.sum()) / norm**2)
    image_term = float(((value["normalized"] - target) * qdot).sum())
    pdot = float(jdot.sum() * .0063**2)
    power_term = float(-.2 * (1 - value["power"] / input_power) * pdot / input_power)
    return value, {"image_term": image_term, "power_term": power_term, "total": image_term + power_term, "sensor_power_dot": pdot}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    case = next(c for c in json.loads((ROOT / "configs/week1/cases.json").read_text())["cases"] if c["id"] == "C1")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    phase_path = ROOT / case["initial_phase_file"]
    target_path = ROOT / case["target_file"]
    phase = np.load(phase_path).astype(np.float64)
    target = np.load(target_path).astype(np.float64)
    direction = make_directions("C1", 4)[0]
    config = protocol["asm_pilot"]
    sizes = config["window_samples"]
    dx = config["dx_mm"]
    smallest = min(sizes)
    source, tangent = physical_source(phase, direction, smallest, dx, case["doe_pitch_mm"], case["aperture_radius_mm"])
    input_power = float(np.square(np.abs(source)).sum() * dx**2)
    rows = []
    raw = {}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    for n in sizes:
        if (n - smallest) % 2:
            raise ValueError("window parity differs")
        pad = (n - smallest) // 2
        e = np.pad(source, pad)
        edot = np.pad(tangent, pad)
        tensor = torch.as_tensor(e, device="cuda", dtype=torch.complex128)
        tangent_tensor = torch.as_tensor(edot, device="cuda", dtype=torch.complex128)
        for variant in ("original", "bandlimited"):
            if variant == "original":
                f = original_asm(tensor, dx, case["wavelength_mm"], case["propagation_mm"])
                fd = original_asm(tangent_tensor, dx, case["wavelength_mm"], case["propagation_mm"])
                extra = {"cutoff_cycles_per_mm": None, "discarded_spectral_energy_fraction": 0.0, "discarded_tangent_spectral_energy_fraction": 0.0}
            else:
                f, extra = bandlimited_propagate(tensor, dx, case["wavelength_mm"], case["propagation_mm"])
                fd, tangent_extra = bandlimited_propagate(tangent_tensor, dx, case["wavelength_mm"], case["propagation_mm"])
                extra["discarded_tangent_spectral_energy_fraction"] = tangent_extra["discarded_spectral_energy_fraction"]
            torch.cuda.synchronize()
            value, derivative = measured(f.detach().cpu().numpy(), fd.detach().cpu().numpy(), dx, target, input_power)
            raw[f"{variant}_{n}_normalized"] = value["normalized"]
            rows.append({"case": "C1", "variant": variant, "n": n, "dx_mm": dx, "window_mm": n*dx, "input_power": input_power, "sensor_power": value["power"], "loss": value["loss"], **derivative, **extra})
        del tensor, tangent_tensor
    # The RS convolution is linear and zero padded. Crop to the same source and
    # detector support only after verifying the omitted source power is zero.
    sensor_width = 64 * round(.0063 / dx)
    crop_n = 2 * sensor_width
    crop_start = (smallest - crop_n) // 2
    if np.any(source[:crop_start]) or np.any(source[crop_start+crop_n:]) or np.any(source[:, :crop_start]) or np.any(source[:, crop_start+crop_n:]):
        raise AssertionError("nonzero source omitted by RS crop")
    cropped = source[crop_start:crop_start+crop_n, crop_start:crop_start+crop_n]
    cropped_d = tangent[crop_start:crop_start+crop_n, crop_start:crop_start+crop_n]
    reference = propagate_rs1(cropped, dx, case["wavelength_mm"], case["propagation_mm"])
    reference_d = propagate_rs1(cropped_d, dx, case["wavelength_mm"], case["propagation_mm"])
    value, derivative = measured(reference, reference_d, dx, target, input_power)
    raw["rs_normalized"] = value["normalized"]
    rows.append({"case": "C1", "variant": "RS", "n": crop_n, "dx_mm": dx, "window_mm": crop_n*dx, "input_power": input_power, "sensor_power": value["power"], "loss": value["loss"], **derivative, "cutoff_cycles_per_mm": None, "discarded_spectral_energy_fraction": 0.0, "discarded_tangent_spectral_energy_fraction": 0.0})
    torch.cuda.synchronize()
    timing = {"elapsed_seconds": time.perf_counter()-started, "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "device": torch.cuda.get_device_name()}
    with (out / "pilot_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(out / "raw.npz", **raw)
    (out / "metrics.json").write_text(json.dumps({"timing": timing, "checks": {"source_crop_power_zero": True, "sensor_centers_matched": True}, "hashes": {"protocol": sha(protocol_path), "cases": sha(ROOT / "configs/week1/cases.json"), "phase": sha(phase_path), "target": sha(target_path), "asm": sha(ROOT / "arw_reference/asm.py"), "asm_variant": sha(ROOT / "arw_reference/asm_variants.py"), "rs": sha(ROOT / "arw_reference/rs.py")}}, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), **timing}))


if __name__ == "__main__":
    main()
