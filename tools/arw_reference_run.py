"""Run the frozen S0 ASM/RS reference and derivative checks."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from arw_reference.asm import propagate as asm
from arw_reference.common import incident_field, metrics as asm_metrics
from arw_reference.rs import propagate_rs1


ROOT = Path(__file__).resolve().parents[1]


def numerical_field(phase: np.ndarray, n: int, dx: float, pitch: float, radius: float) -> np.ndarray:
    axis = (np.arange(n) - (n - 1) / 2) * dx
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    ix = np.floor(xx / pitch + 32).astype(int)
    iy = np.floor(yy / pitch + 32).astype(int)
    mask = (xx * xx + yy * yy < radius * radius) & (ix >= 0) & (ix < 64) & (iy >= 0) & (iy < 64)
    return mask * np.exp(1j * phase[np.clip(iy, 0, 63), np.clip(ix, 0, 63)])


def rs_metrics(field: np.ndarray, dx: float, target: np.ndarray, input_power: float) -> dict:
    q = round(.0063 / dx)
    if abs(q * dx - .0063) > 1e-10:
        raise ValueError("sensor sampling mismatch")
    width = 64 * q
    start = (field.shape[0] - width) // 2
    image = (np.abs(field[start:start + width, start:start + width]) ** 2).reshape(64, q, 64, q).mean(axis=(1, 3))
    power = float(image.sum() * .0063**2)
    normalized = image / max(float(image.sum()), 1e-30)
    loss = float(.5 * np.square(normalized - target).sum() + .1 * (1 - power / input_power) ** 2)
    x = (np.arange(64) - 31.5) * .0063
    xx, yy = np.meshgrid(x, x, indexing="xy")
    centroid = [float((normalized * xx).sum()), float((normalized * yy).sum())]
    return {"image": image, "normalized": normalized, "power": power, "loss": loss, "centroid": centroid}


def settings(protocol: dict) -> list[tuple[str, int, float]]:
    grid = [(f"grid{n}", n, protocol["grid_ladder"]["window_mm"] / n) for n in protocol["grid_ladder"]["samples"]]
    window = [(f"window{n}", n, protocol["window_ladder"]["dx_mm"]) for n in protocol["window_ladder"]["samples"]]
    return grid + [item for item in window if not any(item[1] == old[1] and abs(item[2] - old[2]) < 1e-12 for old in grid)]


def make_directions(case_id: str, count: int) -> list[np.ndarray]:
    rng = np.random.default_rng(101 if case_id == "C1" else 102)
    directions = []
    for _ in range(count):
        v = rng.standard_normal((64, 64))
        v /= np.sqrt(np.square(v).mean())
        directions.append(v)
    return directions


def run_ref(case: dict, protocol: dict, output: Path) -> None:
    phase = np.load(ROOT / case["initial_phase_file"]).astype(np.float64)
    target = np.load(ROOT / case["target_file"]).astype(np.float64)
    series = []
    raw = {}
    for label, n, dx in settings(protocol):
        e0 = numerical_field(phase, n, dx, case["doe_pitch_mm"], case["aperture_radius_mm"])
        input_power = float(np.square(np.abs(e0)).sum() * dx * dx)
        for method in ("ASM", "RS"):
            if method == "ASM":
                field = asm(torch.as_tensor(e0, device="cuda", dtype=torch.complex128), dx, case["wavelength_mm"], case["propagation_mm"]).detach().cpu().numpy()
            else:
                field = propagate_rs1(e0, dx, case["wavelength_mm"], case["propagation_mm"])
            result = rs_metrics(field, dx, target, input_power)
            series.append({"case": case["id"], "method": method, "setting": label, "grid_n": n, "dx_mm": dx, "window_mm": n * dx, "input_power": input_power, "window_power": result["power"], "loss": result["loss"], "centroid_mm": result["centroid"], "finite": bool(np.isfinite(field).all())})
            raw[f"{method}_{label}_image"] = result["image"]
            raw[f"{method}_{label}_normalized"] = result["normalized"]
            if label in ("grid512", "window1024"):
                raw[f"{method}_{label}_field"] = field
    (output / "series.json").write_text(json.dumps(series, indent=2) + "\n")
    np.savez_compressed(output / "raw.npz", **raw)
    (output / "ref_metrics.json").write_text(json.dumps({"status": "SUCCESS", "case": case["id"], "settings": len(series), "finite": all(x["finite"] for x in series)}, indent=2) + "\n")
    print(json.dumps({"case": case["id"], "settings": len(series)}))


def run_check(case: dict, protocol: dict, output: Path) -> None:
    phase0 = np.load(ROOT / case["initial_phase_file"]).astype(np.float64)
    target_np = np.load(ROOT / case["target_file"]).astype(np.float64)
    directions = make_directions(case["id"], protocol["directions_per_case"])
    derivatives = []
    for label, n, dx in settings(protocol):
        phase = torch.tensor(phase0, device="cuda", dtype=torch.float64, requires_grad=True)
        e0 = incident_field(phase, n, dx, case["doe_pitch_mm"], case["aperture_radius_mm"])
        input_power = e0.abs().square().sum() * dx * dx
        field = asm(e0, dx, case["wavelength_mm"], case["propagation_mm"])
        loss = asm_metrics(field, dx, torch.as_tensor(target_np, device="cuda"), input_power)["loss"]
        grad = torch.autograd.grad(loss, phase)[0].detach().cpu().numpy()
        ad = float((grad * directions[0]).sum())
        step = .001
        rs_values = []
        for sign in (1, -1):
            e = numerical_field(phase0 + sign * step * directions[0], n, dx, case["doe_pitch_mm"], case["aperture_radius_mm"])
            power = float(np.square(np.abs(e)).sum() * dx * dx)
            sensor_width = 64 * round(.0063 / dx)
            crop_n = 2 * sensor_width
            crop_start = (n - crop_n) // 2
            cropped = e[crop_start:crop_start + crop_n, crop_start:crop_start + crop_n]
            f = propagate_rs1(cropped, dx, case["wavelength_mm"], case["propagation_mm"])
            rs_values.append(rs_metrics(f, dx, target_np, power)["loss"])
        rs_fd = (rs_values[0] - rs_values[1]) / (2 * step)
        derivatives.append({"case": case["id"], "setting": label, "asm_AD": ad, "rs_FD": rs_fd})
    _, n, dx = settings(protocol)[-1]
    fd_rows = []
    target = torch.as_tensor(target_np, device="cuda", dtype=torch.float64)

    def objective(phase_np):
        phase = torch.as_tensor(phase_np, device="cuda", dtype=torch.float64)
        e0 = incident_field(phase, n, dx, case["doe_pitch_mm"], case["aperture_radius_mm"])
        f = asm(e0, dx, case["wavelength_mm"], case["propagation_mm"])
        return float(asm_metrics(f, dx, target, e0.abs().square().sum() * dx * dx)["loss"].detach().cpu())

    for direction_id, v in enumerate(directions):
        phase = torch.tensor(phase0, device="cuda", dtype=torch.float64, requires_grad=True)
        e0 = incident_field(phase, n, dx, case["doe_pitch_mm"], case["aperture_radius_mm"])
        f = asm(e0, dx, case["wavelength_mm"], case["propagation_mm"])
        loss = asm_metrics(f, dx, target, e0.abs().square().sum() * dx * dx)["loss"]
        ad = float((torch.autograd.grad(loss, phase)[0].detach().cpu().numpy() * v).sum())
        for step in protocol["finite_difference_steps"]:
            fd = (objective(phase0 + step * v) - objective(phase0 - step * v)) / (2 * step)
            fd_rows.append({"case": case["id"], "direction": direction_id, "step": step, "AD": ad, "FD": fd, "absolute_error": abs(ad - fd), "relative_error": abs(ad - fd) / max(abs(ad), abs(fd), 1e-30)})
    # Fundamental phase and energy invariants on the largest window.
    zero = np.zeros_like(phase0)
    constant = np.full_like(phase0, .75)
    zero_loss = objective(zero)
    const_loss = objective(constant)
    image_zero = asm(torch.as_tensor(numerical_field(zero, n, dx, case["doe_pitch_mm"], case["aperture_radius_mm"]), device="cuda"), dx, case["wavelength_mm"], case["propagation_mm"]).detach().cpu().numpy()
    image_const = asm(torch.as_tensor(numerical_field(constant, n, dx, case["doe_pitch_mm"], case["aperture_radius_mm"]), device="cuda"), dx, case["wavelength_mm"], case["propagation_mm"]).detach().cpu().numpy()
    global_phase_l1 = float(np.abs(np.abs(image_zero)**2 - np.abs(image_const)**2).sum() / max(np.square(np.abs(image_zero)).sum(), 1e-30))
    left = numerical_field(zero, n, dx, case["doe_pitch_mm"], case["aperture_radius_mm"])
    x = (np.arange(n) - (n - 1) / 2) * dx
    opposite = left * np.where(x[None, :] < 0, 1, -1)
    f_opposite = asm(torch.as_tensor(opposite, device="cuda"), dx, case["wavelength_mm"], case["propagation_mm"]).detach().cpu().numpy()
    interference_l1 = float(np.abs(np.abs(image_zero)**2 / max(np.square(np.abs(image_zero)).sum(), 1e-30) - np.abs(f_opposite)**2 / max(np.square(np.abs(f_opposite)).sum(), 1e-30)).sum())
    input_power = float(np.square(np.abs(left)).sum() * dx * dx)
    output_power = float(np.square(np.abs(image_zero)).sum() * dx * dx)
    invariants = {"global_phase_intensity_l1": global_phase_l1, "constant_phase_loss_difference": abs(zero_loss - const_loss), "interference_l1": interference_l1, "full_window_power_relative": abs(input_power - output_power) / input_power}
    (output / "checks.json").write_text(json.dumps({"derivatives": derivatives, "fd_rows": fd_rows, "invariants": invariants}, indent=2) + "\n")
    np.savez_compressed(output / "raw.npz", direction0=directions[0], zero_field=image_zero, opposite_field=f_opposite)
    (output / "ref_metrics.json").write_text(json.dumps({"status": "SUCCESS", "case": case["id"], "direction_checks": len(fd_rows), "invariants": invariants}, indent=2) + "\n")
    print(json.dumps({"case": case["id"], "directions": len(directions)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("ref", "check"), required=True)
    parser.add_argument("--case", choices=("C1", "C2"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", default=str(ROOT / "artifacts/G1/reference_protocol.json"))
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    cases = json.loads((ROOT / "configs/week1/cases.json").read_text())
    case = next(c for c in cases["cases"] if c["id"] == args.case)
    output = Path(args.output)
    if args.mode == "ref":
        run_ref(case, protocol, output)
    else:
        run_check(case, protocol, output)


if __name__ == "__main__":
    main()
