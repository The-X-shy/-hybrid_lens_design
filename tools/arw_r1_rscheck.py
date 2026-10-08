"""G1-R1 pilot: direct derivative of the existing discrete RS convolution.

This is a diagnostic runner, not a gate.  The caller supplies a frozen pilot
protocol and accounts for this invocation as one propagation batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arw_reference.rs import propagate_rs1, propagate_rs1_directional  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sampled_fields(phase: np.ndarray, v: np.ndarray, n: int, dx: float, pitch: float, radius: float):
    axis = (np.arange(n, dtype=np.float64) - (n - 1) / 2) * dx
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    ix = np.floor(xx / pitch + 32).astype(np.int64)
    iy = np.floor(yy / pitch + 32).astype(np.int64)
    mask = (xx * xx + yy * yy < radius * radius) & (ix >= 0) & (ix < 64) & (iy >= 0) & (iy < 64)
    ix = np.clip(ix, 0, 63)
    iy = np.clip(iy, 0, 63)
    field = mask * np.exp(1j * phase[iy, ix])
    tangent = 1j * v[iy, ix] * field
    return field, tangent


def crop_source(field: np.ndarray, tangent: np.ndarray, dx: float, sensor_pitch: float, sensor_pixels: int):
    n = field.shape[0]
    q = round(sensor_pitch / dx)
    if q < 1 or abs(q * dx - sensor_pitch) > 1e-10:
        raise ValueError("sensor pixel is not an integer number of samples")
    width = sensor_pixels * q
    if n < width or (n - width) % 2:
        raise ValueError("full detector is not centered on this grid")
    crop_n = min(n, 2 * width)
    crop_start = (n - crop_n) // 2
    if n - crop_n != 2 * crop_start:
        raise ValueError("source crop changes center coordinate")
    outside = field.copy()
    outside[crop_start : crop_start + crop_n, crop_start : crop_start + crop_n] = 0
    outside_power = float(np.vdot(outside, outside).real * dx * dx)
    if outside_power > 1e-24:
        raise ValueError(f"nonzero source power cropped: {outside_power}")
    full_detector_start = (n - width) // 2
    cropped_detector_start = (crop_n - width) // 2
    if full_detector_start != crop_start + cropped_detector_start:
        raise ValueError("detector indexing changes after crop")
    return (field[crop_start : crop_start + crop_n, crop_start : crop_start + crop_n],
            tangent[crop_start : crop_start + crop_n, crop_start : crop_start + crop_n],
            {"crop_n": crop_n, "crop_start": crop_start,
             "outside_source_power": outside_power,
             "full_detector_start": full_detector_start,
             "cropped_detector_start": cropped_detector_start,
             "centers_identical": True, "detector_indices_identical": True})


def sensor_images(field: np.ndarray, tangent: np.ndarray, dx: float, sensor_pitch: float, pixels: int):
    q = round(sensor_pitch / dx)
    width = pixels * q
    start = (field.shape[0] - width) // 2
    if start < 0 or field.shape[0] - width != 2 * start:
        raise ValueError("detector crop is not centered")
    area_shape = (pixels, q, pixels, q)
    f = field[start : start + width, start : start + width]
    df = tangent[start : start + width, start : start + width]
    image = np.square(np.abs(f)).reshape(area_shape).mean(axis=(1, 3))
    dimage = (2 * np.real(np.conj(f) * df)).reshape(area_shape).mean(axis=(1, 3))
    return image, dimage


def loss_parts(image: np.ndarray, target: np.ndarray, input_power: float, sensor_pitch: float):
    total = float(image.sum())
    if total <= 0:
        raise ValueError("nonpositive detector intensity")
    normalized = image / total
    sensor_power = total * sensor_pitch**2
    image_loss = 0.5 * float(np.square(normalized - target).sum())
    power_loss = 0.1 * (1 - sensor_power / input_power) ** 2
    return {
        "normalized": normalized,
        "sensor_power": sensor_power,
        "image_loss": image_loss,
        "power_loss": power_loss,
        "total_loss": image_loss + power_loss,
    }


def direct_parts(image: np.ndarray, dimage: np.ndarray, target: np.ndarray,
                 input_power: float, sensor_pitch: float):
    parts = loss_parts(image, target, input_power, sensor_pitch)
    total = float(image.sum())
    dtotal = float(dimage.sum())
    dnormalized = dimage / total - image * dtotal / (total * total)
    dsensor_power = dtotal * sensor_pitch**2
    dimage_loss = float(np.sum((parts["normalized"] - target) * dnormalized))
    dpower_loss = -0.2 * (1 - parts["sensor_power"] / input_power) * dsensor_power / input_power
    return {"image_derivative": dimage_loss, "power_derivative": dpower_loss,
            "total_derivative": dimage_loss + dpower_loss,
            "sensor_power_derivative": dsensor_power, "dnormalized": dnormalized,
            "image_loss": parts["image_loss"], "power_loss": parts["power_loss"],
            "total_loss": parts["total_loss"], "sensor_power": parts["sensor_power"],
            "normalized": parts["normalized"]}


def evaluate(case: dict, phase: np.ndarray, v: np.ndarray, target: np.ndarray,
             n: int, dx: float, steps: list[float], arrays: dict, tag: str,
             near_zero_amplitude: float, near_zero_absolute: float, relative_max: float):
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    sensor_pitch = float(case["sensor_pitch_mm"])
    sensor_pixels = int(case["sensor_pixels"][0])
    field, tangent = sampled_fields(phase, v, n, dx, case["doe_pitch_mm"], case["aperture_radius_mm"])
    input_power = float(np.vdot(field, field).real * dx * dx)
    input_power_derivative = float(2 * np.real(np.vdot(field, tangent)) * dx * dx)
    if abs(input_power_derivative) > 1e-12 * input_power:
        raise ValueError("phase tangent changed input power")
    source, dsource, crop = crop_source(field, tangent, dx, sensor_pitch, sensor_pixels)
    observed, dobserved = propagate_rs1_directional(source, dsource, dx,
                                                      case["wavelength_mm"], case["propagation_mm"])
    image, dimage = sensor_images(observed, dobserved, dx, sensor_pitch, sensor_pixels)
    direct = direct_parts(image, dimage, target, input_power, sensor_pitch)
    rows = []
    for h in steps:
        sides = []
        for sign in (1, -1):
            perturbed = source * np.exp(1j * sign * h * np.divide(dsource, 1j * source,
                              out=np.zeros_like(source), where=source != 0).real)
            f = propagate_rs1(perturbed, dx, case["wavelength_mm"], case["propagation_mm"])
            side_image, _ = sensor_images(f, np.zeros_like(f), dx, sensor_pitch, sensor_pixels)
            arrays[f"{tag}_h{h:g}_{'plus' if sign > 0 else 'minus'}_image"] = side_image
            sides.append(loss_parts(side_image, target, input_power, sensor_pitch))
        fd_image = (sides[0]["image_loss"] - sides[1]["image_loss"]) / (2 * h)
        fd_power = (sides[0]["power_loss"] - sides[1]["power_loss"]) / (2 * h)
        fd_total = (sides[0]["total_loss"] - sides[1]["total_loss"]) / (2 * h)
        fd_sensor_power = (sides[0]["sensor_power"] - sides[1]["sensor_power"]) / (2 * h)
        d = direct["total_derivative"]
        rel = abs(fd_total - d) / max(abs(fd_total), abs(d), 1e-30)
        near_zero = max(abs(fd_total), abs(d)) < near_zero_amplitude
        agree = abs(fd_total - d) <= near_zero_absolute if near_zero else rel <= relative_max
        rows.append({"h": h, "fd_image_derivative": fd_image, "fd_power_derivative": fd_power,
                     "fd_total_derivative": fd_total, "fd_sensor_power_derivative": fd_sensor_power,
                     "absolute_error": abs(fd_total - d),
                     "relative_error": rel, "near_zero_rule": near_zero,
                     "agree_at_frozen_threshold": bool(agree)})
    arrays[f"{tag}_image"] = image
    arrays[f"{tag}_dimage"] = dimage
    arrays[f"{tag}_normalized"] = direct["normalized"]
    arrays[f"{tag}_dnormalized"] = direct["dnormalized"]
    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"case": case["id"], "tag": tag, "grid_n": n, "dx_mm": dx, "window_mm": n * dx,
            "direction": 0, "input_power": input_power, "input_power_derivative": input_power_derivative,
            "source_crop": crop, "sensor_power": direct["sensor_power"],
            "sensor_power_derivative": direct["sensor_power_derivative"],
            "image_loss": direct["image_loss"], "power_loss": direct["power_loss"],
            "total_loss": direct["total_loss"],
            "direct_image_derivative": direct["image_derivative"],
            "direct_power_derivative": direct["power_derivative"],
            "direct_total_derivative": direct["total_derivative"],
            "fd": rows, "all_fd_agree": all(row["agree_at_frozen_threshold"] for row in rows),
            "wall_seconds": time.perf_counter() - t0, "cpu_seconds": time.process_time() - cpu0,
            "process_peak_rss_mib": rss_kib / 1024}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("C1", "C2", "BOTH"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--direction", type=int, default=0)
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text())
    pilot = protocol["rs_pilot"]
    steps = [float(x) for x in pilot["steps"]]
    if steps != [.003, .001, .0003]:
        raise ValueError("G1-R1 RS pilot requires predeclared h=[.003,.001,.0003]")
    if args.direction != 0:
        raise ValueError("pilot checks predeclared direction 0")
    rules = protocol["preregistered_rules"]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "rs_check.json").exists() or (output / "rs_check.npz").exists():
        raise FileExistsError("RS pilot evidence already exists; use a new run directory")
    cases_path = ROOT / "configs/week1/cases.json"
    cases = json.loads(cases_path.read_text())["cases"]
    chosen = ["C1", "C2"] if args.case == "BOTH" else [args.case]
    arrays = {}
    results = []
    inputs = {"protocol_sha256": sha256(protocol_path), "cases_sha256": sha256(cases_path),
              "rs_sha256": sha256(ROOT / "arw_reference/rs.py"),
              "script_sha256": sha256(Path(__file__).resolve())}
    start = time.perf_counter()
    for case_id in chosen:
        case = next(c for c in cases if c["id"] == case_id)
        phase_path = ROOT / case["initial_phase_file"]
        target_path = ROOT / case["target_file"]
        phase = np.load(phase_path).astype(np.float64)
        target = np.load(target_path).astype(np.float64)
        if phase.shape != (64, 64) or target.shape != (64, 64):
            raise ValueError("physical DOE/target must remain 64x64")
        rng = np.random.default_rng(101 if case_id == "C1" else 102)
        v = rng.standard_normal((64, 64))
        v /= np.sqrt(np.square(v).mean())
        arrays[f"{case_id}_direction0"] = v
        inputs[f"{case_id}_phase_sha256"] = sha256(phase_path)
        inputs[f"{case_id}_target_sha256"] = sha256(target_path)
        inputs[f"{case_id}_direction0_sha256"] = hashlib.sha256(v.tobytes()).hexdigest()
        for setting in pilot["settings"][case_id]:
            n = int(setting["n"])
            dx = float(setting["dx_mm"])
            tag = f"{case_id}_{setting.get('id', f'n{n}')}"
            results.append(evaluate(case, phase, v, target, n, dx, steps, arrays, tag,
                                    float(rules["near_zero_amplitude"]),
                                    float(rules["near_zero_absolute"]),
                                    float(rules["direction_relative_max"])))
    np.savez_compressed(output / "rs_check.npz", **arrays)
    payload = {"status": "SUCCESS", "scope": "one pilot batch; discrete RS direct derivative",
               "inputs": inputs, "results": results,
               "all_fd_agree": all(r["all_fd_agree"] for r in results),
               "total_wall_seconds": time.perf_counter() - start,
               "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               "gpu_seconds": 0, "gpu_usage": "none; NumPy CPU-only process"}
    (output / "rs_check.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(output), "settings": len(results),
                      "all_fd_agree": payload["all_fd_agree"],
                      "total_wall_seconds": payload["total_wall_seconds"]}))


if __name__ == "__main__":
    main()
