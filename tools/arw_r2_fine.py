"""G1-R2 B1: frozen C2 fine-grid RS baseline, with a reused FFT kernel.

This runner only writes to a new output directory.  It does not judge G1 or
change the historical reference.  All optical arrays use float64/complex128.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ReusedRSKernel:
    """Exactly the existing sampled RS1 linear convolution, cached by setting."""

    def __init__(self, n: int, dx: float, wavelength: float, distance: float):
        if n < 1 or min(dx, wavelength, distance) <= 0:
            raise ValueError("invalid RS setting")
        self.n = n
        self.dx = dx
        self.fft_shape = (3 * n - 2, 3 * n - 2)
        offset = np.arange(1 - n, n, dtype=np.float64) * dx
        yy, xx = np.meshgrid(offset, offset, indexing="ij")
        radius = np.sqrt(xx * xx + yy * yy + distance * distance)
        k = 2.0 * math.pi / wavelength
        kernel = (
            distance * np.exp(1j * k * radius)
            / (1j * wavelength * radius * radius)
            * (1.0 - 1.0 / (1j * k * radius))
        )
        self.kernel_fft = np.fft.fft2(kernel, s=self.fft_shape)
        del yy, xx, radius, kernel
        self.calls = 0

    def __call__(self, source: np.ndarray) -> np.ndarray:
        if source.shape != (self.n, self.n) or source.dtype != np.complex128:
            raise ValueError("source shape or dtype changed")
        transformed = np.fft.fft2(source, s=self.fft_shape)
        transformed *= self.kernel_fft
        observed = np.fft.ifft2(transformed)
        self.calls += 1
        return (
            observed[self.n - 1 : 2 * self.n - 1, self.n - 1 : 2 * self.n - 1].copy()
            * (self.dx * self.dx)
        )


def source_fields(phase: np.ndarray, directions: list[np.ndarray], n: int, dx: float,
                  pitch: float, radius: float):
    axis = (np.arange(n, dtype=np.float64) - (n - 1) / 2) * dx
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    ix = np.floor(xx / pitch + 32).astype(np.int64)
    iy = np.floor(yy / pitch + 32).astype(np.int64)
    mask = (xx * xx + yy * yy < radius * radius) & (ix >= 0) & (ix < 64) & (iy >= 0) & (iy < 64)
    ix = ix.clip(0, 63)
    iy = iy.clip(0, 63)
    field = (mask * np.exp(1j * phase[iy, ix])).astype(np.complex128)
    return field, [v[iy, ix] for v in directions]


def detector(field: np.ndarray, tangent: np.ndarray, q: int, sensor_pixels: int):
    width = sensor_pixels * q
    start = (field.shape[0] - width) // 2
    if start < 0 or field.shape[0] - width != 2 * start:
        raise ValueError("detector center differs after source crop")
    f = field[start : start + width, start : start + width]
    df = tangent[start : start + width, start : start + width]
    shape = (sensor_pixels, q, sensor_pixels, q)
    image = np.square(np.abs(f)).reshape(shape).mean(axis=(1, 3))
    dimage = (2 * np.real(np.conj(f) * df)).reshape(shape).mean(axis=(1, 3))
    return image, dimage


def loss_and_direction(image: np.ndarray, dimage: np.ndarray, target: np.ndarray,
                       input_power: float, sensor_pitch: float):
    total = float(image.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("nonpositive or nonfinite detector power")
    normalized = image / total
    dtotal = float(dimage.sum())
    dnormalized = dimage / total - image * (dtotal / total**2)
    sensor_power = total * sensor_pitch**2
    dsensor_power = dtotal * sensor_pitch**2
    image_loss = 0.5 * float(np.square(normalized - target).sum())
    power_loss = 0.1 * (1 - sensor_power / input_power)**2
    dimage_loss = float(np.sum((normalized - target) * dnormalized))
    dpower_loss = -0.2 * (1 - sensor_power / input_power) * dsensor_power / input_power
    result = {
        "sensor_power": sensor_power,
        "sensor_power_derivative": dsensor_power,
        "image_loss": image_loss,
        "power_loss": power_loss,
        "total_loss": image_loss + power_loss,
        "image_derivative": dimage_loss,
        "power_derivative": dpower_loss,
        "total_derivative": dimage_loss + dpower_loss,
    }
    result["cancellation_factor"] = (
        (abs(dimage_loss) + abs(dpower_loss)) / max(abs(result["total_derivative"]), 1e-30)
    )
    return result, normalized, dnormalized


def save_array(path: Path, array: np.ndarray, hashes: dict[str, str], output: Path):
    np.save(path, array)
    hashes[str(path.relative_to(output))] = sha256(path)


def check_protocol(protocol: dict):
    b1 = protocol["B1"]
    if not protocol.get("frozen_before_propagation"):
        raise ValueError("protocol was not frozen")
    required = (b1["case"] == "C2" and b1["q"] == [8, 12, 16]
                and float(b1["window_mm"]) == 1.6128
                and b1["direction_count"] == 4 and b1["fd_final_direction"] == 0
                and [float(x) for x in b1["fd_steps"]] == [.003, .001, .0003]
                and b1["dtype"] == "complex128"
                and protocol["direction_seed"] == 102
                and protocol["direction_generation"] == "numpy.default_rng.standard_normal_sequential_rms")
    if not required:
        raise ValueError("B1 protocol differs from the frozen C2 baseline")
    return b1


def check_schedule(schedule: dict, protocol_sha256: str, b1: dict, sensor_pixels: int,
                   sensor_pitch: float):
    if schedule.get("parent_protocol_sha256") != protocol_sha256 or schedule.get("batch") != "B1":
        raise ValueError("B1 schedule does not match frozen parent protocol")
    if schedule.get("case") != "C2" or len(schedule.get("settings", [])) != 3:
        raise ValueError("B1 schedule settings changed")
    for row, q in zip(schedule["settings"], b1["q"]):
        dx = sensor_pitch / q
        full_n = round(b1["window_mm"] / dx)
        width = sensor_pixels * q
        crop_n = 2 * width
        if (row["q"] != q or abs(row["dx_mm"] - dx) > 1e-15
                or row["full_n"] != full_n or row["sensor_width"] != width
                or row["source_crop_n"] != crop_n or row["rs_fft_n"] != 3 * crop_n - 2
                or row["four_directions"] != [0, 1, 2, 3]
                or abs(row["full_window_mm"] - b1["window_mm"]) > 1e-15):
            raise ValueError(f"B1 schedule mismatch at q={q}")
    fd = schedule.get("fd_final", {})
    if (fd.get("q") != 16 or fd.get("direction") != 0
            or fd.get("steps") != b1["fd_steps"] or sorted(fd.get("sides", [])) != [-1, 1]):
        raise ValueError("B1 finite-difference schedule changed")
    if schedule.get("kernel_fft_reuse") != "one kernel FFT per setting for base plus four tangents plus finite differences":
        raise ValueError("B1 kernel reuse changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, help="frozen G1-R2 protocol.json")
    parser.add_argument("--schedule", required=True, help="frozen G1-R2 B1_schedule.json")
    parser.add_argument("--output", required=True, help="new B1 directory, never an old gate directory")
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve(strict=True)
    protocol = json.loads(protocol_path.read_text())
    b1 = check_protocol(protocol)
    schedule_path = Path(args.schedule).resolve(strict=True)
    schedule = json.loads(schedule_path.read_text())
    output = Path(args.output).resolve()
    if output == ROOT / "artifacts/G1" or output == ROOT / "artifacts/G1-R1" or (
        ROOT / "artifacts/G1" in output.parents or ROOT / "artifacts/G1-R1" in output.parents
    ):
        raise ValueError("B1 output cannot be written into historical G1/R1 evidence")
    if not (ROOT / "artifacts/G1-R2") in output.parents:
        raise ValueError("B1 output must be inside artifacts/G1-R2")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("B1 output is not empty; existing evidence is immutable")
    cases_path = ROOT / "configs/week1/cases.json"
    case = next(c for c in json.loads(cases_path.read_text())["cases"] if c["id"] == "C2")
    check_schedule(schedule, sha256(protocol_path), b1, int(case["sensor_pixels"][0]),
                   float(case["sensor_pitch_mm"]))
    phase_path = ROOT / case["initial_phase_file"]
    target_path = ROOT / case["target_file"]
    source_paths = [cases_path, phase_path, target_path, ROOT / "arw_reference/rs.py",
                    ROOT / "arw_reference/common.py", Path(__file__).resolve()]
    expected = protocol.get("source_hashes", {})
    for path in source_paths[:5]:
        rel = str(path.relative_to(ROOT))
        if rel in expected and sha256(path) != expected[rel]:
            raise ValueError(f"source hash changed: {rel}")
    phase = np.load(phase_path).astype(np.float64)
    target = np.load(target_path).astype(np.float64)
    if phase.shape != (64, 64) or target.shape != (64, 64):
        raise ValueError("physical phase/target changed")
    rng = np.random.default_rng(102)
    directions = []
    for _ in range(4):
        v = rng.standard_normal((64, 64))
        v /= np.sqrt(np.square(v).mean())
        directions.append(v)
    hashes: dict[str, str] = {}
    t_all = time.perf_counter()
    cpu_all = time.process_time()
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw_fields"
    raw.mkdir()
    save_array(raw / "phase.npy", phase, hashes, output)
    save_array(raw / "target.npy", target, hashes, output)
    for direction_id, v in enumerate(directions):
        save_array(raw / f"direction_{direction_id}.npy", v, hashes, output)

    rows: list[dict] = []
    fd_rows: list[dict] = []
    setting_metadata = []
    for q in b1["q"]:
        t_setting = time.perf_counter()
        cpu_setting = time.process_time()
        dx = float(case["sensor_pitch_mm"]) / q
        full_n = round(b1["window_mm"] / dx)
        detector_width = int(case["sensor_pixels"][0]) * q
        crop_n = 2 * detector_width
        if abs(full_n * dx - b1["window_mm"]) > 1e-12 or full_n < crop_n:
            raise ValueError("full window or cropped source cannot be represented")
        crop_start = (full_n - crop_n) // 2
        if full_n - crop_n != 2 * crop_start or (full_n - detector_width) // 2 != crop_start + (crop_n - detector_width) // 2:
            raise ValueError("sensor indexing changes on source crop")
        source, sampled_v = source_fields(phase, directions, crop_n, dx,
                                          float(case["doe_pitch_mm"]), float(case["aperture_radius_mm"]))
        # Verify the cropped region includes every nonzero aperture sample.
        nonzero_extent = float(case["aperture_radius_mm"])
        cropped_edge = (crop_n / 2) * dx
        if cropped_edge < nonzero_extent or not np.isfinite(source).all():
            raise ValueError("cropped source does not contain the whole aperture")
        outside_source_power = 0.0
        input_power = float(np.vdot(source, source).real * dx * dx)
        qdir = raw / f"q{q}"
        qdir.mkdir()
        save_array(qdir / "source.npy", source, hashes, output)
        kernel_start = time.perf_counter()
        rs = ReusedRSKernel(crop_n, dx, float(case["wavelength_mm"]), float(case["propagation_mm"]))
        kernel_build_seconds = time.perf_counter() - kernel_start
        observed = rs(source)
        save_array(qdir / "observed.npy", observed, hashes, output)
        zero = np.zeros_like(observed)
        base_image, _ = detector(observed, zero, q, int(case["sensor_pixels"][0]))
        del zero
        save_array(qdir / "image.npy", base_image, hashes, output)
        save_array(qdir / "normalized_image.npy", base_image / float(base_image.sum()), hashes, output)
        for direction_id, sampled in enumerate(sampled_v):
            tangent = (1j * sampled * source).astype(np.complex128)
            input_power_derivative = float(2 * np.real(np.vdot(source, tangent)) * dx * dx)
            if abs(input_power_derivative) > 1e-12 * input_power:
                raise ValueError("phase direction changes input power")
            save_array(qdir / f"source_direction_{direction_id}.npy", tangent, hashes, output)
            observed_tangent = rs(tangent)
            save_array(qdir / f"observed_direction_{direction_id}.npy", observed_tangent, hashes, output)
            image, dimage = detector(observed, observed_tangent, q, int(case["sensor_pixels"][0]))
            if not np.array_equal(image, base_image):
                raise ValueError("direction changed unperturbed image")
            save_array(qdir / f"dimage_{direction_id}.npy", dimage, hashes, output)
            parts, normalized, dnormalized = loss_and_direction(image, dimage, target, input_power,
                                                                 float(case["sensor_pitch_mm"]))
            save_array(qdir / f"dnormalized_{direction_id}.npy", dnormalized, hashes, output)
            row = {"case": "C2", "q": q, "direction": direction_id,
                   "dx_mm": dx, "full_n": full_n, "full_window_mm": b1["window_mm"],
                   "source_crop_n": crop_n, "source_crop_window_mm": crop_n * dx,
                   "input_power": input_power, "input_power_derivative": input_power_derivative, **parts}
            rows.append(row)
            if direction_id == 0 and q == b1["q"][-1]:
                for h in b1["fd_steps"]:
                    sides = []
                    for sign in (1, -1):
                        perturbed = (source * np.exp(1j * sign * h * sampled)).astype(np.complex128)
                        side_field = rs(perturbed)
                        side_image, _ = detector(side_field, np.zeros_like(side_field), q,
                                                 int(case["sensor_pixels"][0]))
                        side_parts, _, _ = loss_and_direction(side_image, np.zeros_like(side_image),
                                                               target, input_power, float(case["sensor_pitch_mm"]))
                        save_array(qdir / f"fd_h{h:g}_{'plus' if sign > 0 else 'minus'}_observed.npy",
                                   side_field, hashes, output)
                        save_array(qdir / f"fd_h{h:g}_{'plus' if sign > 0 else 'minus'}_image.npy",
                                   side_image, hashes, output)
                        sides.append(side_parts)
                        del side_field, side_image, perturbed
                    fd = {name: (sides[0][name] - sides[1][name]) / (2 * h)
                          for name in ("sensor_power", "image_loss", "power_loss", "total_loss")}
                    absolute_error = abs(fd["total_loss"] - parts["total_derivative"])
                    relative_error = absolute_error / max(abs(fd["total_loss"]),
                                                          abs(parts["total_derivative"]), 1e-30)
                    fd_rows.append({"case": "C2", "q": q, "direction": direction_id, "h": h,
                                    "direct_total_derivative": parts["total_derivative"],
                                    "fd_sensor_power_derivative": fd["sensor_power"],
                                    "fd_image_derivative": fd["image_loss"],
                                    "fd_power_derivative": fd["power_loss"],
                                    "fd_total_derivative": fd["total_loss"],
                                    "absolute_error": absolute_error, "relative_error": relative_error})
            del tangent, observed_tangent, dimage, normalized, dnormalized
        setting_metadata.append({"q": q, "kernel_build_count": 1, "kernel_build_seconds": kernel_build_seconds,
                                 "propagation_calls": rs.calls, "full_n": full_n, "source_crop_n": crop_n,
                                 "crop_start_in_full": crop_start, "outside_source_power": outside_source_power,
                                 "detector_indices_identical": True,
                                 "wall_seconds": time.perf_counter() - t_setting,
                                 "cpu_seconds": time.process_time() - cpu_setting,
                                 "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024})
        del rs, source, sampled_v, observed, base_image

    with (output / "fine_baseline.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "fine_fd.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fd_rows[0]))
        writer.writeheader()
        writer.writerows(fd_rows)
    payload = {
        "status": "SUCCESS", "scope": "G1-R2 B1 baseline; not a gate decision",
        "protocol_sha256": sha256(protocol_path),
        "schedule_sha256": sha256(schedule_path),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        "raw_sha256": hashes, "settings": setting_metadata,
        "rows": len(rows), "fd_rows": len(fd_rows),
        "total_wall_seconds": time.perf_counter() - t_all,
        "total_cpu_seconds": time.process_time() - cpu_all,
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "gpu_process_seconds": 0, "gpu_kernel_seconds": 0,
    }
    (output / "fine_baseline.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps({
        "status": "SUCCESS", "batch": "B1", "wall_seconds": payload["total_wall_seconds"],
        "cpu_process_seconds": payload["total_cpu_seconds"],
        "peak_cpu_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "peak_gpu_bytes": 0, "gpu_process_seconds": 0,
        "gpu_kernel_seconds": 0, "settings": len(setting_metadata),
        "directions": len(rows), "finite_difference_pairs": len(fd_rows),
    }, indent=2) + "\n")
    print(json.dumps({"output": str(output), "rows": len(rows), "fd_rows": len(fd_rows),
                      "wall_seconds": payload["total_wall_seconds"]}))


if __name__ == "__main__":
    main()
