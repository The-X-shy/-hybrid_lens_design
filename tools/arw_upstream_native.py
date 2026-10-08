"""Minimal Fig5-style invocation of the unmodified upstream DeepLens path.

This is a qualification fixture, not a replacement propagation implementation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from deeplens.diffractive_surface import DoeRaywavePlane
from deeplens.geolens import GeoLens
from deeplens.light import Ray


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    device = torch.device(config["device"])
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    phase = torch.from_numpy(np.load(config["phase_path"])).to(device=device, dtype=torch.float32).requires_grad_(True)
    n = phase.shape[0]
    pitch = float(config["pitch_mm"])
    radius = n * pitch / 2
    xs = (torch.arange(n, device=device) - n / 2) * pitch
    xx, yy = torch.meshgrid(xs, xs, indexing="xy")
    aperture = ((xx**2 + yy**2) < radius**2).to(torch.float32)
    field = aperture * torch.exp(1j * phase)
    doe = DoeRaywavePlane(
        r=radius, d=0.0, mat2="air", field=field, dx=pitch, dy=pitch,
        pad_flat=bool(config["pad_flat"]), reset_primary_ray=False,
        sampled_secondary_ray_count=int(config["secondary_samples"]),
        preserve_energy=True, is_tracing=False, device=device,
    )
    lens = GeoLens(primary_wvln=float(config["wavelength_um"]), device=device)
    lens.float_enpd = True
    lens.float_foclen = False
    lens.float_hfov = False
    lens.enpd = None
    lens.surfaces = [doe]
    lens.d_sensor = torch.tensor(float(config["sensor_z_mm"]))
    lens.set_sensor(sensor_res=(n, n), sensor_size=(n * pitch, n * pitch))
    lens.post_computation()
    lens.surfaces[0].is_tracing = True
    origin = torch.tensor([[[0.0, 0.0, 0.0]]], device=device, dtype=torch.float64)
    direction = torch.tensor([[[0.0, 0.0, 1.0]]], device=device, dtype=torch.float64)
    ray = Ray(o=origin, d=direction, wvln=float(config["wavelength_um"]), is_coherent=True, device=device)
    _, _, sensor_field, _ = lens.huygens_psf(ray, oversamp=float(config["oversamp"]))
    intensity = sensor_field.abs().square()
    weight = (xx / radius) + 0.4 * (yy / radius) + 1.0
    loss = (intensity * weight).sum() / intensity.sum().clamp_min(1e-30)
    grad = torch.autograd.grad(loss, phase)[0]
    dv = (grad * torch.from_numpy(np.load(config["direction_path"])).to(device)).sum()
    raw = {
        "field": sensor_field.detach().cpu().numpy(),
        "intensity": intensity.detach().cpu().numpy(),
        "gradient": grad.detach().cpu().numpy(),
    }
    np.savez_compressed(out / "raw.npz", **raw)
    metrics = {
        "loss": float(loss.detach().cpu()),
        "directional_derivative": float(dv.detach().cpu()),
        "raw_power": float(intensity.sum().detach().cpu()),
        "field_shape": list(sensor_field.shape),
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
    }
    (out / "native_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
