"""Paired upstream/host fixture with recorded primary and angular samples."""

import argparse
import importlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("upstream", "integrated"), required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sample-file")
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    device = torch.device(config["device"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    if args.backend == "upstream":
        module = importlib.import_module("deeplens.diffractive_surface.DoeRaywavePlane")
        from deeplens.geometric_surface import Aspheric
        from deeplens.geolens import GeoLens
        from deeplens.light import Ray
    else:
        module = importlib.import_module("arw_integration.doe_plane")
        from arw_integration.raywave import RayWave
        from deeplens.optics.geometric_surface.aspheric import Aspheric
        from deeplens.geolens import GeoLens
        from deeplens.optics.ray import Ray
    DOE = module.DoeRaywavePlane
    source_sample = np.load(args.sample_file) if args.sample_file else None
    records = {}
    original_primary = DOE._sample_primary_xy_from_grid
    original_angular = module._sample_2d_pdf

    def primary_wrapper(self, count):
        if source_sample is None:
            xy, idx = original_primary(self, count)
            records["primary_idx"] = idx.detach().cpu().numpy()
            records["primary_xy"] = xy.detach().cpu().numpy()
            return xy, idx
        idx = torch.as_tensor(source_sample["primary_idx"], device=device, dtype=torch.long)
        if idx.numel() != count:
            raise ValueError("primary sample count mismatch")
        h, w = self.field.shape
        xs, ys = self._xy_axes_centered(h, w)
        xx, yy = torch.meshgrid(xs, ys, indexing="xy")
        xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)[idx]
        if not torch.allclose(xy, torch.as_tensor(source_sample["primary_xy"], device=device), rtol=0, atol=1e-12):
            raise ValueError("primary xy mismatch")
        records["primary_idx"] = idx.detach().cpu().numpy()
        records["primary_xy"] = xy.detach().cpu().numpy()
        return xy, idx

    def angular_wrapper(pdf, count, eps=1e-7):
        if source_sample is None:
            y, x, sampled_pdf = original_angular(pdf, count, eps)
            records["secondary_y"] = y.detach().cpu().numpy()
            records["secondary_x"] = x.detach().cpu().numpy()
            records["sampled_pdf"] = sampled_pdf.detach().cpu().numpy()
            records["full_pdf"] = pdf.detach().cpu().numpy()
            return y, x, sampled_pdf
        y = torch.as_tensor(source_sample["secondary_y"], device=device, dtype=torch.long)
        x = torch.as_tensor(source_sample["secondary_x"], device=device, dtype=torch.long)
        if y.numel() != count or x.numel() != count or tuple(pdf.shape) != tuple(source_sample["full_pdf"].shape):
            raise ValueError("angular sample count or PDF shape mismatch")
        sampled_pdf = pdf[y, x]
        expected = torch.as_tensor(source_sample["sampled_pdf"], device=device, dtype=sampled_pdf.dtype)
        if not torch.allclose(sampled_pdf, expected, rtol=1e-5, atol=1e-8):
            raise ValueError("sample PDF mismatch")
        records["secondary_y"] = y.detach().cpu().numpy()
        records["secondary_x"] = x.detach().cpu().numpy()
        records["sampled_pdf"] = sampled_pdf.detach().cpu().numpy()
        records["full_pdf"] = pdf.detach().cpu().numpy()
        return y, x, sampled_pdf

    DOE._sample_primary_xy_from_grid = primary_wrapper
    module._sample_2d_pdf = angular_wrapper
    phase = torch.from_numpy(np.load(config["phase_path"])).to(device=device, dtype=torch.float32).requires_grad_(True)
    phase_direction = torch.from_numpy(np.load(config["direction_path"])).to(device=device, dtype=torch.float32)
    n = phase.shape[0]
    pitch = float(config["pitch_mm"])
    radius = n * pitch / 2
    xs = (torch.arange(n, device=device) - n / 2) * pitch
    xx, yy = torch.meshgrid(xs, xs, indexing="xy")
    aperture = ((xx**2 + yy**2) < radius**2).float()
    field = aperture * torch.exp(1j * phase)
    doe = DOE(
        r=radius, d=0.0, mat2="air", field=field, dx=pitch, dy=pitch,
        pad_flat=True, reset_primary_ray=True,
        out_primary_ray_count=int(config["primary_samples"]),
        sampled_secondary_ray_count=int(config["secondary_samples"]),
        preserve_energy=True, is_tracing=False, device=device,
    )
    wavelength = float(config["wavelength_um"])
    lens = GeoLens(primary_wvln=wavelength, device=device) if args.backend == "upstream" else GeoLens(device=device)
    if args.backend == "upstream":
        lens.float_enpd = True
        lens.float_foclen = False
        lens.float_hfov = False
        lens.enpd = None
    lens_surfaces = []
    if config["path"] == "lens":
        lens_surfaces = [
            Aspheric(r=12.7, d=3.0, c=1 / 25.8, k=0.0, ai=[0.0] * 6, mat2="1.5131/inf", device=device),
            Aspheric(r=12.7, d=8.3, c=0.0, k=0.0, ai=[0.0] * 6, mat2="air", device=device),
        ]
    sensor_z = float(config["sensor_z_mm"])
    lens.surfaces = ([doe] if args.backend == "upstream" else []) + lens_surfaces
    lens.d_sensor = torch.tensor(sensor_z)
    lens.set_sensor(sensor_res=(n, n), sensor_size=(n * pitch, n * pitch))
    if args.backend == "upstream":
        lens.post_computation()
    doe.is_tracing = True
    locations = np.asarray(config["ray_xy_mm"], dtype=np.float64)
    origins = torch.tensor(np.column_stack((locations, np.zeros(len(locations)))), device=device, dtype=torch.float64)
    directions = torch.zeros_like(origins)
    directions[:, 2] = 1.0
    if args.backend == "upstream":
        ray = Ray(origins, directions, wavelength, is_coherent=True, device=device)
        system = lens
        geometry_source = inspect.getfile(GeoLens)
    else:
        ray = Ray(origins, directions, wavelength, coherent=True, device=device)
        control_before = None
        lens_direction_change = None
        if lens_surfaces:
            control_ray = Ray(origins.clone(), directions.clone(), wavelength, coherent=True, device=device)
            control_before = lens.trace2sensor(control_ray)
            control_before = (control_before.o.detach().clone(), control_before.d.detach().clone(), control_before.opl.detach().clone())

        class IntegratedSystem(RayWave):
            def __init__(self):
                self.device = device
                self.sensor_res = (n, n)
                self.pixel_size = pitch

            def trace2sensor(self, incoming):
                nonlocal lens_direction_change
                outgoing = doe.ray_reaction(incoming, 1.0, 1.0)
                if lens_surfaces:
                    before = outgoing.d.detach().clone()
                    traced = lens.trace2sensor(outgoing)
                    lens_direction_change = float(torch.linalg.vector_norm(traced.d.detach() - before, dim=-1).max().cpu())
                    return traced
                return outgoing.prop_to(sensor_z)

        system = IntegratedSystem()
        geometry_source = inspect.getfile(GeoLens)
    _, _, sensor_field, _ = system.huygens_psf(ray, oversamp=float(config["oversamp"]))
    intensity = sensor_field.abs().square()
    weight = (xx / radius) + 0.4 * (yy / radius) + 1.0
    loss = (intensity * weight).sum() / intensity.sum().clamp_min(1e-30)
    gradient = torch.autograd.grad(loss, phase)[0]
    directional_derivative = (gradient * phase_direction).sum()
    host_regression_difference = None
    if args.backend == "integrated" and lens_surfaces:
        control_ray = Ray(origins.clone(), directions.clone(), wavelength, coherent=True, device=device)
        control_after = lens.trace2sensor(control_ray)
        host_regression_difference = max(
            float((control_after.o - control_before[0]).abs().max().cpu()),
            float((control_after.d - control_before[1]).abs().max().cpu()),
            float((control_after.opl - control_before[2]).abs().max().cpu()),
        )
    if not records or not all(k in records for k in ("primary_idx", "secondary_y", "secondary_x", "sampled_pdf", "full_pdf")):
        raise RuntimeError("sample capture/replay incomplete")
    np.savez_compressed(output / "samples.npz", **records)
    np.savez_compressed(output / "raw.npz", field=sensor_field.detach().cpu().numpy(), intensity=intensity.detach().cpu().numpy(), gradient=gradient.detach().cpu().numpy(), input_o=origins.detach().cpu().numpy(), input_d=directions.detach().cpu().numpy())
    metrics = {
        "backend": args.backend,
        "path": config["path"],
        "loss": float(loss.detach().cpu()),
        "directional_derivative": float(directional_derivative.detach().cpu()),
        "raw_power": float(intensity.sum().detach().cpu()),
        "field_shape": list(sensor_field.shape),
        "geometry_module": geometry_source,
        "doe_module": inspect.getfile(DOE),
        "raywave_module": inspect.getfile(type(system).mro()[1]) if args.backend == "integrated" else inspect.getfile(type(system)),
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
    }
    if args.backend == "integrated" and lens_surfaces:
        metrics["lens_direction_change"] = lens_direction_change
        metrics["host_no_doe_regression_max_difference"] = host_regression_difference
    (output / "pair_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
