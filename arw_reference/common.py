"""Fixed physical DOE representation and sensor measurement for S0."""

import numpy as np
import torch


def incident_field(phase: torch.Tensor, n: int, dx_mm: float, pitch_mm: float, aperture_radius_mm: float) -> torch.Tensor:
    if phase.shape != (64, 64):
        raise ValueError("physical DOE must remain 64x64")
    axis = (torch.arange(n, device=phase.device, dtype=phase.dtype) - (n - 1) / 2) * dx_mm
    xx, yy = torch.meshgrid(axis, axis, indexing="xy")
    ix = torch.floor(xx / pitch_mm + 32).long()
    iy = torch.floor(yy / pitch_mm + 32).long()
    mask = (xx.square() + yy.square() < aperture_radius_mm**2) & (ix >= 0) & (ix < 64) & (iy >= 0) & (iy < 64)
    selected = phase[iy.clamp(0, 63), ix.clamp(0, 63)]
    return mask.to(phase.dtype) * torch.exp(1j * selected)


def sensor_intensity(field: torch.Tensor, dx_mm: float, sensor_pitch_mm: float = .0063, sensor_pixels: int = 64) -> torch.Tensor:
    q = round(sensor_pitch_mm / dx_mm)
    if q < 1 or abs(q * dx_mm - sensor_pitch_mm) > 1e-10:
        raise ValueError("reference grid must resolve whole sensor pixels")
    width = sensor_pixels * q
    start = (field.shape[0] - width) // 2
    if start < 0 or field.shape[0] - width != 2 * start:
        raise ValueError("sensor crop must be centered exactly")
    intensity = field.abs().square()[start:start + width, start:start + width]
    return intensity.reshape(sensor_pixels, q, sensor_pixels, q).mean(dim=(1, 3))


def metrics(field: torch.Tensor, dx_mm: float, target: torch.Tensor, input_power: torch.Tensor) -> dict:
    image = sensor_intensity(field, dx_mm)
    power = image.sum() * (.0063**2)
    normalized = image / image.sum().clamp_min(1e-30)
    loss = .5 * (normalized - target).square().sum() + .1 * (1 - power / input_power).square()
    coord = (torch.arange(64, device=image.device, dtype=image.dtype) - 31.5) * .0063
    xx, yy = torch.meshgrid(coord, coord, indexing="xy")
    centroid = torch.stack([(normalized * xx).sum(), (normalized * yy).sum()])
    return {"image": image, "normalized": normalized, "power": power, "loss": loss, "centroid": centroid}
