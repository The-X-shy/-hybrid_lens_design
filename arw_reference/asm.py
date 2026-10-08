"""Angular-spectrum propagation in physical millimetre units."""

import math

import torch


def propagate(field: torch.Tensor, dx_mm: float, wavelength_mm: float, z_mm: float) -> torch.Tensor:
    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError("expected square 2-D complex field")
    n = field.shape[0]
    fx = torch.fft.fftfreq(n, d=dx_mm, device=field.device, dtype=field.real.dtype)
    fy = torch.fft.fftfreq(n, d=dx_mm, device=field.device, dtype=field.real.dtype)
    fxx, fyy = torch.meshgrid(fx, fy, indexing="xy")
    root = torch.sqrt((1 - (wavelength_mm * fxx).square() - (wavelength_mm * fyy).square()).to(field.dtype))
    # Remove only the global propagation phase to keep phase evaluation precise.
    transfer = torch.exp(1j * (2 * math.pi / wavelength_mm) * z_mm * (root - 1))
    return torch.fft.ifft2(torch.fft.fft2(field) * transfer)
