"""Optional, recorded sampling diagnostic for the unchanged ASM reference."""

import math

import torch


def bandlimited_propagate(field, dx_mm, wavelength_mm, z_mm):
    """Apply a conservative transfer sampling mask; return output and mask loss.

    This is a diagnostic variant, not a replacement for the original method.
    The separable limit requires no more than pi radians of transfer phase
    change between adjacent Fourier samples on either axis.
    """
    n = field.shape[0]
    if field.ndim != 2 or field.shape[1] != n:
        raise ValueError("square field required")
    extent = n * dx_mm
    frequencies = torch.fft.fftfreq(n, d=dx_mm, device=field.device, dtype=field.real.dtype)
    fx, fy = torch.meshgrid(frequencies, frequencies, indexing="xy")
    root = torch.sqrt((1 - (wavelength_mm * fx).square() - (wavelength_mm * fy).square()).to(field.dtype))
    transfer = torch.exp(1j * (2 * math.pi / wavelength_mm) * z_mm * (root - 1))
    cutoff = extent / (wavelength_mm * math.sqrt(4 * z_mm * z_mm + extent * extent))
    mask = ((fx.abs() <= cutoff) & (fy.abs() <= cutoff)).to(field.real.dtype)
    spectrum = torch.fft.fft2(field)
    discarded = float((spectrum.abs().square() * (1 - mask)).sum().div(spectrum.abs().square().sum().clamp_min(1e-30)).item())
    result = torch.fft.ifft2(spectrum * transfer * mask)
    return result, {"cutoff_cycles_per_mm": cutoff, "discarded_spectral_energy_fraction": discarded}
