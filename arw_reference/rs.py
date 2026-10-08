"""Independent scalar Rayleigh–Sommerfeld free-space reference.

Coordinates and distances are in millimetres.  This module deliberately uses
the spatial-domain first Rayleigh–Sommerfeld integral, not an angular-spectrum
transfer function.  The source and observation planes have the same sampled
extent and pitch; values outside the source array are assumed to be zero.
"""

from __future__ import annotations

import math


def propagate_rs1(field, pitch_mm: float, wavelength_mm: float, distance_mm: float):
    """Propagate a 2-D complex field with the first RS diffraction integral.

    Args:
        field: NumPy complex array ``[y, x]`` on the source plane.
        pitch_mm: Square-grid sample spacing in mm, positive.
        wavelength_mm: Vacuum wavelength in mm, positive.
        distance_mm: Forward free-space propagation distance in mm, nonnegative.

    Returns:
        Complex128 NumPy array on the same physical grid.  At zero distance,
        returns a copy of the input array (the mathematical identity limit).

    The discrete convolution samples

    ``h(x,y,z) = z exp(i k r) / (i lambda r^2) * (1 - 1/(i k r))``,

    where ``r = sqrt(x^2 + y^2 + z^2)`` and ``k = 2 pi/lambda``.  The sum is
    multiplied by the source pixel area.  FFTs only accelerate this *linear*
    convolution; zero padding prevents circular wraparound.

    This is a sampled scalar approximation, not an analytic reference.  The
    kernel phase must be resolved at ``pitch_mm`` across the required offsets;
    very short distances, coarse grids, truncated source windows and large
    angles can give substantial error.  Check grid and window convergence
    independently before using a result as a reference.
    """
    if getattr(field, "ndim", None) != 2 or min(field.shape) < 1:
        raise ValueError("field must be a nonempty 2-D NumPy array")
    if getattr(getattr(field, "dtype", None), "kind", None) != "c":
        raise TypeError("field must have a complex NumPy dtype")
    for name, value, allow_zero in (
        ("pitch_mm", pitch_mm, False),
        ("wavelength_mm", wavelength_mm, False),
        ("distance_mm", distance_mm, True),
    ):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        if value < 0 or (value == 0 and not allow_zero):
            raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")

    if distance_mm == 0:
        return field.copy()

    import numpy as np

    source = np.asarray(field, dtype=np.complex128)
    if not np.isfinite(source).all():
        raise ValueError("field must contain only finite values")

    height, width = source.shape
    y_offsets = np.arange(1 - height, height, dtype=np.float64) * pitch_mm
    x_offsets = np.arange(1 - width, width, dtype=np.float64) * pitch_mm
    yy, xx = np.meshgrid(y_offsets, x_offsets, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy + distance_mm * distance_mm)
    wave_number = 2.0 * math.pi / wavelength_mm
    kernel = (
        distance_mm
        * np.exp(1j * wave_number * radius)
        / (1j * wavelength_mm * radius * radius)
        * (1.0 - 1.0 / (1j * wave_number * radius))
    )

    # Full convolution is (3H-2, 3W-2); its central H x W region is the
    # observation grid aligned with the input grid.
    fft_shape = (3 * height - 2, 3 * width - 2)
    full = np.fft.ifft2(
        np.fft.fft2(source, s=fft_shape) * np.fft.fft2(kernel, s=fft_shape)
    )
    return (
        full[height - 1 : 2 * height - 1, width - 1 : 2 * width - 1]
        * (pitch_mm * pitch_mm)
    )


def propagate_rs1_directional(field, direction_field, pitch_mm: float, wavelength_mm: float, distance_mm: float):
    """Apply the same discrete RS operator to a field and its phase tangent.

    The second result is the *direct derivative of this discrete operator*;
    it makes no claim about the spatial quadrature error of the operator.
    """
    import numpy as np

    if getattr(direction_field, "shape", None) != getattr(field, "shape", None):
        raise ValueError("direction_field must match field shape")
    if getattr(getattr(direction_field, "dtype", None), "kind", None) != "c":
        raise TypeError("direction_field must have a complex NumPy dtype")
    if not np.isfinite(direction_field).all():
        raise ValueError("direction_field must contain only finite values")
    return (
        propagate_rs1(field, pitch_mm, wavelength_mm, distance_mm),
        propagate_rs1(direction_field, pitch_mm, wavelength_mm, distance_mm),
    )
