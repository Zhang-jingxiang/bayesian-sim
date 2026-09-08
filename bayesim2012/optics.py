"""Optical models and differential operators."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import tifffile

from .geometry import Shape2D, resolve_reconstruction_shape


def airy_otf(shape: tuple[int, int], cutoff_frequency: float) -> np.ndarray:
    """Create the 2D incoherent Airy OTF in reduced frequency."""
    _validate_cutoff(cutoff_frequency)
    fx, fy = frequency_grid(shape)
    radius = np.sqrt(fx**2 + fy**2)
    normalized = np.zeros_like(radius)
    inside = radius <= cutoff_frequency
    normalized[inside] = radius[inside] / cutoff_frequency
    otf = np.zeros_like(radius)
    otf[inside] = _airy_profile(normalized[inside])
    return otf.astype(np.float64)


def load_otf(
    path: str | Path,
    measurement_shape: Shape2D,
    reconstruction_shape: Shape2D | None = None,
) -> np.ndarray:
    """Load an external OTF, normalize it, and align DC with FFT indexing."""
    target_shape = resolve_reconstruction_shape(measurement_shape, reconstruction_shape)
    otf_path = Path(path)
    if otf_path.suffix.lower() == ".npy":
        otf = np.load(otf_path)
    else:
        otf = tifffile.imread(otf_path)
    centered = _centered_otf(otf, measurement_shape)
    if measurement_shape == target_shape:
        return _normalize_otf(centered)
    return _resample_otf(centered, target_shape)


def latent_cutoff_frequency(
    measurement_shape: Shape2D,
    reconstruction_shape: Shape2D | None,
    cutoff_frequency: float,
) -> float:
    target_shape = resolve_reconstruction_shape(measurement_shape, reconstruction_shape)
    if target_shape == measurement_shape:
        return cutoff_frequency
    scale = _reconstruction_scale(measurement_shape, target_shape)
    return cutoff_frequency / scale


def convolve_with_otf(image: np.ndarray, otf: np.ndarray) -> np.ndarray:
    """Apply an OTF in the Fourier domain."""
    spectrum = np.fft.fft2(image)
    return np.fft.ifft2(otf * spectrum).real


def correlate_with_otf(image: np.ndarray, otf: np.ndarray) -> np.ndarray:
    """Apply the adjoint of an OTF convolution."""
    spectrum = np.fft.fft2(image)
    return np.fft.ifft2(np.conjugate(otf) * spectrum).real


def laplacian(image: np.ndarray) -> np.ndarray:
    """Compute the periodic 2D Laplacian."""
    neighbors = _neighbor_sum(image)
    return neighbors - 4.0 * image


def laplacian_energy(image: np.ndarray) -> float:
    """Return ||D f||^2 for the periodic Laplacian D."""
    diff = laplacian(image)
    return float(np.sum(diff * diff))


@lru_cache(maxsize=None)
def laplacian_spectrum(shape: Shape2D) -> np.ndarray:
    """Return the Fourier symbol of the periodic Laplacian."""
    impulse = np.zeros(shape, dtype=np.float64)
    impulse[0, 0] = 1.0
    return np.fft.fft2(laplacian(impulse)).real


def frequency_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return frequency coordinates in cycles per pixel."""
    rows, cols = shape
    fy = np.fft.fftfreq(rows)
    fx = np.fft.fftfreq(cols)
    return np.meshgrid(fx, fy)


def _validate_cutoff(cutoff_frequency: float) -> None:
    if not 0.0 < cutoff_frequency <= 0.5:
        raise ValueError(f"cutoff_frequency must lie in (0, 0.5], got {cutoff_frequency}")


def _airy_profile(normalized_frequency: np.ndarray) -> np.ndarray:
    angle = np.arccos(normalized_frequency)
    sqrt_term = np.sqrt(1.0 - normalized_frequency**2)
    return (2.0 / np.pi) * (angle - normalized_frequency * sqrt_term)


def _neighbor_sum(image: np.ndarray) -> np.ndarray:
    up = np.roll(image, 1, axis=0)
    down = np.roll(image, -1, axis=0)
    left = np.roll(image, 1, axis=1)
    right = np.roll(image, -1, axis=1)
    return up + down + left + right


def _centered_otf(otf: np.ndarray, measurement_shape: Shape2D) -> np.ndarray:
    if otf.shape != measurement_shape:
        raise ValueError(f"OTF shape {otf.shape} does not match measurement shape {measurement_shape}")
    return np.asarray(otf, dtype=np.float64)


def _normalize_otf(centered_otf: np.ndarray) -> np.ndarray:
    maximum = float(np.max(centered_otf))
    if maximum <= 0.0:
        raise ValueError("OTF maximum must be positive")
    return np.fft.ifftshift(centered_otf / maximum)


def _resample_otf(centered_otf: np.ndarray, target_shape: Shape2D) -> np.ndarray:
    measurement_shape = centered_otf.shape
    scale = _reconstruction_scale(measurement_shape, target_shape)
    fy_in, fx_in = _centered_frequency_axes(measurement_shape)
    fy_out, fx_out = _centered_frequency_axes(target_shape)
    grid_y, grid_x = np.meshgrid(fy_out * scale, fx_out * scale, indexing="ij")
    points = np.stack([grid_y.ravel(), grid_x.ravel()], axis=1)
    interpolator = RegularGridInterpolator((fy_in, fx_in), centered_otf, bounds_error=False, fill_value=0.0)
    resampled = interpolator(points).reshape(target_shape)
    return _normalize_otf(resampled)


def _centered_frequency_axes(shape: Shape2D) -> tuple[np.ndarray, np.ndarray]:
    fy = (np.arange(shape[0], dtype=np.float64) - shape[0] / 2.0) / shape[0]
    fx = (np.arange(shape[1], dtype=np.float64) - shape[1] / 2.0) / shape[1]
    return fy, fx


def _reconstruction_scale(measurement_shape: Shape2D, reconstruction_shape: Shape2D) -> float:
    scale_y = reconstruction_shape[0] / measurement_shape[0]
    scale_x = reconstruction_shape[1] / measurement_shape[1]
    if not np.isclose(scale_y, scale_x):
        raise ValueError(f"anisotropic scale is not supported: {measurement_shape} -> {reconstruction_shape}")
    return float(scale_y)
