"""Input loading and simulation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import data as skdata
import tifffile

from .config import InputConfig, NoiseConfig
from .constants import SAMPLING_MODE_BLOCK_AVERAGE
from .geometry import downsample_measurement, resolve_reconstruction_shape
from .optics import convolve_with_otf
from .synthetic import build_test_pattern


def load_reference_image(config: InputConfig) -> np.ndarray:
    """Load or synthesize a reference image."""
    image_shape = resolve_reconstruction_shape(config.image_shape, config.reconstruction_shape)
    if config.image_source == "synthetic_test_pattern":
        return build_test_pattern(image_shape, config.intensity_scale)
    if config.image_source == "camera":
        image = skdata.camera().astype(np.float64)
        return _resize_or_crop(image, image_shape, config.intensity_scale)
    if config.image_source == "path":
        if config.image_path is None:
            raise ValueError("image_path is required when image_source='path'")
        image = tifffile.imread(Path(config.image_path)).astype(np.float64)
        return _resize_or_crop(image, image_shape, config.intensity_scale)
    raise ValueError(f"unsupported image_source: {config.image_source}")


def load_measurement_stack(path: str | Path) -> np.ndarray:
    """Load a stack of measurements from TIFF or NPY."""
    stack_path = Path(path)
    if stack_path.suffix.lower() == ".npy":
        data = np.load(stack_path)
    else:
        data = tifffile.imread(stack_path)
    if data.ndim != 3:
        raise ValueError(f"measurement stack must be 3D, got shape {data.shape}")
    return data.astype(np.float64)


def select_frames(stack: np.ndarray, frame_indices: tuple[int, ...] | None) -> np.ndarray:
    """Return a frame subset while preserving the requested order."""
    if frame_indices is None:
        return stack
    frame_count = stack.shape[0]
    _validate_frame_indices(frame_indices, frame_count)
    return stack[np.asarray(frame_indices, dtype=np.int64)]


def simulate_measurements(
    reference: np.ndarray,
    patterns: np.ndarray,
    otf: np.ndarray,
    noise: NoiseConfig,
    intensity_scale: float,
    rng: np.random.Generator,
    measurement_shape: tuple[int, int] | None = None,
    measurement_sampling: str = SAMPLING_MODE_BLOCK_AVERAGE,
) -> tuple[np.ndarray, float]:
    """Simulate LR measurements according to the paper forward model."""
    clean = _simulate_clean_measurements(
        reference,
        patterns,
        otf,
        measurement_shape,
        measurement_sampling,
    )
    noise_std = resolve_noise_std(noise, intensity_scale)
    noisy = clean + rng.normal(loc=0.0, scale=noise_std, size=clean.shape)
    return noisy.astype(np.float64), noise_std


def resolve_noise_std(noise: NoiseConfig, intensity_scale: float) -> float:
    """Resolve Gaussian noise standard deviation from config."""
    if noise.noise_std is not None:
        return noise.noise_std
    if noise.snr_db is None:
        raise ValueError("either noise_std or snr_db must be provided")
    return intensity_scale / (10.0 ** (noise.snr_db / 20.0))


def _resize_or_crop(image: np.ndarray, shape: tuple[int, int], scale: float) -> np.ndarray:
    cropped = image[: shape[0], : shape[1]]
    normalized = cropped - np.min(cropped)
    if np.max(normalized) == 0.0:
        raise ValueError("reference image is constant after normalization")
    return scale * normalized / np.max(normalized)


def _validate_frame_indices(frame_indices: tuple[int, ...], frame_count: int) -> None:
    if any(index >= frame_count for index in frame_indices):
        raise ValueError(f"frame_indices {frame_indices} exceed available frames 0..{frame_count - 1}")


def _simulate_clean_measurements(
    reference: np.ndarray,
    patterns: np.ndarray,
    otf: np.ndarray,
    measurement_shape: tuple[int, int] | None,
    measurement_sampling: str,
) -> np.ndarray:
    target_shape = reference.shape if measurement_shape is None else measurement_shape
    frames = []
    for pattern in patterns:
        blurred = convolve_with_otf(pattern * reference, otf)
        frames.append(downsample_measurement(blurred, target_shape, measurement_sampling))
    return np.stack(frames, axis=0)
