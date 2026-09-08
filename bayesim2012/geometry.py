"""Grid geometry helpers for latent/measurement shape conversions."""

from __future__ import annotations

import numpy as np

from .constants import SAMPLING_MODE_BLOCK_AVERAGE, SAMPLING_MODE_POINT_SAMPLE


Shape2D = tuple[int, int]


def resolve_reconstruction_shape(
    measurement_shape: Shape2D,
    reconstruction_shape: Shape2D | None,
) -> Shape2D:
    return measurement_shape if reconstruction_shape is None else reconstruction_shape


def scale_factors(source_shape: Shape2D, target_shape: Shape2D) -> tuple[int, int]:
    if source_shape == target_shape:
        return (1, 1)
    if source_shape[0] > target_shape[0] or source_shape[1] > target_shape[1]:
        raise ValueError(f"cannot scale up from {source_shape} to smaller shape {target_shape}")
    factor_y = target_shape[0] // source_shape[0]
    factor_x = target_shape[1] // source_shape[1]
    if source_shape[0] * factor_y != target_shape[0] or source_shape[1] * factor_x != target_shape[1]:
        raise ValueError(f"non-integer scale from {source_shape} to {target_shape}")
    return (factor_y, factor_x)


def block_average_downsample(image: np.ndarray, output_shape: Shape2D) -> np.ndarray:
    if image.shape == output_shape:
        return image.astype(np.float64, copy=False)
    factor_y, factor_x = scale_factors(output_shape, image.shape)
    reshaped = np.asarray(image, dtype=np.float64).reshape(
        output_shape[0], factor_y, output_shape[1], factor_x
    )
    return reshaped.mean(axis=(1, 3))


def block_average_adjoint(image: np.ndarray, output_shape: Shape2D) -> np.ndarray:
    if image.shape == output_shape:
        return image.astype(np.float64, copy=False)
    factor_y, factor_x = scale_factors(image.shape, output_shape)
    repeated = np.repeat(np.repeat(np.asarray(image, dtype=np.float64), factor_y, axis=0), factor_x, axis=1)
    return repeated / float(factor_y * factor_x)


def downsample_measurement(
    image: np.ndarray,
    output_shape: Shape2D,
    sampling_mode: str,
) -> np.ndarray:
    """Apply the configured detector-grid sampling operator."""
    if sampling_mode == SAMPLING_MODE_BLOCK_AVERAGE:
        return block_average_downsample(image, output_shape)
    if sampling_mode == SAMPLING_MODE_POINT_SAMPLE:
        return point_sample_downsample(image, output_shape)
    raise ValueError(f"unsupported measurement sampling mode: {sampling_mode}")


def measurement_adjoint(
    image: np.ndarray,
    output_shape: Shape2D,
    sampling_mode: str,
) -> np.ndarray:
    """Apply the adjoint of the configured detector-grid sampling operator."""
    if sampling_mode == SAMPLING_MODE_BLOCK_AVERAGE:
        return block_average_adjoint(image, output_shape)
    if sampling_mode == SAMPLING_MODE_POINT_SAMPLE:
        return point_sample_adjoint(image, output_shape)
    raise ValueError(f"unsupported measurement sampling mode: {sampling_mode}")


def point_sample_downsample(image: np.ndarray, output_shape: Shape2D) -> np.ndarray:
    """Sample the top-left latent-grid point of each detector pixel footprint."""
    if image.shape == output_shape:
        return image.astype(np.float64, copy=False)
    factor_y, factor_x = scale_factors(output_shape, image.shape)
    return np.asarray(image, dtype=np.float64)[::factor_y, ::factor_x]


def point_sample_adjoint(image: np.ndarray, output_shape: Shape2D) -> np.ndarray:
    """Embed detector samples at their latent-grid sampling coordinates."""
    if image.shape == output_shape:
        return image.astype(np.float64, copy=False)
    factor_y, factor_x = scale_factors(image.shape, output_shape)
    lifted = np.zeros(output_shape, dtype=np.float64)
    lifted[::factor_y, ::factor_x] = image
    return lifted


def nearest_upsample(image: np.ndarray, output_shape: Shape2D) -> np.ndarray:
    if image.shape == output_shape:
        return image.astype(np.float64, copy=False)
    factor_y, factor_x = scale_factors(image.shape, output_shape)
    return np.repeat(np.repeat(np.asarray(image, dtype=np.float64), factor_y, axis=0), factor_x, axis=1)
