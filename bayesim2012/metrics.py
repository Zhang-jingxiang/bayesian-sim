"""Reconstruction metrics used by the paper."""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def isnr(reference: np.ndarray, estimate: np.ndarray, baseline: np.ndarray) -> float:
    """Compute improved signal-to-noise ratio."""
    baseline_error = np.sum((reference - baseline) ** 2)
    estimate_error = np.sum((reference - estimate) ** 2)
    if estimate_error == 0.0:
        raise ValueError("estimate_error is zero; ISNR is infinite")
    return float(10.0 * np.log10(baseline_error / estimate_error))


def ssim(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute structural similarity on the intensity scale of the input."""
    data_range = float(np.max(reference) - np.min(reference))
    if data_range == 0.0:
        raise ValueError("reference has zero dynamic range")
    return float(structural_similarity(reference, estimate, data_range=data_range))


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute PSNR on the reference dynamic range."""
    data_range = float(np.max(reference) - np.min(reference))
    if data_range == 0.0:
        raise ValueError("reference has zero dynamic range")
    return float(peak_signal_noise_ratio(reference, estimate, data_range=data_range))


def fit_affine_intensity(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Fit the least-squares intensity map from an estimate to a reference."""
    _validate_metric_shapes(reference, estimate)
    estimate_mean = float(np.mean(estimate))
    reference_mean = float(np.mean(reference))
    centered_estimate = estimate - estimate_mean
    denominator = float(np.sum(centered_estimate * centered_estimate))
    if denominator == 0.0:
        raise ValueError("estimate is constant; affine intensity fitting is undefined")
    scale = float(np.sum(centered_estimate * (reference - reference_mean)) / denominator)
    offset = reference_mean - scale * estimate_mean
    return scale * estimate + offset, scale, offset


def affine_reference_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    """Evaluate structural agreement after a global affine intensity fit."""
    matched, scale, offset = fit_affine_intensity(reference, estimate)
    residual = reference - matched
    reference_centered = reference - np.mean(reference)
    estimate_centered = estimate - np.mean(estimate)
    correlation_denominator = float(
        np.sqrt(np.sum(reference_centered**2) * np.sum(estimate_centered**2))
    )
    if correlation_denominator == 0.0:
        raise ValueError("Pearson correlation is undefined for a constant image")
    return {
        "affine_scale": scale,
        "affine_offset": offset,
        "affine_rmse": float(np.sqrt(np.mean(residual * residual))),
        "affine_ssim": ssim(reference, matched),
        "affine_psnr": psnr(reference, matched),
        "pearson_correlation": float(
            np.sum(reference_centered * estimate_centered) / correlation_denominator
        ),
    }


def _validate_metric_shapes(reference: np.ndarray, estimate: np.ndarray) -> None:
    if reference.shape != estimate.shape:
        raise ValueError(
            f"reference shape {reference.shape} does not match estimate shape {estimate.shape}"
        )
