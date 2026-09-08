"""Conditional Gaussian updates for a shared smooth background field."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import BackgroundConfig
from .constants import BACKGROUND_MODE_SHARED_SMOOTH
from .geometry import Shape2D
from .optics import laplacian, laplacian_spectrum


@dataclass(frozen=True)
class BackgroundModel:
    """Estimate the non-structured component common to all measurement frames."""

    config: BackgroundConfig
    measurement_shape: Shape2D

    @property
    def enabled(self) -> bool:
        """Whether this model contributes a background field."""
        return self.config.mode == BACKGROUND_MODE_SHARED_SMOOTH

    def initial_state(self) -> np.ndarray:
        """Return the neutral background state."""
        return np.zeros(self.measurement_shape, dtype=np.float64)

    def conditional_mean(self, residuals: np.ndarray, gamma_n: np.ndarray) -> np.ndarray:
        """Return E[b | f, gamma_n, g] for the current frame residuals."""
        if not self.enabled:
            return self.initial_state()
        _validate_residuals(residuals, gamma_n, self.measurement_shape)
        weighted_residual = np.sum(gamma_n[:, None, None] * residuals, axis=0)
        return _solve_background_system(weighted_residual, self.precision_spectrum(gamma_n))

    def sample(
        self,
        residuals: np.ndarray,
        gamma_n: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw b from its conditional Gaussian law."""
        mean = self.conditional_mean(residuals, gamma_n)
        if not self.enabled:
            return mean
        perturbation = _background_perturbation(
            self.measurement_shape,
            float(np.sum(gamma_n)),
            self.config.smoothness_precision,
            rng,
        )
        return mean + _solve_background_system(perturbation, self.precision_spectrum(gamma_n))

    def precision_spectrum(self, gamma_n: np.ndarray) -> np.ndarray:
        """Return the Fourier-diagonal conditional precision of the background."""
        data_precision = float(np.sum(gamma_n))
        if data_precision <= 0.0:
            raise ValueError("sum of noise precisions must be positive")
        return data_precision + self.config.smoothness_precision * laplacian_spectrum(
            self.measurement_shape
        ) ** 2


def _validate_residuals(
    residuals: np.ndarray,
    gamma_n: np.ndarray,
    measurement_shape: Shape2D,
) -> None:
    if residuals.shape[1:] != measurement_shape:
        raise ValueError(f"background residual shape {residuals.shape[1:]} does not match {measurement_shape}")
    if residuals.shape[0] != gamma_n.size:
        raise ValueError("background residual frame count does not match noise precisions")


def _solve_background_system(rhs: np.ndarray, precision_spectrum: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(np.fft.fft2(rhs) / precision_spectrum).real


def _background_perturbation(
    shape: Shape2D,
    data_precision: float,
    smoothness_precision: float,
    rng: np.random.Generator,
) -> np.ndarray:
    data_noise = np.sqrt(data_precision) * rng.normal(size=shape)
    prior_noise = np.sqrt(smoothness_precision) * laplacian(rng.normal(size=shape))
    return data_noise + prior_noise
