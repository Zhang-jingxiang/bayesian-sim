"""Gibbs sampler for Bayesian SIM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .background import BackgroundModel
from .config import SamplerConfig
from .geometry import nearest_upsample
from .operators import BayesianSIMOperators
from .optics import laplacian_energy
from .statistics import OnlineMoments, relative_mean_change


@dataclass(frozen=True)
class ReconstructionResult:
    """Outputs of one Bayesian SIM reconstruction run."""

    measurements: np.ndarray
    patterns: np.ndarray
    otf: np.ndarray
    posterior_mean: np.ndarray
    posterior_variance: np.ndarray
    background_mean: np.ndarray | None
    background_variance: np.ndarray | None
    gamma_n_chain: np.ndarray
    gamma_f_chain: np.ndarray
    mean_change_chain: np.ndarray
    burn_in: int
    total_samples: int
    posterior_samples: int
    reference: np.ndarray | None


@dataclass(frozen=True)
class _SamplerState:
    """One Gibbs state after all conditional updates."""

    image: np.ndarray
    background: np.ndarray
    gamma_n: np.ndarray
    gamma_f: float


class BayesianSIMGibbsSampler:
    """Gibbs sampler following the paper's Gaussian model."""

    def __init__(
        self,
        operators: BayesianSIMOperators,
        config: SamplerConfig,
        background: BackgroundModel,
        rng: np.random.Generator,
    ) -> None:
        self.operators = operators
        self.config = config
        self.background = background
        self.rng = rng

    def run(
        self,
        measurements: np.ndarray,
        patterns: np.ndarray,
        otf: np.ndarray,
        reference: np.ndarray | None,
        initial_image: np.ndarray | None = None,
    ) -> ReconstructionResult:
        state = self._initial_state(measurements, patterns.shape[1:], initial_image)
        gamma_n_chain: list[np.ndarray] = []
        gamma_f_chain: list[float] = []
        mean_change_chain: list[float] = []
        image_moments = OnlineMoments.zeros(patterns.shape[1:])
        background_moments = self._background_moments()
        total_samples = 0
        for iteration in range(self.config.max_samples):
            state = self._draw_state(measurements, state)
            total_samples = iteration + 1
            gamma_n_chain.append(state.gamma_n.copy())
            gamma_f_chain.append(state.gamma_f)
            if total_samples <= self.config.burn_in:
                continue
            change = _record_posterior_sample(image_moments, background_moments, state)
            if change is not None:
                mean_change_chain.append(change)
            if _has_converged(change, self.config.convergence_epsilon):
                break
        if image_moments.count == 0:
            raise RuntimeError("no posterior samples collected after burn-in")
        return ReconstructionResult(
            measurements=measurements,
            patterns=patterns,
            otf=otf,
            posterior_mean=image_moments.mean,
            posterior_variance=image_moments.variance(),
            background_mean=_moment_mean(background_moments),
            background_variance=_moment_variance(background_moments),
            gamma_n_chain=np.stack(gamma_n_chain, axis=0),
            gamma_f_chain=np.asarray(gamma_f_chain, dtype=np.float64),
            mean_change_chain=np.asarray(mean_change_chain, dtype=np.float64),
            burn_in=self.config.burn_in,
            total_samples=total_samples,
            posterior_samples=image_moments.count,
            reference=reference,
        )

    def _initial_state(
        self,
        measurements: np.ndarray,
        latent_shape: tuple[int, int],
        initial_image: np.ndarray | None,
    ) -> _SamplerState:
        return _SamplerState(
            image=_initial_image(measurements, latent_shape, initial_image),
            background=self.background.initial_state(),
            gamma_n=np.ones(measurements.shape[0], dtype=np.float64),
            gamma_f=1.0,
        )

    def _background_moments(self) -> OnlineMoments | None:
        if not self.background.enabled:
            return None
        return OnlineMoments.zeros(self.operators.measurement_shape)

    def _draw_state(self, measurements: np.ndarray, state: _SamplerState) -> _SamplerState:
        image = self._sample_image(
            measurements - state.background[None, :, :], state.gamma_n, state.gamma_f, state.image
        )
        constrained_image = _apply_image_constraints(image, self.config)
        background = self._sample_background(constrained_image, measurements, state.gamma_n)
        gamma_n = self._sample_noise_precisions(constrained_image, background, measurements)
        gamma_f = self._sample_image_precision(constrained_image)
        return _SamplerState(constrained_image, background, gamma_n, gamma_f)

    def _sample_image(
        self,
        measurements: np.ndarray,
        gamma_n: np.ndarray,
        gamma_f: float,
        current: np.ndarray,
    ) -> np.ndarray:
        mean_precision = self.background.config.signal_mean_precision
        rhs = self.operators.sampled_rhs(measurements, gamma_n, gamma_f, mean_precision, self.rng)
        return self.operators.solve_posterior_system(
            rhs, gamma_n, gamma_f, mean_precision, self.config, current
        )

    def _sample_background(
        self,
        image: np.ndarray,
        measurements: np.ndarray,
        gamma_n: np.ndarray,
    ) -> np.ndarray:
        residuals = measurements - self.operators.stacked_forward(image)
        return self.background.sample(residuals, gamma_n, self.rng)

    def _sample_noise_precisions(
        self,
        image: np.ndarray,
        background: np.ndarray,
        measurements: np.ndarray,
    ) -> np.ndarray:
        shape = measurements.shape[1] * measurements.shape[2] / 2.0
        samples = [
            self._sample_noise_precision(image, background, measurements[index], index, shape)
            for index in range(measurements.shape[0])
        ]
        return np.asarray(samples, dtype=np.float64)

    def _sample_noise_precision(
        self,
        image: np.ndarray,
        background: np.ndarray,
        frame: np.ndarray,
        frame_index: int,
        shape: float,
    ) -> float:
        residual = frame - self.operators.forward_frame(image, frame_index) - background
        energy = float(np.sum(residual * residual))
        if energy <= 0.0:
            raise ValueError("noise residual energy must be positive")
        return float(self.rng.gamma(shape=shape, scale=2.0 / energy))

    def _sample_image_precision(self, image: np.ndarray) -> float:
        energy = laplacian_energy(image)
        if energy <= 0.0:
            raise ValueError("image regularity energy must be positive")
        shape = (image.size - 1.0) / 2.0
        return float(self.rng.gamma(shape=shape, scale=2.0 / energy))


def _initial_image(
    measurements: np.ndarray,
    output_shape: tuple[int, int],
    initial_image: np.ndarray | None,
) -> np.ndarray:
    if initial_image is None:
        mean_image = np.mean(measurements, axis=0)
        return nearest_upsample(mean_image, output_shape)
    if initial_image.shape != output_shape:
        raise ValueError(
            f"initial_image shape {initial_image.shape} does not match output shape {output_shape}"
        )
    return initial_image.copy()


def _apply_image_constraints(image: np.ndarray, config: SamplerConfig) -> np.ndarray:
    if not config.enforce_nonnegative_image:
        return image
    return np.maximum(image, 0.0)


def _record_posterior_sample(
    image_moments: OnlineMoments,
    background_moments: OnlineMoments | None,
    state: _SamplerState,
) -> float | None:
    previous_mean = image_moments.update(state.image)
    if background_moments is not None:
        background_moments.update(state.background)
    if previous_mean is None:
        return None
    return relative_mean_change(previous_mean, image_moments.mean)


def _has_converged(change: float | None, convergence_epsilon: float | None) -> bool:
    """Return whether the configured posterior-mean convergence criterion is met."""
    return change is not None and convergence_epsilon is not None and change <= convergence_epsilon


def _moment_mean(moments: OnlineMoments | None) -> np.ndarray | None:
    return None if moments is None else moments.mean


def _moment_variance(moments: OnlineMoments | None) -> np.ndarray | None:
    return None if moments is None else moments.variance()
