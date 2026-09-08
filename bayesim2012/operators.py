"""Matrix-free forward and posterior operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from .config import SamplerConfig
from .constants import SAMPLING_MODE_BLOCK_AVERAGE
from .geometry import Shape2D, downsample_measurement, measurement_adjoint
from .optics import convolve_with_otf, correlate_with_otf, laplacian, laplacian_spectrum


@dataclass(frozen=True)
class BayesianSIMOperators:
    """Forward, adjoint, and posterior precision operators."""

    patterns: np.ndarray
    otf: np.ndarray
    measurement_shape: Shape2D
    measurement_sampling: str = SAMPLING_MODE_BLOCK_AVERAGE

    def forward_frame(self, image: np.ndarray, frame_index: int) -> np.ndarray:
        pattern = self.patterns[frame_index]
        blurred = convolve_with_otf(pattern * image, self.otf)
        return downsample_measurement(blurred, self.measurement_shape, self.measurement_sampling)

    def adjoint_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        lifted = measurement_adjoint(frame, self.patterns.shape[1:], self.measurement_sampling)
        correlated = correlate_with_otf(lifted, self.otf)
        return self.patterns[frame_index] * correlated

    def stacked_forward(self, image: np.ndarray) -> np.ndarray:
        frames = [self.forward_frame(image, index) for index in range(self.patterns.shape[0])]
        return np.stack(frames, axis=0)

    def posterior_mean_rhs(self, measurements: np.ndarray, gamma_n: np.ndarray) -> np.ndarray:
        rhs_terms = [gamma_n[index] * self.adjoint_frame(measurements[index], index) for index in range(len(gamma_n))]
        return np.sum(rhs_terms, axis=0)

    def sampled_rhs(
        self,
        measurements: np.ndarray,
        gamma_n: np.ndarray,
        gamma_f: float,
        signal_mean_precision: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        rhs = self.posterior_mean_rhs(_perturb_measurements(measurements, gamma_n, rng), gamma_n)
        return rhs + _prior_perturbation(
            self.patterns.shape[1:], gamma_f, signal_mean_precision, rng
        )

    def apply_posterior_precision(
        self,
        image: np.ndarray,
        gamma_n: np.ndarray,
        gamma_f: float,
        signal_mean_precision: float,
    ) -> np.ndarray:
        data_term = _sum_data_terms(self, image, gamma_n)
        prior_term = gamma_f * laplacian(laplacian(image))
        if signal_mean_precision != 0.0:
            prior_term += signal_mean_precision * _mean_projection(image)
        return data_term + prior_term

    def solve_posterior_system(
        self,
        rhs: np.ndarray,
        gamma_n: np.ndarray,
        gamma_f: float,
        signal_mean_precision: float,
        sampler: SamplerConfig,
        x0: np.ndarray | None,
    ) -> np.ndarray:
        linear_operator = _build_linear_operator(
            self, rhs.shape, gamma_n, gamma_f, signal_mean_precision
        )
        preconditioner = _build_preconditioner(
            self, rhs.shape, gamma_n, gamma_f, signal_mean_precision
        )
        start = None if x0 is None else x0.ravel()
        solution, info = cg(
            linear_operator,
            rhs.ravel(),
            M=preconditioner,
            x0=start,
            maxiter=sampler.cg_maxiter,
            rtol=sampler.cg_rtol,
        )
        if info != 0:
            raise RuntimeError(f"conjugate gradient failed with info={info}")
        return solution.reshape(rhs.shape)


def _build_linear_operator(
    operators: BayesianSIMOperators,
    shape: tuple[int, int],
    gamma_n: np.ndarray,
    gamma_f: float,
    signal_mean_precision: float,
) -> LinearOperator:
    size = shape[0] * shape[1]

    def matvec(vector: np.ndarray) -> np.ndarray:
        image = vector.reshape(shape)
        applied = operators.apply_posterior_precision(
            image, gamma_n, gamma_f, signal_mean_precision
        )
        return applied.ravel()

    return LinearOperator((size, size), matvec=matvec, dtype=np.float64)


def _build_preconditioner(
    operators: BayesianSIMOperators,
    shape: tuple[int, int],
    gamma_n: np.ndarray,
    gamma_f: float,
    signal_mean_precision: float,
) -> LinearOperator:
    size = shape[0] * shape[1]
    spectrum = _preconditioner_spectrum(
        operators, shape, gamma_n, gamma_f, signal_mean_precision
    )
    _validate_preconditioner_spectrum(spectrum)

    def matvec(vector: np.ndarray) -> np.ndarray:
        image = vector.reshape(shape)
        filtered = np.fft.ifft2(np.fft.fft2(image) / spectrum).real
        return filtered.ravel()

    return LinearOperator((size, size), matvec=matvec, dtype=np.float64)


def _perturb_measurements(
    measurements: np.ndarray,
    gamma_n: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    noisy_frames = []
    for index, frame in enumerate(measurements):
        std = 1.0 / np.sqrt(gamma_n[index])
        noisy_frames.append(frame + rng.normal(loc=0.0, scale=std, size=frame.shape))
    return np.stack(noisy_frames, axis=0)


def _prior_perturbation(
    shape: tuple[int, int],
    gamma_f: float,
    signal_mean_precision: float,
    rng: np.random.Generator,
) -> np.ndarray:
    white = rng.normal(loc=0.0, scale=1.0, size=shape)
    perturbation = np.sqrt(gamma_f) * laplacian(white)
    if signal_mean_precision == 0.0:
        return perturbation
    return perturbation + np.sqrt(signal_mean_precision) * _mean_projection(white)


def _sum_data_terms(
    operators: BayesianSIMOperators,
    image: np.ndarray,
    gamma_n: np.ndarray,
) -> np.ndarray:
    terms = []
    for index, gamma_value in enumerate(gamma_n):
        forward = operators.forward_frame(image, index)
        terms.append(gamma_value * operators.adjoint_frame(forward, index))
    return np.sum(terms, axis=0)


def _preconditioner_spectrum(
    operators: BayesianSIMOperators,
    shape: tuple[int, int],
    gamma_n: np.ndarray,
    gamma_f: float,
    signal_mean_precision: float,
) -> np.ndarray:
    otf_power = np.abs(operators.otf) ** 2
    mean_pattern_power = np.mean(operators.patterns**2, axis=(1, 2))
    data_term = np.sum(
        gamma_n[:, None, None] * mean_pattern_power[:, None, None] * otf_power[None, :, :],
        axis=0,
    )
    prior_term = gamma_f * (laplacian_spectrum(shape) ** 2)
    prior_term[0, 0] += signal_mean_precision
    return data_term + prior_term


def _validate_preconditioner_spectrum(spectrum: np.ndarray) -> None:
    minimum = float(np.min(spectrum))
    if minimum <= 0.0:
        raise ValueError(f"preconditioner spectrum must stay positive, got minimum={minimum}")


def _mean_projection(image: np.ndarray) -> np.ndarray:
    """Project an image onto its spatially constant component."""
    return np.full(image.shape, float(np.mean(image)), dtype=np.float64)
