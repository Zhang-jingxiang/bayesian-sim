"""Unit tests for posterior building blocks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import unittest

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from bayesim2012.background import BackgroundModel
from bayesim2012.config import BackgroundConfig, SamplerConfig
from bayesim2012.constants import (
    BACKGROUND_MODE_NONE,
    DEFAULT_CG_MAXITER,
    SAMPLING_MODE_POINT_SAMPLE,
)
from bayesim2012.gibbs import BayesianSIMGibbsSampler, _apply_image_constraints, _initial_image
from bayesim2012.operators import BayesianSIMOperators
from bayesim2012.optics import airy_otf
from bayesim2012.patterns import generate_four_frame_patterns
from bayesim2012.simulation import simulate_measurements
from bayesim2012.synthetic import build_test_pattern


class PosteriorTests(unittest.TestCase):
    """Posterior and operator smoke tests."""

    def setUp(self) -> None:
        self.shape = (65, 65)
        self.rng = np.random.default_rng(7)
        self.reference = build_test_pattern(self.shape, 255.0)
        self.patterns = generate_four_frame_patterns(self.shape, 0.12, (-30.0, 30.0, 90.0), 1.0, 0.0)
        self.otf = airy_otf(self.shape, 0.12)
        self.measurements, _ = simulate_measurements(
            self.reference,
            self.patterns,
            self.otf,
            _NoiseStub(),
            255.0,
            self.rng,
        )
        self.operators = BayesianSIMOperators(self.patterns, self.otf, self.shape)

    def test_forward_adjoint_consistency(self) -> None:
        image = self.rng.normal(size=self.shape)
        frame = self.rng.normal(size=self.shape)
        forward = self.operators.forward_frame(image, 1)
        adjoint = self.operators.adjoint_frame(frame, 1)
        lhs = float(np.sum(forward * frame))
        rhs = float(np.sum(image * adjoint))
        self.assertAlmostEqual(lhs, rhs, places=8)

    def test_gibbs_sampler_outputs_finite_arrays(self) -> None:
        sampler = BayesianSIMGibbsSampler(
            self.operators,
            _sampler_config(),
            _disabled_background(self.shape),
            np.random.default_rng(11),
        )
        result = sampler.run(self.measurements, self.patterns, self.otf, self.reference)
        self.assertEqual(result.posterior_mean.shape, self.shape)
        self.assertEqual(result.posterior_variance.shape, self.shape)
        self.assertIsNone(result.background_mean)
        self.assertIsNone(result.background_variance)
        self.assertTrue(np.all(np.isfinite(result.posterior_mean)))
        self.assertTrue(np.all(result.posterior_variance >= 0.0))
        self.assertGreater(result.posterior_samples, 0)

    def test_none_convergence_threshold_runs_all_configured_samples(self) -> None:
        config = _sampler_config(convergence_epsilon=None)
        sampler = BayesianSIMGibbsSampler(
            self.operators,
            config,
            _disabled_background(self.shape),
            np.random.default_rng(11),
        )
        result = sampler.run(self.measurements, self.patterns, self.otf, self.reference)
        self.assertEqual(result.total_samples, config.max_samples)

    def test_superres_forward_adjoint_consistency(self) -> None:
        measurement_shape = (32, 32)
        latent_shape = (64, 64)
        reference = build_test_pattern(latent_shape, 255.0)
        patterns = generate_four_frame_patterns(latent_shape, 0.08, (-30.0, 30.0, 90.0), 1.0, 0.0)
        otf = airy_otf(latent_shape, 0.08 / 2.0)
        measurements, _ = simulate_measurements(
            reference,
            patterns,
            otf,
            _NoiseStub(),
            255.0,
            self.rng,
            measurement_shape=measurement_shape,
        )
        operators = BayesianSIMOperators(patterns, otf, measurement_shape)
        image = self.rng.normal(size=latent_shape)
        frame = self.rng.normal(size=measurement_shape)
        forward = operators.forward_frame(image, 1)
        adjoint = operators.adjoint_frame(frame, 1)
        lhs = float(np.sum(forward * frame))
        rhs = float(np.sum(image * adjoint))
        self.assertEqual(measurements.shape, (4, 32, 32))
        self.assertAlmostEqual(lhs, rhs, places=8)

    def test_superres_initial_image_uses_nearest_upsample(self) -> None:
        latent_shape = (32, 32)
        measurements = np.ones((4, 16, 16), dtype=np.float64)
        initial = _initial_image(measurements, latent_shape, None)
        self.assertEqual(initial.shape, latent_shape)
        self.assertTrue(np.allclose(initial[::2, ::2], 1.0))

    def test_point_sample_forward_adjoint_consistency(self) -> None:
        measurement_shape = (32, 32)
        latent_shape = (64, 64)
        patterns = generate_four_frame_patterns(latent_shape, 0.08, (-30.0, 30.0, 90.0), 1.0, 0.0)
        otf = airy_otf(latent_shape, 0.04)
        operators = BayesianSIMOperators(
            patterns,
            otf,
            measurement_shape,
            measurement_sampling=SAMPLING_MODE_POINT_SAMPLE,
        )
        image = self.rng.normal(size=latent_shape)
        frame = self.rng.normal(size=measurement_shape)
        lhs = float(np.sum(operators.forward_frame(image, 1) * frame))
        rhs = float(np.sum(image * operators.adjoint_frame(frame, 1)))
        self.assertAlmostEqual(lhs, rhs, places=8)

    def test_nonnegative_constraint_is_explicit(self) -> None:
        image = np.asarray([[-1.0, 2.0]], dtype=np.float64)
        constrained = _apply_image_constraints(image, _sampler_config(enforce_nonnegative=True))
        unconstrained = _apply_image_constraints(image, _sampler_config())
        self.assertTrue(np.array_equal(constrained, np.asarray([[0.0, 2.0]])))
        self.assertTrue(np.array_equal(unconstrained, image))

    def test_signal_mean_prior_regularizes_only_the_dc_component(self) -> None:
        gamma_n = np.zeros(self.patterns.shape[0], dtype=np.float64)
        precision = 3.0
        zero_mean = self.rng.normal(size=self.shape)
        zero_mean -= np.mean(zero_mean)
        constant = np.ones(self.shape, dtype=np.float64)
        zero_mean_term = self.operators.apply_posterior_precision(
            zero_mean, gamma_n, 0.0, precision
        )
        constant_term = self.operators.apply_posterior_precision(
            constant, gamma_n, 0.0, precision
        )
        self.assertTrue(np.allclose(zero_mean_term, 0.0, atol=1e-12))
        self.assertTrue(np.allclose(constant_term, precision))


@dataclass(frozen=True)
class _NoiseStub:
    snr_db: float | None = None
    noise_std: float | None = 1.0
    seed: int = 7
    shared_precision: bool = False


def _sampler_config(
    enforce_nonnegative: bool = False,
    convergence_epsilon: float | None = 1e-3,
) -> SamplerConfig:
    return SamplerConfig(
        burn_in=2,
        max_samples=6,
        convergence_epsilon=convergence_epsilon,
        cg_rtol=1e-6,
        cg_maxiter=DEFAULT_CG_MAXITER,
        enforce_nonnegative_image=enforce_nonnegative,
    )


def _disabled_background(shape: tuple[int, int]) -> BackgroundModel:
    config = BackgroundConfig(
        mode=BACKGROUND_MODE_NONE,
        smoothness_precision=0.0,
        signal_mean_precision=0.0,
    )
    return BackgroundModel(config, shape)
