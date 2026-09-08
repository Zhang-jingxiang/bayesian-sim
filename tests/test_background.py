"""Tests for the shared smooth background conditional update."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from bayesim2012.background import BackgroundModel
from bayesim2012.config import BackgroundConfig
from bayesim2012.constants import BACKGROUND_MODE_NONE, BACKGROUND_MODE_SHARED_SMOOTH


class BackgroundModelTests(unittest.TestCase):
    """The background update must separate a frame-shared DC component."""

    def test_shared_smooth_model_recovers_constant_residual(self) -> None:
        shape = (8, 8)
        residual_value = 17.0
        model = BackgroundModel(_shared_smooth_config(), shape)
        residuals = np.full((3, *shape), residual_value, dtype=np.float64)
        gamma_n = np.asarray([0.5, 1.0, 2.0], dtype=np.float64)
        mean = model.conditional_mean(residuals, gamma_n)
        self.assertTrue(np.allclose(mean, residual_value, atol=1e-12))

    def test_disabled_model_returns_zero_background(self) -> None:
        shape = (8, 8)
        model = BackgroundModel(_disabled_config(), shape)
        residuals = np.ones((2, *shape), dtype=np.float64)
        gamma_n = np.ones(2, dtype=np.float64)
        self.assertTrue(np.array_equal(model.conditional_mean(residuals, gamma_n), np.zeros(shape)))


def _shared_smooth_config() -> BackgroundConfig:
    return BackgroundConfig(
        mode=BACKGROUND_MODE_SHARED_SMOOTH,
        smoothness_precision=0.03,
        signal_mean_precision=0.0007,
    )


def _disabled_config() -> BackgroundConfig:
    return BackgroundConfig(
        mode=BACKGROUND_MODE_NONE,
        smoothness_precision=0.0,
        signal_mean_precision=0.0,
    )


if __name__ == "__main__":
    unittest.main()
