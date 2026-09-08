"""Regression tests for the calibrated COS7 high-resolution configuration."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from bayesim2012.cli import _build_otf, _build_patterns
from bayesim2012.config import load_config
from bayesim2012.constants import BACKGROUND_MODE_SHARED_SMOOTH, SAMPLING_MODE_POINT_SAMPLE


CONFIG_PATH = PACKAGE_ROOT / "configs" / "cos7_first9_hr_background.json"
EXPECTED_PHASES_RAD = np.asarray(
    (
        1.29807783846691,
        -0.8248926829831328,
        -2.90520117993519,
        2.3308307249308635,
        0.25637094119797244,
        -1.7543493192433308,
        -2.85030705443388,
        1.3673463393115424,
        -0.687898642931042,
    )
)


class Cos7ConfigurationTests(unittest.TestCase):
    """The real-data configuration must preserve the requested 2x geometry."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(CONFIG_PATH)

    def test_config_uses_2x_latent_shape_and_shared_background(self) -> None:
        self.assertEqual(self.config.input.image_shape, (512, 512))
        self.assertEqual(self.config.input.reconstruction_shape, (1024, 1024))
        self.assertEqual(self.config.input.measurement_sampling, SAMPLING_MODE_POINT_SAMPLE)
        self.assertEqual(
            self.config.input.measurement_stack_path,
            str(PACKAGE_ROOT / "data" / "3-63倍1.47-cos7-pkmito-50mWb-50ms-3.35stripe-delay5s-49-first9.tif"),
        )
        self.assertEqual(self.config.optics.otf_path, str(REPOSITORY_ROOT / "SIM4Expt" / "OTF.tif"))
        self.assertEqual(self.config.background.mode, BACKGROUND_MODE_SHARED_SMOOTH)
        self.assertEqual(self.config.sampler.max_samples, 40)
        self.assertIsNone(self.config.sampler.convergence_epsilon)
        self.assertEqual(len(self.config.optics.frame_wavevectors_px or ()), 9)
        self.assertEqual(len(self.config.optics.frame_phases_rad or ()), 9)
        np.testing.assert_allclose(self.config.optics.frame_phases_rad, EXPECTED_PHASES_RAD)

    def test_configured_patterns_and_otf_use_latent_shape(self) -> None:
        patterns = _build_patterns(self.config)
        otf = _build_otf(self.config)
        self.assertEqual(patterns.shape, (9, 1024, 1024))
        self.assertEqual(otf.shape, (1024, 1024))
        expected = 1.0 + self.config.optics.modulation_depth * np.cos(
            self.config.optics.frame_phases_rad[0]
        )
        self.assertAlmostEqual(patterns[0, 0, 0], expected)


if __name__ == "__main__":
    unittest.main()
