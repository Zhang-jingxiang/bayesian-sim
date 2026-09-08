"""Tests for GT evaluation helpers."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from bayesim2012.metrics import affine_reference_metrics, fit_affine_intensity


class AffineReferenceMetricTests(unittest.TestCase):
    """Intensity calibration must not obscure spatial agreement metrics."""

    def test_affine_fit_recovers_exact_reference(self) -> None:
        estimate = np.arange(64, dtype=np.float64).reshape(8, 8)
        reference = 2.5 * estimate + 17.0
        matched, scale, offset = fit_affine_intensity(reference, estimate)
        self.assertTrue(np.allclose(matched, reference))
        self.assertAlmostEqual(scale, 2.5)
        self.assertAlmostEqual(offset, 17.0)

    def test_affine_metrics_report_near_perfect_structural_agreement(self) -> None:
        estimate = np.arange(64, dtype=np.float64).reshape(8, 8)
        reference = 2.5 * estimate + 17.0
        reference[0, 0] += 1.0
        metrics = affine_reference_metrics(reference, estimate)
        self.assertGreater(metrics["affine_rmse"], 0.0)
        self.assertLess(metrics["affine_rmse"], 1.0)
        self.assertGreater(metrics["affine_ssim"], 0.99)
        self.assertGreater(metrics["pearson_correlation"], 0.99)

    def test_affine_fit_rejects_mismatched_shapes(self) -> None:
        with self.assertRaises(ValueError):
            fit_affine_intensity(np.zeros((8, 8)), np.zeros((4, 4)))


if __name__ == "__main__":
    unittest.main()
