"""Tests for configurable SIM pattern stacks and metric summaries."""

from __future__ import annotations

from pathlib import Path
import tempfile
import sys
import unittest

import numpy as np
import tifffile

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from bayesim2012.gibbs import ReconstructionResult
from bayesim2012.optics import load_otf
from bayesim2012.outputs import build_summary
from bayesim2012.patterns import (
    generate_calibrated_pattern,
    generate_calibrated_patterns,
    generate_phase_shifted_patterns,
)


class PatternGenerationTests(unittest.TestCase):
    """Smoke tests for phase-shifted SIM stacks."""

    def test_three_orientations_three_phases_make_nine_frames(self) -> None:
        patterns = generate_phase_shifted_patterns(
            shape=(32, 32),
            modulation_frequency=0.12,
            orientations_deg=(-60.0, 0.0, 60.0),
            modulation_depth=1.0,
            phase_degs=(0.0, 120.0, 240.0),
            include_widefield=False,
        )
        self.assertEqual(patterns.shape, (9, 32, 32))

    def test_calibrated_frame_patterns_make_nine_frames(self) -> None:
        patterns = generate_calibrated_patterns(
            shape=(32, 32),
            frame_wavevectors_px=((10.0, 0.0),) * 9,
            frame_phases_rad=tuple(float(index) for index in range(9)),
            modulation_depth=0.32,
            include_widefield=False,
        )
        self.assertEqual(patterns.shape, (9, 32, 32))

    def test_calibrated_pattern_uses_image_origin_for_phase(self) -> None:
        phase = 0.7
        modulation_depth = 0.32
        pattern = generate_calibrated_pattern((10, 20), (4.0, -3.0), phase, modulation_depth)
        self.assertAlmostEqual(pattern[0, 0], 1.0 + modulation_depth * np.cos(phase))
        expected_phase = 2.0 * np.pi * (4.0 * 5.0 / 20.0 - 3.0 * 3.0 / 10.0) + phase
        self.assertAlmostEqual(pattern[3, 5], 1.0 + modulation_depth * np.cos(expected_phase))

    def test_structured_only_summary_reports_similarity_without_isnr(self) -> None:
        reference = np.linspace(0.0, 255.0, 64, dtype=np.float64).reshape(8, 8)
        estimate = reference * 0.95
        patterns = generate_phase_shifted_patterns(
            shape=(8, 8),
            modulation_frequency=0.12,
            orientations_deg=(-60.0, 0.0, 60.0),
            modulation_depth=1.0,
            phase_degs=(0.0, 120.0, 240.0),
            include_widefield=False,
        )
        result = ReconstructionResult(
            measurements=np.ones((9, 8, 8), dtype=np.float64),
            patterns=patterns,
            otf=np.ones((8, 8), dtype=np.float64),
            posterior_mean=estimate,
            posterior_variance=np.ones((8, 8), dtype=np.float64),
            background_mean=None,
            background_variance=None,
            gamma_n_chain=np.ones((2, 9), dtype=np.float64),
            gamma_f_chain=np.ones(2, dtype=np.float64),
            mean_change_chain=np.ones(1, dtype=np.float64),
            burn_in=1,
            total_samples=2,
            posterior_samples=1,
            reference=reference,
        )
        summary = build_summary(result)
        self.assertEqual(summary["measurement_shape"], [8, 8])
        self.assertEqual(summary["posterior_shape"], [8, 8])
        self.assertIn("ssim", summary)
        self.assertIn("psnr", summary)
        self.assertNotIn("isnr", summary)

    def test_load_otf_normalizes_and_aligns_centered_input(self) -> None:
        centered_otf = np.zeros((8, 8), dtype=np.uint16)
        centered_otf[4, 4] = 10
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "otf.tif"
            tifffile.imwrite(path, centered_otf)
            otf = load_otf(path, (8, 8))
        self.assertAlmostEqual(float(otf[0, 0]), 1.0)
        self.assertAlmostEqual(float(np.max(otf)), 1.0)


if __name__ == "__main__":
    unittest.main()
