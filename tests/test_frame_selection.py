"""Tests for frame subset selection."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from bayesim2012.simulation import select_frames


class FrameSelectionTests(unittest.TestCase):
    """Frame subset selection should be ordered, unique, and bounded."""

    def test_select_frames_preserves_requested_order(self) -> None:
        stack = np.arange(5 * 2 * 2, dtype=np.float64).reshape(5, 2, 2)
        selected = select_frames(stack, (3, 1, 4))
        self.assertTrue(np.array_equal(selected[0], stack[3]))
        self.assertTrue(np.array_equal(selected[1], stack[1]))
        self.assertTrue(np.array_equal(selected[2], stack[4]))

    def test_select_frames_rejects_out_of_range_indices(self) -> None:
        stack = np.zeros((3, 4, 4), dtype=np.float64)
        with self.assertRaises(ValueError):
            select_frames(stack, (0, 3))


if __name__ == "__main__":
    unittest.main()
