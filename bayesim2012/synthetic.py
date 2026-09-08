"""Synthetic reference image generators."""

from __future__ import annotations

import numpy as np


def build_test_pattern(shape: tuple[int, int], intensity_scale: float) -> np.ndarray:
    """Create a paper-like synthetic test pattern with mixed frequencies."""
    star = _siemens_star(shape)
    blocks = _corner_blocks(shape)
    pattern = np.maximum(star, blocks)
    return intensity_scale * pattern / np.max(pattern)


def _siemens_star(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    y = np.linspace(-1.0, 1.0, rows)
    x = np.linspace(-1.0, 1.0, cols)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    radius = np.sqrt(xx**2 + yy**2)
    angle = np.arctan2(yy, xx)
    spokes = np.sign(np.cos(36.0 * angle))
    disk = radius <= 0.75
    star = np.zeros(shape, dtype=np.float64)
    star[disk] = 0.5 * (spokes[disk] + 1.0)
    return star


def _corner_blocks(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    block = np.zeros(shape, dtype=np.float64)
    size = max(8, min(rows, cols) // 12)
    margin = max(4, size // 3)
    positions = [
        (margin, margin),
        (margin, cols - margin - size),
        (rows - margin - size, margin),
        (rows - margin - size, cols - margin - size),
    ]
    for top, left in positions:
        _write_checkerboard(block, top, left, size)
    return block


def _write_checkerboard(image: np.ndarray, top: int, left: int, size: int) -> None:
    half = size // 2
    image[top : top + size, left : left + size] = 0.25
    image[top : top + half, left : left + half] = 1.0
    image[top + half : top + size, left + half : left + size] = 1.0
