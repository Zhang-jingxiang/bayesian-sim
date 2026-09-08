"""Structured illumination pattern generation."""

from __future__ import annotations

import numpy as np


def generate_four_frame_patterns(
    shape: tuple[int, int],
    modulation_frequency: float,
    orientations_deg: tuple[float, ...],
    modulation_depth: float,
    phase_deg: float,
) -> np.ndarray:
    """Backward-compatible wrapper for the paper's 4-frame setting."""
    return generate_phase_shifted_patterns(
        shape=shape,
        modulation_frequency=modulation_frequency,
        orientations_deg=orientations_deg,
        modulation_depth=modulation_depth,
        phase_degs=(phase_deg,),
        include_widefield=True,
    )


def generate_phase_shifted_patterns(
    shape: tuple[int, int],
    modulation_frequency: float,
    orientations_deg: tuple[float, ...],
    modulation_depth: float,
    phase_degs: tuple[float, ...],
    include_widefield: bool,
) -> np.ndarray:
    """Generate a SIM stack from orientation and phase lists."""
    patterns = []
    if include_widefield:
        patterns.append(np.ones(shape, dtype=np.float64))
    for orientation in orientations_deg:
        patterns.extend(
            _patterns_for_orientation(shape, modulation_frequency, orientation, modulation_depth, phase_degs)
        )
    return np.stack(patterns, axis=0)


def generate_calibrated_patterns(
    shape: tuple[int, int],
    frame_wavevectors_px: tuple[tuple[float, float], ...],
    frame_phases_rad: tuple[float, ...],
    modulation_depth: float,
    include_widefield: bool,
) -> np.ndarray:
    """Generate a stack using per-frame calibrated wavevectors and phases."""
    patterns = []
    if include_widefield:
        patterns.append(np.ones(shape, dtype=np.float64))
    patterns.extend(
        _patterns_for_calibrated_frames(shape, frame_wavevectors_px, frame_phases_rad, modulation_depth)
    )
    return np.stack(patterns, axis=0)


def generate_pattern(
    shape: tuple[int, int],
    modulation_frequency: float,
    orientation_deg: float,
    modulation_depth: float,
    phase_deg: float,
) -> np.ndarray:
    """Generate one cosine illumination pattern."""
    y_coords, x_coords = centered_coordinates(shape)
    angle_rad = np.deg2rad(orientation_deg)
    phase_rad = np.deg2rad(phase_deg)
    carrier = modulation_frequency * (np.cos(angle_rad) * x_coords + np.sin(angle_rad) * y_coords)
    pattern = 1.0 + modulation_depth * np.cos(2.0 * np.pi * carrier + phase_rad)
    return pattern.astype(np.float64)


def generate_calibrated_pattern(
    shape: tuple[int, int],
    wavevector_px: tuple[float, float],
    phase_rad: float,
    modulation_depth: float,
) -> np.ndarray:
    """Generate one pattern from field-of-view carrier cycles and origin-referenced phase."""
    rows, cols = shape
    y_coords, x_coords = np.indices(shape, dtype=np.float64)
    kx_px, ky_px = wavevector_px
    carrier = (kx_px / cols) * x_coords + (ky_px / rows) * y_coords
    pattern = 1.0 + modulation_depth * np.cos(2.0 * np.pi * carrier + phase_rad)
    return pattern.astype(np.float64)


def centered_coordinates(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Create centered pixel coordinates in pixel units."""
    rows, cols = shape
    y = np.arange(rows, dtype=np.float64) - rows / 2.0
    x = np.arange(cols, dtype=np.float64) - cols / 2.0
    return np.meshgrid(y, x, indexing="ij")


def _patterns_for_orientation(
    shape: tuple[int, int],
    modulation_frequency: float,
    orientation_deg: float,
    modulation_depth: float,
    phase_degs: tuple[float, ...],
) -> list[np.ndarray]:
    return [
        generate_pattern(shape, modulation_frequency, orientation_deg, modulation_depth, phase_deg)
        for phase_deg in phase_degs
    ]


def _patterns_for_calibrated_frames(
    shape: tuple[int, int],
    frame_wavevectors_px: tuple[tuple[float, float], ...],
    frame_phases_rad: tuple[float, ...],
    modulation_depth: float,
) -> list[np.ndarray]:
    return [
        generate_calibrated_pattern(shape, wavevector_px, phase_rad, modulation_depth)
        for wavevector_px, phase_rad in zip(frame_wavevectors_px, frame_phases_rad)
    ]
