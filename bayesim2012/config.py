"""Configuration loading for Bayesian SIM experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .constants import (
    BACKGROUND_MODE_NONE,
    BACKGROUND_MODE_SHARED_SMOOTH,
    DEFAULT_BURN_IN,
    DEFAULT_CG_MAXITER,
    DEFAULT_CG_RTOL,
    DEFAULT_CONVERGENCE_EPSILON,
    DEFAULT_ENFORCE_NONNEGATIVE_IMAGE,
    DEFAULT_INTENSITY_SCALE,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_MODULATION_DEPTH,
    DEFAULT_PHASE_DEG,
    DEFAULT_SEED,
    DEFAULT_SHARED_PRECISION,
    DEFAULT_TEST_PATTERN_SHAPE,
    PAPER_ORIENTATIONS_DEG,
    SAMPLING_MODE_BLOCK_AVERAGE,
    SAMPLING_MODE_POINT_SAMPLE,
)


Shape2D = tuple[int, int]


@dataclass(frozen=True)
class InputConfig:
    image_source: str
    image_path: str | None
    measurement_stack_path: str | None
    image_shape: Shape2D
    reconstruction_shape: Shape2D | None
    measurement_sampling: str
    intensity_scale: float
    frame_indices: tuple[int, ...] | None


@dataclass(frozen=True)
class OpticsConfig:
    cutoff_frequency: float
    otf_path: str | None
    modulation_frequency: float
    orientations_deg: tuple[float, ...]
    modulation_depth: float
    phase_degs: tuple[float, ...]
    include_widefield: bool
    frame_wavevectors_px: tuple[tuple[float, float], ...] | None
    frame_phases_rad: tuple[float, ...] | None


@dataclass(frozen=True)
class NoiseConfig:
    snr_db: float | None
    noise_std: float | None
    seed: int
    shared_precision: bool


@dataclass(frozen=True)
class BackgroundConfig:
    """Configuration for background and the image-mean identifiability prior."""

    mode: str
    smoothness_precision: float
    signal_mean_precision: float


@dataclass(frozen=True)
class SamplerConfig:
    burn_in: int
    max_samples: int
    convergence_epsilon: float | None
    cg_rtol: float
    cg_maxiter: int
    enforce_nonnegative_image: bool


@dataclass(frozen=True)
class OutputConfig:
    directory: str


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str
    name: str
    input: InputConfig
    optics: OpticsConfig
    noise: NoiseConfig
    background: BackgroundConfig
    sampler: SamplerConfig
    output: OutputConfig


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a JSON experiment configuration."""
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    config_directory = config_path.parent
    return ExperimentConfig(
        mode=raw["experiment"]["mode"],
        name=raw["experiment"]["name"],
        input=_parse_input(raw.get("input", {}), config_directory),
        optics=_parse_optics(raw.get("optics", {}), config_directory),
        noise=_parse_noise(raw.get("noise", {})),
        background=_parse_background(raw.get("background", {})),
        sampler=_parse_sampler(raw.get("sampler", {})),
        output=_parse_output(raw.get("output", {}), config_directory),
    )


def _parse_input(raw: dict[str, Any], config_directory: Path) -> InputConfig:
    image_shape = _parse_shape(raw.get("image_shape"), DEFAULT_TEST_PATTERN_SHAPE)
    return InputConfig(
        image_source=raw.get("image_source", "synthetic_test_pattern"),
        image_path=_resolve_config_path(raw.get("image_path"), config_directory),
        measurement_stack_path=_resolve_config_path(raw.get("measurement_stack_path"), config_directory),
        image_shape=image_shape,
        reconstruction_shape=_parse_optional_shape(raw.get("reconstruction_shape")),
        measurement_sampling=_parse_measurement_sampling(raw),
        intensity_scale=float(raw.get("intensity_scale", DEFAULT_INTENSITY_SCALE)),
        frame_indices=_parse_frame_indices(raw),
    )


def _parse_measurement_sampling(raw: dict[str, Any]) -> str:
    sampling = str(raw.get("measurement_sampling", SAMPLING_MODE_BLOCK_AVERAGE))
    supported = (SAMPLING_MODE_BLOCK_AVERAGE, SAMPLING_MODE_POINT_SAMPLE)
    if sampling not in supported:
        raise ValueError(f"unsupported measurement_sampling: {sampling}")
    return sampling


def _parse_optics(raw: dict[str, Any], config_directory: Path) -> OpticsConfig:
    orientations = tuple(float(value) for value in raw.get("orientations_deg", PAPER_ORIENTATIONS_DEG))
    phase_degs = _parse_phase_degs(raw)
    frame_wavevectors_px = _parse_frame_wavevectors_px(raw)
    frame_phases_rad = _parse_frame_phases_rad(raw)
    _validate_frame_calibration(frame_wavevectors_px, frame_phases_rad)
    return OpticsConfig(
        cutoff_frequency=float(raw["cutoff_frequency"]),
        otf_path=_resolve_config_path(raw.get("otf_path"), config_directory),
        modulation_frequency=float(raw.get("modulation_frequency", raw["cutoff_frequency"])),
        orientations_deg=orientations,
        modulation_depth=float(raw.get("modulation_depth", DEFAULT_MODULATION_DEPTH)),
        phase_degs=phase_degs,
        include_widefield=bool(raw.get("include_widefield", True)),
        frame_wavevectors_px=frame_wavevectors_px,
        frame_phases_rad=frame_phases_rad,
    )


def _parse_noise(raw: dict[str, Any]) -> NoiseConfig:
    snr_db = raw.get("snr_db")
    noise_std = raw.get("noise_std")
    return NoiseConfig(
        snr_db=None if snr_db is None else float(snr_db),
        noise_std=None if noise_std is None else float(noise_std),
        seed=int(raw.get("seed", DEFAULT_SEED)),
        shared_precision=bool(raw.get("shared_precision", DEFAULT_SHARED_PRECISION)),
    )


def _parse_background(raw: dict[str, Any]) -> BackgroundConfig:
    mode = str(raw.get("mode", BACKGROUND_MODE_NONE))
    smoothness_precision = float(raw.get("smoothness_precision", 0.0))
    signal_mean_precision = float(raw.get("signal_mean_precision", 0.0))
    _validate_background(mode, smoothness_precision, signal_mean_precision)
    return BackgroundConfig(mode, smoothness_precision, signal_mean_precision)


def _validate_background(
    mode: str,
    smoothness_precision: float,
    signal_mean_precision: float,
) -> None:
    if mode not in (BACKGROUND_MODE_NONE, BACKGROUND_MODE_SHARED_SMOOTH):
        raise ValueError(f"unsupported background mode: {mode}")
    if smoothness_precision < 0.0 or signal_mean_precision < 0.0:
        raise ValueError("background precisions must be non-negative")
    if mode == BACKGROUND_MODE_NONE:
        if smoothness_precision != 0.0 or signal_mean_precision != 0.0:
            raise ValueError("background precisions require mode='shared_smooth'")
        return
    if smoothness_precision == 0.0 or signal_mean_precision == 0.0:
        raise ValueError("shared_smooth requires positive background precisions")


def _parse_sampler(raw: dict[str, Any]) -> SamplerConfig:
    convergence_epsilon = raw.get("convergence_epsilon", DEFAULT_CONVERGENCE_EPSILON)
    return SamplerConfig(
        burn_in=int(raw.get("burn_in", DEFAULT_BURN_IN)),
        max_samples=int(raw.get("max_samples", DEFAULT_MAX_SAMPLES)),
        convergence_epsilon=None if convergence_epsilon is None else float(convergence_epsilon),
        cg_rtol=float(raw.get("cg_rtol", DEFAULT_CG_RTOL)),
        cg_maxiter=int(raw.get("cg_maxiter", DEFAULT_CG_MAXITER)),
        enforce_nonnegative_image=bool(
            raw.get("enforce_nonnegative_image", DEFAULT_ENFORCE_NONNEGATIVE_IMAGE)
        ),
    )


def _parse_output(raw: dict[str, Any], config_directory: Path) -> OutputConfig:
    return OutputConfig(directory=_resolve_required_config_path(raw["directory"], config_directory))


def _parse_phase_degs(raw: dict[str, Any]) -> tuple[float, ...]:
    phase_degs = raw.get("phase_degs")
    if phase_degs is not None:
        parsed = tuple(float(value) for value in phase_degs)
        if not parsed:
            raise ValueError("phase_degs must not be empty")
        return parsed
    return (float(raw.get("phase_deg", DEFAULT_PHASE_DEG)),)


def _parse_frame_wavevectors_px(
    raw: dict[str, Any],
) -> tuple[tuple[float, float], ...] | None:
    wavevectors = raw.get("frame_wavevectors_px")
    if wavevectors is None:
        return None
    parsed = tuple((float(pair[0]), float(pair[1])) for pair in wavevectors)
    if not parsed:
        raise ValueError("frame_wavevectors_px must not be empty")
    return parsed


def _parse_frame_phases_rad(raw: dict[str, Any]) -> tuple[float, ...] | None:
    phases = raw.get("frame_phases_rad")
    if phases is None:
        return None
    parsed = tuple(float(value) for value in phases)
    if not parsed:
        raise ValueError("frame_phases_rad must not be empty")
    return parsed


def _validate_frame_calibration(
    frame_wavevectors_px: tuple[tuple[float, float], ...] | None,
    frame_phases_rad: tuple[float, ...] | None,
) -> None:
    if frame_wavevectors_px is None and frame_phases_rad is None:
        return
    if frame_wavevectors_px is None or frame_phases_rad is None:
        raise ValueError("frame_wavevectors_px and frame_phases_rad must be provided together")
    if len(frame_wavevectors_px) != len(frame_phases_rad):
        raise ValueError("frame_wavevectors_px and frame_phases_rad must have the same length")


def _parse_frame_indices(raw: dict[str, Any]) -> tuple[int, ...] | None:
    indices = raw.get("frame_indices")
    if indices is None:
        return None
    parsed = tuple(int(value) for value in indices)
    _validate_frame_indices(parsed)
    return parsed


def _validate_frame_indices(frame_indices: tuple[int, ...]) -> None:
    if not frame_indices:
        raise ValueError("frame_indices must not be empty")
    if any(index < 0 for index in frame_indices):
        raise ValueError(f"frame_indices must be non-negative, got {frame_indices}")
    if len(set(frame_indices)) != len(frame_indices):
        raise ValueError(f"frame_indices must be unique, got {frame_indices}")


def _parse_optional_shape(raw: Any) -> Shape2D | None:
    if raw is None:
        return None
    return _parse_shape(raw, DEFAULT_TEST_PATTERN_SHAPE)


def _resolve_config_path(value: Any, config_directory: Path) -> str | None:
    if value is None:
        return None
    return _resolve_required_config_path(value, config_directory)


def _resolve_required_config_path(value: Any, config_directory: Path) -> str:
    if not isinstance(value, str):
        raise ValueError(f"configuration path must be a string, got {type(value).__name__}")
    path = Path(value)
    return str(path if path.is_absolute() else (config_directory / path).resolve())


def _parse_shape(raw: Any, default: Shape2D) -> Shape2D:
    shape = default if raw is None else raw
    return (int(shape[0]), int(shape[1]))
