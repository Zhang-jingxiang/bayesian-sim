"""CLI entrypoints for Bayesian SIM experiments."""

from __future__ import annotations

import argparse

import numpy as np

from .background import BackgroundModel
from .config import ExperimentConfig, load_config
from .geometry import resolve_reconstruction_shape
from .gibbs import BayesianSIMGibbsSampler
from .operators import BayesianSIMOperators
from .optics import airy_otf, latent_cutoff_frequency, load_otf
from .outputs import build_summary, save_run_outputs
from .patterns import generate_calibrated_patterns, generate_phase_shifted_patterns
from .simulation import load_measurement_stack, load_reference_image, select_frames, simulate_measurements


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    result = run_experiment(config, args.output, args.dry_run)
    if result is None:
        return 0
    print(build_summary(result))
    return 0


def run_experiment(
    config: ExperimentConfig,
    output_override: str | None,
    dry_run: bool,
    initial_image: np.ndarray | None = None,
):
    """Run one configured experiment."""
    _validate_supported_options(config)
    measurement_rng, sampler_rng = _build_rngs(config.noise.seed)
    patterns = _select_pattern_frames(_build_patterns(config), config)
    otf = _build_otf(config)
    reference, measurements = _prepare_data(config, patterns, otf, measurement_rng)
    _validate_measurements(measurements, patterns, config)
    if dry_run:
        print(_dry_run_message(config, measurements))
        return None
    operators = BayesianSIMOperators(
        patterns=patterns,
        otf=otf,
        measurement_shape=measurements.shape[1:],
        measurement_sampling=config.input.measurement_sampling,
    )
    background = BackgroundModel(config.background, measurements.shape[1:])
    sampler = BayesianSIMGibbsSampler(operators, config.sampler, background, sampler_rng)
    result = sampler.run(measurements, patterns, otf, reference, initial_image=initial_image)
    output_dir = output_override or config.output.directory
    save_run_outputs(result, output_dir)
    return result


def _prepare_data(
    config: ExperimentConfig,
    patterns: np.ndarray,
    otf: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, np.ndarray]:
    if config.mode == "simulation":
        reference = load_reference_image(config.input)
        measurements, _ = simulate_measurements(
            reference,
            patterns,
            otf,
            config.noise,
            config.input.intensity_scale,
            rng,
            measurement_shape=config.input.image_shape,
            measurement_sampling=config.input.measurement_sampling,
        )
        return reference, measurements
    if config.mode == "reconstruction":
        if config.input.measurement_stack_path is None:
            raise ValueError("measurement_stack_path is required in reconstruction mode")
        measurements = load_measurement_stack(config.input.measurement_stack_path)
        measurements = select_frames(measurements, config.input.frame_indices)
        return None, measurements
    raise ValueError(f"unsupported experiment mode: {config.mode}")


def _build_rngs(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    sequence = np.random.SeedSequence(seed)
    measurement_seed, sampler_seed = sequence.spawn(2)
    return np.random.default_rng(measurement_seed), np.random.default_rng(sampler_seed)


def _build_patterns(config: ExperimentConfig) -> np.ndarray:
    reconstruction_shape = _reconstruction_shape(config)
    if config.optics.frame_wavevectors_px is not None:
        return generate_calibrated_patterns(
            reconstruction_shape,
            config.optics.frame_wavevectors_px,
            config.optics.frame_phases_rad or (),
            config.optics.modulation_depth,
            config.optics.include_widefield,
        )
    return generate_phase_shifted_patterns(
        reconstruction_shape,
        config.optics.modulation_frequency,
        config.optics.orientations_deg,
        config.optics.modulation_depth,
        config.optics.phase_degs,
        config.optics.include_widefield,
    )


def _build_otf(config: ExperimentConfig) -> np.ndarray:
    reconstruction_shape = _reconstruction_shape(config)
    if config.optics.otf_path is not None:
        return load_otf(config.optics.otf_path, config.input.image_shape, reconstruction_shape)
    cutoff_frequency = latent_cutoff_frequency(
        config.input.image_shape,
        config.input.reconstruction_shape,
        config.optics.cutoff_frequency,
    )
    return airy_otf(reconstruction_shape, cutoff_frequency)


def _select_pattern_frames(patterns: np.ndarray, config: ExperimentConfig) -> np.ndarray:
    return select_frames(patterns, config.input.frame_indices)


def _validate_supported_options(config: ExperimentConfig) -> None:
    if config.noise.shared_precision:
        raise NotImplementedError("shared_precision is not part of the paper-consistent reproduction")


def _validate_measurements(measurements: np.ndarray, patterns: np.ndarray, config: ExperimentConfig) -> None:
    if measurements.shape[0] != patterns.shape[0]:
        raise ValueError(f"frame count mismatch: measurements={measurements.shape[0]} patterns={patterns.shape[0]}")
    expected_shape = config.input.image_shape
    if measurements.shape[1:] != expected_shape:
        raise ValueError(f"measurement shape {measurements.shape[1:]} does not match config image_shape {expected_shape}")


def _dry_run_message(config: ExperimentConfig, measurements: np.ndarray) -> dict[str, object]:
    return {
        "mode": config.mode,
        "name": config.name,
        "measurement_shape": list(measurements.shape),
        "reconstruction_shape": list(_reconstruction_shape(config)),
        "measurement_sampling": config.input.measurement_sampling,
        "frame_indices": None if config.input.frame_indices is None else list(config.input.frame_indices),
        "pattern_model": _pattern_model(config),
        "otf_source": "file" if config.optics.otf_path is not None else "airy",
        "phase_degs": list(config.optics.phase_degs),
        "include_widefield": config.optics.include_widefield,
        "enforce_nonnegative_image": config.sampler.enforce_nonnegative_image,
        "background_mode": config.background.mode,
        "background_smoothness_precision": config.background.smoothness_precision,
        "signal_mean_precision": config.background.signal_mean_precision,
        "cutoff_frequency": config.optics.cutoff_frequency,
        "orientations_deg": list(config.optics.orientations_deg),
    }


def _pattern_model(config: ExperimentConfig) -> str:
    if config.optics.frame_wavevectors_px is not None:
        return "calibrated_frames"
    return "phase_shifted"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Orieux 2012 Bayesian SIM")
    parser.add_argument("--config", required=True, help="Path to a JSON configuration file")
    parser.add_argument("--output", default=None, help="Override output directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and inputs only")
    return parser


def _reconstruction_shape(config: ExperimentConfig) -> tuple[int, int]:
    return resolve_reconstruction_shape(config.input.image_shape, config.input.reconstruction_shape)
