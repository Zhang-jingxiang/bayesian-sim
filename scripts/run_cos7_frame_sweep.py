"""Run calibrated COS7 reconstructions for 5 through 9 selected frames."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import tifffile

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from bayesim2012.cli import run_experiment
from bayesim2012.config import ExperimentConfig, load_config
from bayesim2012.metrics import affine_reference_metrics, fit_affine_intensity


FRAME_ORDER = (0, 3, 6, 1, 4, 7, 2, 5, 8)
FRAME_COUNTS = (5, 6, 7, 8)
BASE_CONFIG_PATH = PACKAGE_ROOT / "configs" / "cos7_first9_hr_background.json"
GROUND_TRUTH_PATH = PACKAGE_ROOT / "data" / "gt.tif"
SWEEP_OUTPUT_DIR = PACKAGE_ROOT / "results" / "cos7_first9_hr_background_phase_refined_frame_sweep"


def main(argv: list[str] | None = None) -> int:
    """Run the frame-count sweep and save GT-based comparisons."""
    args = _build_parser().parse_args(argv)
    frame_counts = tuple(args.frame_counts)
    _validate_frame_counts(frame_counts)
    base_config = load_config(BASE_CONFIG_PATH)
    reference = _load_ground_truth(GROUND_TRUTH_PATH, base_config)
    entries = _run_entries(base_config, reference, frame_counts, args.workers)
    montage_path = _save_comparison_montage(reference, entries, SWEEP_OUTPUT_DIR / "comparison.png")
    summary = _build_sweep_summary(base_config, entries, montage_path, frame_counts)
    _write_json(SWEEP_OUTPUT_DIR / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def _run_entries(
    base_config: ExperimentConfig,
    reference: np.ndarray,
    frame_counts: tuple[int, ...],
    workers: int,
) -> list[dict[str, object]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        return [_run_one(base_config, reference, frame_count) for frame_count in frame_counts]
    requests = [(base_config, reference, frame_count) for frame_count in frame_counts]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_run_entry, requests))


def _run_entry(request: tuple[ExperimentConfig, np.ndarray, int]) -> dict[str, object]:
    base_config, reference, frame_count = request
    return _run_one(base_config, reference, frame_count)


def _run_one(
    base_config: ExperimentConfig,
    reference: np.ndarray,
    frame_count: int,
) -> dict[str, object]:
    frame_indices = FRAME_ORDER[:frame_count]
    output_dir = SWEEP_OUTPUT_DIR / f"frames_{frame_count}"
    config = _subset_config(base_config, frame_count, frame_indices, output_dir)
    result = run_experiment(config, None, False)
    metrics = affine_reference_metrics(reference, result.posterior_mean)
    evaluation = {
        "ground_truth_path": str(GROUND_TRUTH_PATH),
        "posterior_shape": list(result.posterior_mean.shape),
        "metrics": metrics,
    }
    _write_json(output_dir / "gt_evaluation.json", evaluation)
    return {
        "frame_count": frame_count,
        "frame_indices": list(frame_indices),
        "output_dir": str(output_dir),
        "posterior_mean_path": str(output_dir / "posterior_mean.npy"),
        "gt_evaluation": evaluation,
        "summary": _load_json(output_dir / "summary.json"),
    }


def _load_ground_truth(path: Path, config: ExperimentConfig) -> np.ndarray:
    reference = tifffile.imread(path).astype(np.float64)
    expected_shape = config.input.reconstruction_shape
    if expected_shape is None or reference.shape != expected_shape:
        raise ValueError(f"GT shape {reference.shape} does not match reconstruction shape {expected_shape}")
    return reference


def _subset_config(
    base_config: ExperimentConfig,
    frame_count: int,
    frame_indices: tuple[int, ...],
    output_dir: Path,
) -> ExperimentConfig:
    input_config = replace(base_config.input, frame_indices=frame_indices)
    output_config = replace(base_config.output, directory=str(output_dir))
    return replace(
        base_config,
        name=f"{base_config.name}_frames_{frame_count}",
        input=input_config,
        output=output_config,
    )


def _save_comparison_montage(
    reference: np.ndarray,
    entries: list[dict[str, object]],
    output_path: Path,
) -> Path:
    figure, axes = plt.subplots(1, len(entries) + 1, figsize=(4 * (len(entries) + 1), 4), constrained_layout=True)
    axes[0].imshow(_normalize_display(reference), cmap="gray")
    axes[0].set_title("GT")
    axes[0].set_axis_off()
    for axis, entry in zip(axes[1:], entries):
        estimate = np.load(Path(str(entry["posterior_mean_path"])))
        matched, _, _ = fit_affine_intensity(reference, estimate)
        axis.imshow(_normalize_display(matched), cmap="gray")
        axis.set_title(f"{entry['frame_count']} frames")
        axis.set_axis_off()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _normalize_display(image: np.ndarray) -> np.ndarray:
    lower = float(np.percentile(image, 0.5))
    upper = float(np.percentile(image, 99.5))
    if upper <= lower:
        raise ValueError("display normalization range is invalid")
    return np.clip((image - lower) / (upper - lower), 0.0, 1.0)


def _build_sweep_summary(
    config: ExperimentConfig,
    entries: list[dict[str, object]],
    montage_path: Path,
    frame_counts: tuple[int, ...],
) -> dict[str, object]:
    return {
        "dataset": str(config.input.measurement_stack_path),
        "ground_truth": str(GROUND_TRUTH_PATH),
        "frame_order": list(FRAME_ORDER),
        "counts": list(frame_counts),
        "montage_path": str(montage_path),
        "results": entries,
    }


def _write_json(path: Path, content: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run calibrated COS7 Bayesian frame sweeps")
    parser.add_argument(
        "--frame-counts",
        nargs="+",
        type=int,
        default=FRAME_COUNTS,
        help="Numbers of selected frames to reconstruct",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of independent frame-count reconstructions to run concurrently",
    )
    return parser


def _validate_frame_counts(frame_counts: tuple[int, ...]) -> None:
    if not frame_counts:
        raise ValueError("frame_counts must not be empty")
    if any(count < 1 or count > len(FRAME_ORDER) for count in frame_counts):
        raise ValueError(f"frame_counts must lie in 1..{len(FRAME_ORDER)}")
    if len(set(frame_counts)) != len(frame_counts):
        raise ValueError("frame_counts must be unique")


if __name__ == "__main__":
    raise SystemExit(main())
