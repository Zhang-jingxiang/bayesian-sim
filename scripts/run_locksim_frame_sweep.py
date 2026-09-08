"""Run 5/6/7/8/9-frame Bayesian reconstructions on the LockSIM dataset."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from bayesim2012.cli import run_experiment
from bayesim2012.config import ExperimentConfig, load_config


FRAME_ORDER = (0, 3, 6, 1, 4, 7, 2, 5, 8)
FRAME_COUNTS = (5, 6, 7, 8, 9)
BASE_CONFIG_PATH = PACKAGE_ROOT / "configs" / "locksim_mito_real9.json"
SWEEP_OUTPUT_DIR = PACKAGE_ROOT / "results" / "locksim_mito_frame_sweep"
PROXY_SR_PATH = PACKAGE_ROOT.parents[1] / "data" / "2dsim" / "2025LockSIM" / "Mitochondria_LockinSIM_488nm_1.49NA_65nm_SR_f32.tif"


def main() -> int:
    base_config = load_config(BASE_CONFIG_PATH)
    proxy_reference = _load_proxy_reference(PROXY_SR_PATH)
    sweep_entries = []
    initial_image = None
    for frame_count in FRAME_COUNTS:
        entry, initial_image = _run_one(base_config, frame_count, proxy_reference, initial_image)
        sweep_entries.append(entry)
        print(f"completed frame_count={frame_count} -> {entry['posterior_mean_path']}")
    montage_path = _save_posterior_montage(sweep_entries, SWEEP_OUTPUT_DIR / "posterior_mean_sweep.png")
    summary = {
        "dataset": str(base_config.input.measurement_stack_path),
        "frame_order": list(FRAME_ORDER),
        "counts": list(FRAME_COUNTS),
        "montage_path": str(montage_path),
        "results": sweep_entries,
    }
    summary_path = SWEEP_OUTPUT_DIR / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _run_one(
    base_config: ExperimentConfig,
    frame_count: int,
    proxy_reference: np.ndarray,
    initial_image: np.ndarray | None,
) -> tuple[dict[str, object], np.ndarray]:
    frame_indices = FRAME_ORDER[:frame_count]
    output_dir = SWEEP_OUTPUT_DIR / f"frames_{frame_count}"
    config = _config_for_subset(base_config, frame_count, frame_indices, output_dir)
    result = run_experiment(config, None, False, initial_image=initial_image)
    posterior_proxy = _proxy_metrics(proxy_reference, result.posterior_mean)
    baseline_proxy = _proxy_metrics(proxy_reference, np.mean(result.measurements, axis=0))
    entry = {
        "frame_count": frame_count,
        "frame_indices": list(frame_indices),
        "output_dir": str(output_dir),
        "posterior_mean_path": str(output_dir / "posterior_mean.png"),
        "posterior_mean_npy": str(output_dir / "posterior_mean.npy"),
        "proxy_vs_sr_downsampled": {
            "posterior_mean": posterior_proxy,
            "measurement_mean_baseline": baseline_proxy,
        },
        "summary": _load_summary(output_dir / "summary.json"),
    }
    return entry, result.posterior_mean


def _config_for_subset(
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


def _load_proxy_reference(path: Path) -> np.ndarray:
    proxy = tifffile.imread(path).astype(np.float64)
    return proxy.reshape(512, 2, 512, 2).mean(axis=(1, 3))


def _proxy_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    reference_norm = _normalize_percentile(reference)
    estimate_norm = _normalize_percentile(estimate)
    return {
        "ssim": float(structural_similarity(reference_norm, estimate_norm, data_range=1.0)),
        "psnr": float(peak_signal_noise_ratio(reference_norm, estimate_norm, data_range=1.0)),
    }


def _normalize_percentile(image: np.ndarray) -> np.ndarray:
    lower = float(np.percentile(image, 0.5))
    upper = float(np.percentile(image, 99.5))
    if upper <= lower:
        raise ValueError("percentile normalization range is invalid")
    clipped = np.clip(image, lower, upper)
    return (clipped - lower) / (upper - lower)


def _save_posterior_montage(entries: list[dict[str, object]], output_path: Path) -> Path:
    figure, axes = plt.subplots(1, len(entries), figsize=(4 * len(entries), 4), constrained_layout=True)
    for axis, entry in zip(np.atleast_1d(axes), entries):
        image = np.load(Path(entry["posterior_mean_npy"]))
        axis.imshow(_normalize_display(image), cmap="gray")
        axis.set_title(f"{entry['frame_count']} frames")
        axis.set_axis_off()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _normalize_display(image: np.ndarray) -> np.ndarray:
    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if maximum == minimum:
        raise ValueError("cannot display a constant image")
    return (image - minimum) / (maximum - minimum)


def _load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
