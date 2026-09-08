"""Output serialization and visualizations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .gibbs import ReconstructionResult
from .metrics import isnr, psnr, ssim


def save_run_outputs(result: ReconstructionResult, output_dir: str | Path) -> Path:
    """Write arrays, plots, and summary artifacts for one run."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    np.save(base / "measurements.npy", result.measurements)
    np.save(base / "patterns.npy", result.patterns)
    np.save(base / "otf.npy", result.otf)
    np.save(base / "posterior_mean.npy", result.posterior_mean)
    np.save(base / "posterior_variance.npy", result.posterior_variance)
    np.save(base / "gamma_n_chain.npy", result.gamma_n_chain)
    np.save(base / "gamma_f_chain.npy", result.gamma_f_chain)
    np.save(base / "mean_change_chain.npy", result.mean_change_chain)
    _save_image(base / "posterior_mean.png", result.posterior_mean)
    _save_image(base / "posterior_std.png", np.sqrt(np.maximum(result.posterior_variance, 0.0)))
    _save_measurement_strip(base / "measurements.png", result.measurements)
    _save_chain_plot(base / "hyperparameters.png", result.gamma_f_chain, result.gamma_n_chain)
    _save_background_outputs(base, result)
    if result.reference is not None:
        _save_image(base / "reference.png", result.reference)
    summary = build_summary(result)
    (base / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return base


def build_summary(result: ReconstructionResult) -> dict[str, object]:
    """Build a JSON-safe summary."""
    summary: dict[str, object] = {
        "burn_in": result.burn_in,
        "total_samples": result.total_samples,
        "posterior_samples": result.posterior_samples,
        "measurement_shape": list(result.measurements.shape[1:]),
        "posterior_shape": list(result.posterior_mean.shape),
        "final_gamma_f": float(result.gamma_f_chain[-1]),
        "final_gamma_n": [float(value) for value in result.gamma_n_chain[-1]],
    }
    if result.background_mean is not None:
        summary["background"] = _background_summary(result.background_mean)
    if result.reference is not None:
        summary.update(_metric_summary(result))
    return summary


def _metric_summary(result: ReconstructionResult) -> dict[str, float]:
    summary = {
        "ssim": ssim(result.reference, result.posterior_mean),
        "psnr": psnr(result.reference, result.posterior_mean),
    }
    if _has_explicit_widefield(result.patterns):
        summary["isnr"] = isnr(result.reference, result.posterior_mean, result.measurements[0])
    return summary


def _save_background_outputs(base: Path, result: ReconstructionResult) -> None:
    if result.background_mean is None or result.background_variance is None:
        return
    np.save(base / "background_mean.npy", result.background_mean)
    np.save(base / "background_variance.npy", result.background_variance)
    _save_image(base / "background_mean.png", result.background_mean)
    _save_image(
        base / "background_std.png",
        np.sqrt(np.maximum(result.background_variance, 0.0)),
    )


def _background_summary(background: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(background)),
        "median": float(np.median(background)),
        "mean": float(np.mean(background)),
        "max": float(np.max(background)),
    }


def _save_image(path: Path, image: np.ndarray) -> None:
    plt.imsave(path, _normalize_for_display(image), cmap="gray")


def _save_measurement_strip(path: Path, measurements: np.ndarray) -> None:
    panels = [_normalize_for_display(frame) for frame in measurements]
    strip = np.concatenate(panels, axis=1)
    plt.imsave(path, strip, cmap="gray")


def _save_chain_plot(path: Path, gamma_f: np.ndarray, gamma_n: np.ndarray) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
    axes[0].plot(gamma_f, color="black")
    axes[0].set_title("gamma_f chain")
    for index in range(gamma_n.shape[1]):
        axes[1].plot(gamma_n[:, index], label=f"gamma_n[{index}]")
    axes[1].set_title("gamma_n chains")
    axes[1].legend(loc="upper right")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if maximum == minimum:
        raise ValueError("cannot visualize a constant image")
    return (image - minimum) / (maximum - minimum)


def _has_explicit_widefield(patterns: np.ndarray) -> bool:
    return bool(np.allclose(patterns[0], 1.0))
