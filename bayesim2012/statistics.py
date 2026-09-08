"""Online sample statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OnlineMoments:
    """Track posterior mean and variance without storing all samples."""

    count: int
    mean: np.ndarray
    m2: np.ndarray

    @classmethod
    def zeros(cls, shape: tuple[int, int]) -> "OnlineMoments":
        zeros = np.zeros(shape, dtype=np.float64)
        return cls(count=0, mean=zeros.copy(), m2=zeros.copy())

    def update(self, value: np.ndarray) -> np.ndarray | None:
        previous = None if self.count == 0 else self.mean.copy()
        self.count += 1
        delta = value - self.mean
        self.mean = self.mean + delta / self.count
        delta2 = value - self.mean
        self.m2 = self.m2 + delta * delta2
        return previous

    def variance(self) -> np.ndarray:
        if self.count < 2:
            return np.zeros_like(self.mean)
        return self.m2 / (self.count - 1)


def relative_mean_change(previous: np.ndarray, current: np.ndarray) -> float:
    """Compute the paper stopping statistic between two successive means."""
    numerator = np.linalg.norm(current - previous) ** 2
    denominator = np.linalg.norm(current) ** 2
    if denominator == 0.0:
        raise ValueError("posterior mean norm is zero; stopping statistic undefined")
    return float(numerator / denominator)
