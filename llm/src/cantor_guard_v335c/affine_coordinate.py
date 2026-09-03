"""Metric-preserving affine P0 coordinate on one frozen operating window."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OUTSIDE_WINDOW = "OUTSIDE_WINDOW"


def calibrate_window(z_clean, tau: float, *, quantile: float = 0.99, padding: float = 1.05) -> float:
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0,1]")
    if padding < 1:
        raise ValueError("padding must be >= 1")
    z = np.asarray(z_clean, dtype=float).reshape(-1)
    if not len(z) or not np.isfinite(z).all() or not np.isfinite(tau):
        raise ValueError("finite calibration projections and tau are required")
    width = float(padding * np.quantile(np.abs(z - float(tau)), quantile))
    if width <= 0:
        raise ValueError("calibrated W must be positive")
    return width


@dataclass(frozen=True)
class AffineCoordinate:
    tau: float
    W: float
    orientation: int

    def __post_init__(self):
        if not np.isfinite(self.tau):
            raise ValueError("tau must be finite")
        if not np.isfinite(self.W) or self.W <= 0:
            raise ValueError("W must be finite and positive")
        if self.orientation not in (-1, 1):
            raise ValueError("orientation must be -1 or +1")

    def inside(self, z):
        arr = np.asarray(z, dtype=float)
        return np.isfinite(arr) & (np.abs(arr - self.tau) <= self.W)

    def transform(self, z):
        """Map inside values exactly; return NaN outside (never clip)."""
        arr = np.asarray(z, dtype=float)
        scalar = arr.ndim == 0
        arr = np.atleast_1d(arr)
        result = np.full(arr.shape, np.nan, dtype=float)
        ok = self.inside(arr)
        result[ok] = 0.5 + self.orientation * (arr[ok] - self.tau) / (2 * self.W)
        return float(result[0]) if scalar else result

    def inverse(self, r):
        arr = np.asarray(r, dtype=float)
        if np.any((arr < 0) | (arr > 1) | ~np.isfinite(arr)):
            raise ValueError("r must be finite and inside [0,1]")
        return self.tau + self.orientation * (arr - 0.5) * (2 * self.W)

    @property
    def metric_scale(self) -> float:
        return 1 / (2 * self.W)

    def coverage(self, z) -> float:
        arr = np.asarray(z, dtype=float).reshape(-1)
        if not len(arr):
            return float("nan")
        return float(np.mean(self.inside(arr)))
