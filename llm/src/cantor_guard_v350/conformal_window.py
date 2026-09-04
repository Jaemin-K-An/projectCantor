"""Distribution-free one-sided calibration of the risk operating radius."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ConformalWindow:
    n: int
    alpha: float
    order_index_one_based: int
    W_R: float
    empirical_coverage: float


def conformal_order_index(n: int, alpha: float = 0.05) -> int:
    """Finite-sample split-conformal index ``ceil((n+1)(1-alpha))``."""
    if int(n) != n or n <= 0:
        raise ValueError("n must be a positive integer")
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    return int(math.ceil((int(n) + 1) * (1.0 - float(alpha))))


def calibrate_upper_window(x, alpha: float = 0.05) -> ConformalWindow:
    """Return the preregistered one-sided upper order statistic.

    The standard convention gives +infinity when the requested index exceeds
    ``n``.  V3.5.0 uses n>=300 and alpha=.05, so its calibrated value is finite.
    """
    values = np.asarray(x, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("x must be a non-empty finite non-negative sample")
    k = conformal_order_index(values.size, alpha)
    W_R = float("inf") if k > values.size else float(np.sort(values)[k - 1])
    coverage = float(np.mean(values <= W_R))
    return ConformalWindow(int(values.size), float(alpha), k, W_R, coverage)
