"""Outcome-free attacked-state intervention-budget utilities."""
from __future__ import annotations

import numpy as np


def capped_q(actions, eta: float, q_cap: float = 0.05) -> np.ndarray:
    actions = np.asarray(actions, dtype=float)
    if np.any(actions < 0) or not np.isfinite(actions).all():
        raise ValueError("actions must be finite and nonnegative")
    if eta < 0 or q_cap <= 0:
        raise ValueError("eta must be nonnegative and q_cap positive")
    return np.minimum(float(eta) * actions, float(q_cap))


def q_rms(actions, eta: float, q_cap: float = 0.05) -> float:
    q = capped_q(actions, eta, q_cap)
    return float(np.sqrt(np.mean(q ** 2)))


def solve_eta(actions, target: float = 0.03, q_cap: float = 0.05) -> float:
    """Solve the monotone hard-capped RMS equation without using outcomes."""
    actions = np.asarray(actions, dtype=float)
    if target <= 0 or target > q_cap:
        raise ValueError("target must lie in (0,q_cap]")
    if actions.size == 0 or np.all(actions == 0) or q_rms(actions, 1e9, q_cap) < target:
        raise ValueError("target is not attainable")
    lo, hi = 0.0, 1.0
    while q_rms(actions, hi, q_cap) < target:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if q_rms(actions, mid, q_cap) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


__all__ = ["capped_q", "q_rms", "solve_eta"]
