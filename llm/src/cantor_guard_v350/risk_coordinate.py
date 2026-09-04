"""The V3.5.0 one-sided risk coordinate.

For the frozen signed refusal-sensor distance ``d`` (larger is safer), define

    x(h) = max(0, -d(h)).

The scalar hinge is 1-Lipschitz, so composing it with the 1-Lipschitz signed
Euclidean sensor distance gives ``|Delta x| <= ||Delta h||_2``.  Positive safe
tails collapse to exactly zero rather than overflowing a symmetric window.
"""
from __future__ import annotations

import numpy as np


def risk_magnitude(d):
    """Return ``max(0,-d)`` elementwise, preserving scalar inputs."""
    arr = np.asarray(d, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("sensor distances must be finite")
    out = np.maximum(0.0, -arr)
    return float(out) if out.ndim == 0 else out


def risk_ratio(x, W_R: float):
    """Map in-window risk magnitude to ``r_R=x/W_R``.

    Values above ``W_R`` are deliberately not clipped: they become NaN here
    and are handled by the controller's explicit OUTSIDE_RISK_WINDOW policy.
    """
    if not np.isfinite(W_R) or W_R <= 0:
        raise ValueError("W_R must be finite and positive")
    arr = np.asarray(x, dtype=float)
    if not np.isfinite(arr).all() or np.any(arr < 0):
        raise ValueError("risk magnitudes must be finite and non-negative")
    out = np.full(arr.shape, np.nan, dtype=float)
    inside = arr <= float(W_R)
    out[inside] = arr[inside] / float(W_R)
    return float(out) if out.ndim == 0 else out


def residual_risk(sensor, h):
    """Evaluate ``d``, ``x`` and in-window ``r_R`` for residual states."""
    d = np.atleast_1d(np.asarray(sensor.distance(h), dtype=float))
    x = np.atleast_1d(np.asarray(risk_magnitude(d), dtype=float))
    return d, x


def lipschitz_slack(sensor, h, delta_h):
    """Return ``||Delta h||_2 - |Delta x|``; Theorem R1 makes it nonnegative."""
    base = np.atleast_2d(np.asarray(h, dtype=float))
    delta = np.atleast_2d(np.asarray(delta_h, dtype=float))
    if base.shape != delta.shape:
        raise ValueError("h and delta_h must have the same batched shape")
    x0 = np.asarray(risk_magnitude(sensor.distance(base)), dtype=float)
    x1 = np.asarray(risk_magnitude(sensor.distance(base + delta)), dtype=float)
    return np.linalg.norm(delta, axis=1) - np.abs(x1 - x0)
