"""Signed Euclidean distance to a learned behavioural decision hyperplane.

The sensor is a linear map ``f(h) = w^T h + b``.  The geometric coordinate is
NOT that logit but its normalised form

    d(h) = (w^T h + b) / ||w||_2,

which is the signed Euclidean distance from ``h`` to the hyperplane
``w^T h + b = 0``.

THEOREM S (1-Lipschitz).  For any perturbation ``dh``

    d(h + dh) - d(h) = w^T dh / ||w||,

so by Cauchy-Schwarz ``|dd| <= ||dh||_2`` with constant exactly 1, and the
bound is attained when ``dh`` is parallel to ``w_hat = w/||w||``.  That
tightness is what makes ``d`` a usable certified residual coordinate: no
perturbation of L2 size ``eps`` can move the sensor state further than ``eps``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_batch(h) -> tuple[np.ndarray, bool]:
    arr = np.asarray(h, dtype=float)
    if arr.ndim == 1:
        return arr[None, :], True
    if arr.ndim != 2:
        raise ValueError("h must have shape [d] or [batch,d]")
    return arr, False


@dataclass(frozen=True)
class SensorHyperplane:
    """A frozen linear behavioural sensor and its induced distance."""

    w: np.ndarray
    b: float

    def __post_init__(self):
        w = np.asarray(self.w, dtype=float).reshape(-1)
        norm = float(np.linalg.norm(w))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("sensor normal must have finite non-zero norm")
        if not np.isfinite(float(self.b)):
            raise ValueError("sensor intercept must be finite")
        object.__setattr__(self, "w", w)
        object.__setattr__(self, "b", float(self.b))

    @property
    def w_norm(self) -> float:
        return float(np.linalg.norm(self.w))

    @property
    def w_hat(self) -> np.ndarray:
        return self.w / self.w_norm

    def logit(self, h) -> np.ndarray:
        batch, squeeze = _as_batch(h)
        if batch.shape[1] != self.w.size:
            raise ValueError("residual and sensor dimensions differ")
        out = batch @ self.w + self.b
        return float(out[0]) if squeeze else out

    def distance(self, h):
        """Signed Euclidean distance d(h); positive means the y=1 side."""
        batch, squeeze = _as_batch(h)
        if batch.shape[1] != self.w.size:
            raise ValueError("residual and sensor dimensions differ")
        out = (batch @ self.w + self.b) / self.w_norm
        return float(out[0]) if squeeze else out

    def delta_distance(self, delta_h):
        """Exact sensor movement induced by a residual perturbation."""
        batch, squeeze = _as_batch(delta_h)
        if batch.shape[1] != self.w.size:
            raise ValueError("perturbation and sensor dimensions differ")
        out = (batch @ self.w) / self.w_norm
        return float(out[0]) if squeeze else out

    def lipschitz_slack(self, delta_h):
        """``||dh|| - |dd|`` -- non-negative by Theorem S, zero along w_hat."""
        batch, _ = _as_batch(delta_h)
        return np.linalg.norm(batch, axis=1) - np.abs(np.atleast_1d(self.delta_distance(batch)))

    def project_to_hyperplane(self, h):
        """Closest point on ``w^T h + b = 0``; used only for tests."""
        batch, squeeze = _as_batch(h)
        moved = batch - np.atleast_1d(self.distance(batch))[:, None] * self.w_hat[None, :]
        return moved[0] if squeeze else moved

    def coupling(self, v) -> float:
        """kappa = <w_hat, v>; the exact sensor gain of a unit actuator step."""
        direction = np.asarray(v, dtype=float).reshape(-1)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("actuator direction must have finite non-zero norm")
        if direction.size != self.w.size:
            raise ValueError("actuator and sensor dimensions differ")
        return float(self.w_hat @ (direction / norm))

    def to_dict(self) -> dict:
        return {"b": self.b, "w_norm": self.w_norm, "d_model": int(self.w.size)}
