"""Signed, scale-normalized P0 residual dosing.

For unit direction v and P0 residual h, the intervention is

    delta_h = u ||h||_2 v.

Consequently ||delta_h||/||h|| = |u| and the realised projection shift is
u||h||.  Downstream boundary fitting must nevertheless use measured z_after,
not the nominal dose index.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DoseResult:
    h_after: np.ndarray
    delta_h: np.ndarray
    u: np.ndarray
    h_clean_norm: np.ndarray
    relative_norm: np.ndarray
    z_clean: np.ndarray
    z_after: np.ndarray


def normalize_direction(v, *, atol: float = 1e-12) -> np.ndarray:
    direction = np.asarray(v, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= atol:
        raise ValueError("refusal direction must have finite non-zero norm")
    return direction / norm


def _as_batch(h) -> tuple[np.ndarray, bool]:
    arr = np.asarray(h, dtype=float)
    if arr.ndim == 1:
        return arr[None, :], True
    if arr.ndim != 2:
        raise ValueError("h must have shape [d] or [batch,d]")
    return arr, False


def apply_normalized_dose(h, v, u) -> DoseResult:
    """Apply signed relative dose and return all realised quantities."""
    batch, squeeze = _as_batch(h)
    direction = normalize_direction(v)
    if batch.shape[1] != direction.size:
        raise ValueError("h and v dimensions differ")
    dose = np.asarray(u, dtype=float)
    if dose.ndim == 0:
        dose = np.full(batch.shape[0], float(dose))
    dose = dose.reshape(-1)
    if dose.size != batch.shape[0]:
        raise ValueError("u must be scalar or one value per residual")
    norms = np.linalg.norm(batch, axis=1)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("every P0 residual must have finite positive norm")
    delta = dose[:, None] * norms[:, None] * direction[None, :]
    after = batch + delta
    relative = np.linalg.norm(delta, axis=1) / norms
    z_clean = batch @ direction
    z_after = after @ direction

    def maybe_squeeze(x):
        return x[0] if squeeze else x

    return DoseResult(
        h_after=maybe_squeeze(after),
        delta_h=maybe_squeeze(delta),
        u=maybe_squeeze(dose),
        h_clean_norm=maybe_squeeze(norms),
        relative_norm=maybe_squeeze(relative),
        z_clean=maybe_squeeze(z_clean),
        z_after=maybe_squeeze(z_after),
    )


def dose_norm_summary(relative_norm) -> dict[str, float]:
    q = np.abs(np.asarray(relative_norm, dtype=float).reshape(-1))
    if q.size == 0:
        raise ValueError("at least one realised dose is required")
    return {
        "median_abs_delta_over_h": float(np.median(q)),
        "p95_abs_delta_over_h": float(np.quantile(q, 0.95)),
        "max_abs_delta_over_h": float(np.max(q)),
    }


def choose_safe_orientation(
    refusal_at_negative: float,
    refusal_at_positive: float,
    *,
    harmful_at_negative: float | None = None,
    harmful_at_positive: float | None = None,
) -> int:
    """Freeze one sign that improves refusal and, when present, compliance.

    Ties or disagreement are not silently resolved because orientation is a
    scientific gate, not a tunable controller parameter.
    """
    dr = float(refusal_at_positive) - float(refusal_at_negative)
    refusal_sign = 1 if dr > 0 else (-1 if dr < 0 else 0)
    if harmful_at_negative is None or harmful_at_positive is None:
        if refusal_sign == 0:
            raise ValueError("safe orientation is not identifiable from tied refusal rates")
        return refusal_sign
    dh = float(harmful_at_positive) - float(harmful_at_negative)
    harmful_safe_sign = -1 if dh > 0 else (1 if dh < 0 else 0)
    if refusal_sign == 0 or harmful_safe_sign == 0 or refusal_sign != harmful_safe_sign:
        raise ValueError("refusal and harmful-compliance endpoints do not identify one safe sign")
    return refusal_sign
