"""Shared, arm-independent risk eligibility and conditional budget fitting."""
from __future__ import annotations

import numpy as np


def risk_eligibility(d_attacked) -> np.ndarray:
    """The only allowed budget mask: frozen pre-control ``d_attacked < 0``."""
    d = np.asarray(d_attacked, dtype=float).reshape(-1)
    if not np.isfinite(d).all():
        raise ValueError("pre-control sensor distances must be finite")
    return d < 0


def budget_metrics(q, eligibility, q_raw=None) -> dict:
    q = np.asarray(q, dtype=float).reshape(-1)
    mask = np.asarray(eligibility, dtype=bool).reshape(-1)
    if q.shape != mask.shape or not mask.any():
        raise ValueError("q and a nonempty common eligibility mask are required")
    raw = q if q_raw is None else np.asarray(q_raw, dtype=float).reshape(-1)
    return {
        "risk_q_rms": float(np.sqrt(np.mean(q[mask] ** 2))),
        "global_q_rms": float(np.sqrt(np.mean(q ** 2))),
        "q_mean": float(q.mean()), "q_p95": float(np.quantile(q, .95)),
        "q_max": float(q.max()), "clipping_fraction": float(np.mean(raw > q)),
        "intervention_frequency": float(np.mean(q > 0)),
        "safe_side_intervention_frequency": float(np.mean(q[~mask] > 0)) if (~mask).any() else 0.0,
        "risk_eligible_prevalence": float(mask.mean()),
    }


def fit_eta_risk_conditional(actions, eligibility, *, target: float = 0.03,
                             q_cap: float = 0.05) -> tuple[float, dict]:
    """Fit eta using only the common risk mask; global RMS never enters solving."""
    action = np.asarray(actions, dtype=float).reshape(-1)
    mask = np.asarray(eligibility, dtype=bool).reshape(-1)
    if action.shape != mask.shape or not mask.any():
        raise ValueError("actions and a nonempty common risk mask are required")
    if np.any(action[mask] <= 0):
        raise ValueError("all risk-eligible actions must be strictly positive")
    if np.any(action[~mask] != 0):
        raise ValueError("safe-side actions must be exactly zero")
    def risk_rms(eta):
        q = np.minimum(float(eta) * action, q_cap)
        return float(np.sqrt(np.mean(q[mask] ** 2)))
    maximum = risk_rms(1e15)
    if maximum + 1e-12 < target:
        return float("nan"), {"maximum_attainable_risk_q_rms": maximum}
    lo, hi = 0.0, 1.0
    while risk_rms(hi) < target: hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if risk_rms(mid) < target: lo = mid
        else: hi = mid
    eta = (lo + hi) / 2
    q_raw = eta * action; q = np.minimum(q_raw, q_cap)
    metrics = budget_metrics(q, mask, q_raw)
    metrics["maximum_attainable_risk_q_rms"] = maximum
    return eta, metrics
