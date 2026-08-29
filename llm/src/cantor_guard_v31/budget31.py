"""V3.1 realised residual-budget matching (harness §19).

The fairness constraint is measured on the ACTUAL model, not analytically:

    q      = ||dh|| / (||h|| + 1e-12)          per token, per layer
    C_rms  = sqrt(E[q^2])                       <- the matched quantity

Gains are found by bisection because C_rms is monotone increasing in eta for a
fixed controller shape. V2 matched only the analytic integral and the realised
cost differed by more than 6x across families.
"""
from __future__ import annotations
import numpy as np


def q_stats(qs: np.ndarray) -> dict:
    q = np.asarray(qs, float)
    q = q[np.isfinite(q)]
    if q.size == 0:
        return {"C_mean": 0.0, "C_rms": 0.0, "C95": 0.0, "C_max": 0.0}
    return {"C_mean": float(q.mean()), "C_rms": float(np.sqrt((q ** 2).mean())),
            "C95": float(np.quantile(q, 0.95)), "C_max": float(q.max())}


def match_eta(measure_fn, target: float, *, lo: float = 1e-3, hi: float = 1e3,
              tol: float = 0.03, iters: int = 18) -> tuple[float, float, bool]:
    """Bisect eta so that `measure_fn(eta)` (a C_rms) hits `target`.

    Returns (eta, achieved, ok). `ok` is False when the target is outside the
    bracket -- recorded rather than silently clipped, so an unmatched family
    can be excluded from the matched comparison (harness STOP D).
    """
    a, b = lo, hi
    fa, fb = measure_fn(a), measure_fn(b)
    if fa > target:
        return a, fa, False
    if fb < target:
        return b, fb, False
    best = (b, fb)
    for _ in range(iters):
        m = float(np.sqrt(a * b))
        fm = measure_fn(m)
        best = (m, fm)
        if abs(fm - target) / target < tol:
            return m, fm, True
        if fm < target:
            a = m
        else:
            b = m
    return best[0], best[1], abs(best[1] - target) / target < tol
