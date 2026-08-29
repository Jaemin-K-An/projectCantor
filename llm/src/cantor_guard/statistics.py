"""Paired statistics. Effect sizes and bootstrap CIs are primary; p-values are
secondary (harness §37).

Every controller is evaluated on the SAME prompts and the SAME attacks, so all
comparisons are paired; unpaired tests would throw away most of the power and
would also let prompt-difficulty differences masquerade as controller effects.
"""
from __future__ import annotations
import numpy as np


def paired_bootstrap(a: np.ndarray, b: np.ndarray, *, n_boot: int = 10000,
                     level: float = 0.95, seed: int = 0, statistic=np.mean) -> dict:
    """Bootstrap the paired difference `a - b` by resampling PAIRS."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    assert a.shape == b.shape, "paired arrays must align"
    d = a - b
    n = len(d)
    if n == 0:
        return {"mean_diff": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "n": 0, "p_two_sided": np.nan, "cohen_dz": np.nan}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = statistic(d[idx], axis=1)
    lo, hi = np.quantile(boot, [(1 - level) / 2, 1 - (1 - level) / 2])
    # two-sided bootstrap p: how often the resampled statistic crosses 0
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    sd = d.std(ddof=1) if n > 1 else 0.0
    return {"mean_diff": float(statistic(d)), "median_diff": float(np.median(d)),
            "ci_lo": float(lo), "ci_hi": float(hi), "n": int(n),
            "p_two_sided": float(min(1.0, p)),
            "cohen_dz": float(np.mean(d) / sd) if sd > 0 else 0.0,
            "frac_positive": float((d > 0).mean())}


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact McNemar test for paired BINARY outcomes (e.g. attack success)."""
    from scipy.stats import binomtest
    a = np.asarray(a).astype(bool); b = np.asarray(b).astype(bool)
    n01 = int((~a & b).sum()); n10 = int((a & ~b).sum())
    if n01 + n10 == 0:
        return {"n01": n01, "n10": n10, "p_exact": 1.0}
    p = binomtest(n10, n01 + n10, 0.5).pvalue
    return {"n01": n01, "n10": n10, "p_exact": float(p)}


def auc_log(x: np.ndarray, y: np.ndarray) -> float:
    """∫ y d(log x) by the trapezoid rule -- the log-scale robustness area
    (harness §35). `x` must be positive and sorted."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    o = np.argsort(x); x, y = x[o], y[o]
    lx = np.log(x)
    return float(np.trapezoid(y, lx)) if hasattr(np, "trapezoid") else \
           float(np.trapz(y, lx))


def worst_over_scale(y: np.ndarray) -> float:
    """R_worst: the worst value across the attack-magnitude sweep."""
    return float(np.min(np.asarray(y, float)))


def pareto_front(cost: np.ndarray, benefit: np.ndarray) -> np.ndarray:
    """Indices on the Pareto front of (low cost, high benefit)."""
    cost = np.asarray(cost, float); benefit = np.asarray(benefit, float)
    order = np.argsort(cost)
    front, best = [], -np.inf
    for i in order:
        if benefit[i] > best:
            front.append(i); best = benefit[i]
    return np.array(front, dtype=int)
