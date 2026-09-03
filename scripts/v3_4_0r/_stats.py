"""Shared paired bootstrap and censor-aware survival."""
from __future__ import annotations

import numpy as np


def shared_index(n_prompts: int, *, n_boot: int, seed: int) -> np.ndarray:
    """ONE IDX[replicate, prompt] reused across every arm, epsilon and endpoint."""
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(n_prompts), size=(int(n_boot), int(n_prompts)))


def max_t_intervals(per_prompt: dict, contrasts, idx: np.ndarray, *, alpha: float = 0.05) -> dict:
    stats = {}
    for a, b in contrasts:
        diff = np.asarray(per_prompt[a], dtype=float) - np.asarray(per_prompt[b], dtype=float)
        boot = diff[idx].mean(axis=1)
        stats[(a, b)] = {"mean": float(diff.mean()), "boot": boot, "se": float(boot.std(ddof=1))}
    t_max = np.zeros(idx.shape[0], dtype=float)
    for row in stats.values():
        if row["se"] > 0:
            t_max = np.maximum(t_max, np.abs(row["boot"] - row["mean"]) / row["se"])
    crit = float(np.quantile(t_max, 1 - alpha)) if np.any(t_max > 0) else 0.0
    out = {"critical_value": crit, "n_boot": int(idx.shape[0]), "contrasts": []}
    for (a, b), row in stats.items():
        lo, hi = row["mean"] - crit * row["se"], row["mean"] + crit * row["se"]
        out["contrasts"].append({"arm": a, "reference": b, "mean_difference": row["mean"],
                                 "se": row["se"], "simultaneous_lo": float(lo),
                                 "simultaneous_hi": float(hi),
                                 "excludes_zero": bool(lo > 0 or hi < 0)})
    return out


def auc_per_prompt(frame, *, value_col: str, x_col: str, group_cols):
    import pandas as pd  # noqa: F401

    def one(group):
        g = group.sort_values(x_col)
        x, y = g[x_col].to_numpy(dtype=float), g[value_col].to_numpy(dtype=float)
        if x.size < 2 or x.max() == x.min():
            return float("nan")
        return float(np.trapz(y, x) / (x.max() - x.min()))

    return frame.groupby(list(group_cols), sort=False).apply(one).rename("auc").reset_index()


def discrete_survival(first_failure, censored_at, grid) -> dict:
    """Event-free survival on a discrete attack grid, honouring right-censoring.

    V3.4.0 took a median over observed events only, under 62-82% censoring; that
    is not a population median. Here a prompt with no failure is censored at
    epsilon_max, the survival curve is estimated over the grid, and the median
    is reported ONLY if the curve actually crosses 0.5.
    """
    eps = sorted(float(e) for e in grid if e > 0)
    first = np.asarray(first_failure, dtype=float)
    n = first.size
    if n == 0:
        return {"n": 0, "median": None, "median_status": "NO_DATA"}
    curve, at_risk = [], n
    survival = 1.0
    prev = 0.0
    rmst = 0.0
    for e in eps:
        events = int(np.sum(first == e))
        if at_risk > 0:
            survival *= (1 - events / at_risk)
        rmst += survival * (e - prev)
        prev = e
        at_risk -= events
        curve.append({"epsilon": e, "events": events, "survival": float(survival),
                      "at_risk_after": int(at_risk)})
    crossed = [row for row in curve if row["survival"] <= 0.5]
    median = float(crossed[0]["epsilon"]) if crossed else None
    return {
        "n": int(n),
        "n_events": int(np.sum(np.isfinite(first))),
        "censoring_rate": float(np.mean(~np.isfinite(first))),
        "curve": curve,
        "final_survival": float(survival),
        "restricted_mean_failure_free_epsilon": float(rmst),
        "median": median,
        "median_status": "IDENTIFIED" if median is not None else "NOT_IDENTIFIED_IN_TESTED_RANGE",
        "censored_at": float(max(eps)) if eps else None,
    }
