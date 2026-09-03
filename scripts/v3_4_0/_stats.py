"""Shared paired bootstrap: ONE prompt-resample matrix for the whole family."""
from __future__ import annotations

import numpy as np


def shared_index(n_prompts: int, *, n_boot: int, seed: int) -> np.ndarray:
    """IDX[replicate, prompt] reused across every rho, epsilon and endpoint.

    Building a fresh resample inside each loop -- the V3.3.5b defect corrected
    in V3.3.5c -- destroys the repeated-measures covariance that makes these
    comparisons paired in the first place.
    """
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(n_prompts), size=(int(n_boot), int(n_prompts)))


def max_t_intervals(per_prompt: dict, contrasts, idx: np.ndarray, *, alpha: float = 0.05) -> dict:
    """Simultaneous intervals over a preregistered contrast family.

    ``per_prompt`` maps an arm name to a length-n vector of per-prompt values.
    Every contrast is recomputed on the SAME resampled prompts, and the family
    maximum of |t| gives one critical value shared by all of them.
    """
    stats = {}
    for a, b in contrasts:
        diff = np.asarray(per_prompt[a], dtype=float) - np.asarray(per_prompt[b], dtype=float)
        boot = diff[idx].mean(axis=1)
        se = float(boot.std(ddof=1))
        stats[(a, b)] = {"mean": float(diff.mean()), "boot": boot, "se": se}
    t_max = np.zeros(idx.shape[0], dtype=float)
    for key, row in stats.items():
        if row["se"] > 0:
            t_max = np.maximum(t_max, np.abs(row["boot"] - row["mean"]) / row["se"])
    crit = float(np.quantile(t_max, 1 - alpha))
    out = {"critical_value": crit, "n_boot": int(idx.shape[0]), "contrasts": []}
    for (a, b), row in stats.items():
        lo, hi = row["mean"] - crit * row["se"], row["mean"] + crit * row["se"]
        out["contrasts"].append({
            "arm": a, "reference": b, "mean_difference": row["mean"], "se": row["se"],
            "simultaneous_lo": float(lo), "simultaneous_hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0),
        })
    return out


def auc_per_prompt(frame, *, value_col: str, x_col: str, group_cols) -> "object":
    """Trapezoidal area under the endpoint-vs-attack curve, per prompt.

    Normalised by the epsilon range so the value stays on the endpoint's own
    scale and is comparable across rho.
    """
    import pandas as pd

    def one(group):
        g = group.sort_values(x_col)
        x, y = g[x_col].to_numpy(dtype=float), g[value_col].to_numpy(dtype=float)
        if x.size < 2 or x.max() == x.min():
            return float("nan")
        return float(np.trapz(y, x) / (x.max() - x.min()))

    return frame.groupby(list(group_cols), sort=False).apply(one).rename("auc").reset_index()
