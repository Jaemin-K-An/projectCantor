"""Frozen prompt-level shared bootstrap and max-T simultaneous intervals."""
from __future__ import annotations

import numpy as np


def shared_index(n_prompts: int, *, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(n_prompts), size=(int(n_boot), int(n_prompts)))


def max_t_intervals(per_prompt: dict, contrasts, idx, alpha=.05):
    stats = {}
    for arm, reference in contrasts:
        diff = np.asarray(per_prompt[arm], float) - np.asarray(per_prompt[reference], float)
        boot = diff[idx].mean(axis=1)
        se = float(boot.std(ddof=1))
        stats[(arm, reference)] = (float(diff.mean()), se, boot)
    tmax = np.zeros(len(idx))
    for mean, se, boot in stats.values():
        if se > 0:
            tmax = np.maximum(tmax, np.abs(boot - mean) / se)
    critical = float(np.quantile(tmax, 1 - alpha)) if np.any(tmax) else 0.0
    rows = []
    for (arm, reference), (mean, se, _boot) in stats.items():
        rows.append({
            "arm": arm, "reference": reference, "mean_difference": mean, "se": se,
            "simultaneous_lo": mean - critical * se,
            "simultaneous_hi": mean + critical * se,
        })
    return {"n_boot": len(idx), "critical_value": critical, "contrasts": rows}


def auc_per_prompt(frame, value_col="y_safe"):
    def one(group):
        group = group.sort_values("epsilon")
        x = group.epsilon.to_numpy(float)
        y = group[value_col].to_numpy(float)
        if len(x) <= 1 or x.max() <= x.min():
            return float("nan")
        return float(np.trapz(y, x) / (x.max() - x.min()))
    return (frame.groupby(["family", "arm", "pid"], sort=False)
            .apply(one).rename("auc").reset_index())
