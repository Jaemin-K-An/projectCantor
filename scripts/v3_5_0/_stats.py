"""Prompt-level shared bootstrap, max-T intervals and censor-aware survival."""
from __future__ import annotations

import numpy as np


def shared_index(n_prompts: int, *, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(n_prompts), size=(int(n_boot), int(n_prompts)))


def max_t_intervals(per_prompt: dict, contrasts, idx, alpha=.05):
    stats = {}
    for arm, ref in contrasts:
        diff = np.asarray(per_prompt[arm], float) - np.asarray(per_prompt[ref], float)
        boot = diff[idx].mean(axis=1); se = float(boot.std(ddof=1))
        stats[(arm, ref)] = (float(diff.mean()), se, boot)
    tmax = np.zeros(len(idx))
    for mean, se, boot in stats.values():
        if se > 0: tmax = np.maximum(tmax, np.abs(boot - mean) / se)
    crit = float(np.quantile(tmax, 1 - alpha)) if np.any(tmax) else 0.0
    rows = []
    for (arm, ref), (mean, se, _boot) in stats.items():
        rows.append({"arm": arm, "reference": ref, "mean_difference": mean, "se": se,
                     "simultaneous_lo": mean - crit * se, "simultaneous_hi": mean + crit * se})
    return {"n_boot": len(idx), "critical_value": crit, "contrasts": rows}


def auc_per_prompt(frame, value_col="y_safe"):
    def one(group):
        group = group.sort_values("epsilon")
        x = group.epsilon.to_numpy(float); y = group[value_col].to_numpy(float)
        return float(np.trapz(y, x) / (x.max() - x.min())) if len(x) > 1 and x.max() > x.min() else float("nan")
    return frame.groupby(["family", "arm", "pid"], sort=False).apply(one).rename("auc").reset_index()


def discrete_survival(first_failure, grid):
    first = np.asarray(first_failure, float); eps = sorted(float(e) for e in grid if e > 0)
    at_risk, surv, prev, rmst, curve = len(first), 1.0, 0.0, 0.0, []
    for e in eps:
        events = int(np.sum(first == e))
        if at_risk: surv *= 1 - events / at_risk
        rmst += surv * (e - prev); prev = e; at_risk -= events
        curve.append({"epsilon": e, "events": events, "survival": surv, "at_risk_after": at_risk})
    crossed = [x for x in curve if x["survival"] <= .5]
    return {"n": len(first), "n_events": int(np.isfinite(first).sum()),
            "censoring_rate": float(np.mean(~np.isfinite(first))), "curve": curve,
            "median": crossed[0]["epsilon"] if crossed else None,
            "median_status": "IDENTIFIED" if crossed else "NOT_IDENTIFIED_IN_TESTED_RANGE",
            "restricted_mean_failure_free_epsilon": rmst, "censored_at": max(eps)}
