"""V3.2 cluster-aware inference (harness §4-§8).

V3.1 DEFECT: the paired unit was (goal x attack x Delta x eps), giving 480
"independent" cells from only 12 harmful goals -- a 40x inflation of the
effective sample size. Repeated conditions on the same goal are not
independent observations of controller quality.

The primary inference unit here is the GOAL. A cluster bootstrap resamples
whole goals with replacement, carrying every (attack, Delta, eps) observation
of a chosen goal along with it, so within-goal correlation is preserved
instead of being counted as extra evidence.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def cluster_bootstrap_by_goal(df: pd.DataFrame, col_a: str, col_b: str, *,
                              goal_col: str = "pid", n_boot: int = 20000,
                              level: float = 0.95, seed: int = 0) -> dict:
    """Paired difference `a - b` with a goal-clustered bootstrap CI.

    `df` must be one row per (goal, condition) with both controllers' scores in
    `col_a`/`col_b`. The statistic is the mean over goals of the within-goal
    mean difference, which weights every goal equally regardless of how many
    conditions it contributed.
    """
    rng = np.random.default_rng(seed)
    goals = df[goal_col].unique()
    per = {}
    for g in goals:
        s = df[df[goal_col] == g]
        per[g] = (s[col_a].to_numpy() - s[col_b].to_numpy())
    gmeans = np.array([per[g].mean() for g in goals])
    obs = float(gmeans.mean())
    idx = rng.integers(0, len(goals), size=(n_boot, len(goals)))
    boot = gmeans[idx].mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(boot, [a, 1 - a])
    sd = gmeans.std(ddof=1) if len(goals) > 1 else 0.0
    return {"mean_diff": obs, "ci_lo": float(lo), "ci_hi": float(hi),
            "half_width": float((hi - lo) / 2), "n_goals": int(len(goals)),
            "n_rows": int(len(df)), "between_goal_sd": float(sd),
            "cohen_dz": float(obs / sd) if sd > 0 else 0.0,
            "p_two_sided": float(min(1.0, 2 * min((boot <= 0).mean(),
                                                  (boot >= 0).mean())))}


def hierarchical_bootstrap(df: pd.DataFrame, col_a: str, col_b: str, *,
                           goal_col: str = "pid", inner_col: str = "attack",
                           n_boot: int = 20000, level: float = 0.95,
                           seed: int = 0) -> dict:
    """Nested bootstrap: resample goals, then attacks WITHIN each chosen goal.

    Reported as a sensitivity analysis; it additionally accounts for the fact
    that attack templates are themselves a small sample.
    """
    rng = np.random.default_rng(seed)
    goals = df[goal_col].unique()
    tree = {}
    for g in goals:
        s = df[df[goal_col] == g]
        tree[g] = {k: (v[col_a].to_numpy() - v[col_b].to_numpy())
                   for k, v in s.groupby(inner_col)}
    obs = float(np.mean([np.mean([d.mean() for d in tree[g].values()]) for g in goals]))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        gs = rng.choice(goals, size=len(goals), replace=True)
        vals = []
        for g in gs:
            keys = list(tree[g])
            ks = rng.choice(len(keys), size=len(keys), replace=True)
            vals.append(np.mean([tree[g][keys[j]].mean() for j in ks]))
        boot[i] = np.mean(vals)
    a = (1 - level) / 2
    lo, hi = np.quantile(boot, [a, 1 - a])
    return {"mean_diff": obs, "ci_lo": float(lo), "ci_hi": float(hi),
            "half_width": float((hi - lo) / 2), "n_goals": int(len(goals))}


def naive_cell_bootstrap(df: pd.DataFrame, col_a: str, col_b: str, *,
                         n_boot: int = 20000, level: float = 0.95,
                         seed: int = 0) -> dict:
    """The V3.1 procedure, kept so the inflation can be quantified side by side.
    NOT to be used for any claim."""
    rng = np.random.default_rng(seed)
    x = df[col_a].to_numpy() - df[col_b].to_numpy()
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boot = x[idx].mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(boot, [a, 1 - a])
    return {"mean_diff": float(x.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "half_width": float((hi - lo) / 2), "n_cells": int(len(x))}


def tost_equivalence(stat: dict, sesoi: float = 0.03) -> dict:
    """Equivalence by CI inclusion: the two one-sided tests are passed exactly
    when the (1-2a) CI lies inside +-sesoi. We use the 95% CI, which is
    conservative relative to the usual 90% TOST interval."""
    inside = (stat["ci_lo"] > -sesoi) and (stat["ci_hi"] < sesoi)
    return {"sesoi": sesoi, "equivalent": bool(inside),
            "ci_lo": stat["ci_lo"], "ci_hi": stat["ci_hi"],
            "margin_lo": float(stat["ci_lo"] + sesoi),
            "margin_hi": float(sesoi - stat["ci_hi"])}
