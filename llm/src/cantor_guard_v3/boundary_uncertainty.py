"""V3 PHASE 3 — is the LLM safety boundary actually non-stationary?

This is the empirical question the whole of V3 rests on (harness §37). If
tau(x, l, t, a) barely moves, then V2's fixed calibration was fine, the
boundary-uncertainty framing is unmotivated, and that must be reported.

We measure the refusal projection z = <h, v_ref> and re-estimate the decision
threshold tau within each condition, then report

    Delta_norm = (tau_condition - tau_global) / sigma_global

in units of the global within-class spread, so shifts are comparable across
layers. Conditions: prompt phase (prefill vs generation), token position bin,
attack family, layer, and prompt group.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def threshold_from_projections(z_harm: np.ndarray, z_ben: np.ndarray
                               ) -> tuple[float, float]:
    """Midpoint threshold and pooled spread, matching V2's `calibrate`.

    Using the same estimator as V2 is deliberate: the question is how much V2's
    OWN quantity moves across conditions, not whether a better estimator exists.
    """
    tau = 0.5 * (float(np.mean(z_harm)) + float(np.mean(z_ben)))
    sigma = float(np.sqrt(0.5 * (np.var(z_harm) + np.var(z_ben)))) + 1e-8
    return tau, sigma


def normalised_shift(tau_cond: float, tau_global: float, sigma_global: float
                     ) -> float:
    """Boundary displacement in units of the global spread."""
    return (tau_cond - tau_global) / sigma_global


def summarise_shifts(df: pd.DataFrame, by: list[str],
                     value: str = "delta_norm") -> pd.DataFrame:
    """Distribution of |Delta| per grouping, with the quantiles that define the
    pre-registered uncertainty set U_Delta."""
    g = df.groupby(by)[value]
    out = g.agg(n="size", mean="mean", std="std",
                q50=lambda s: s.abs().median(),
                q75=lambda s: s.abs().quantile(0.75),
                q90=lambda s: s.abs().quantile(0.90),
                q95=lambda s: s.abs().quantile(0.95),
                qmax=lambda s: s.abs().max()).reset_index()
    return out


def uncertainty_set(df: pd.DataFrame, quantiles=(0.75, 0.95),
                    value: str = "delta_norm") -> dict:
    """U_Delta declared from the measured NATURAL shift distribution.

    Declared from calibration/dev measurements only, then frozen -- never
    chosen after seeing test robustness (harness §26).
    """
    a = df[value].abs().dropna().values
    return {f"q{int(100*q)}": float(np.quantile(a, q)) for q in quantiles} | {
        "median": float(np.median(a)), "max": float(np.max(a)), "n": int(len(a))}
