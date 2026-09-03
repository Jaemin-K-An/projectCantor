"""Logistic and isotonic P0 behavioural-boundary estimation with hard gates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


def fit_logistic(z, y, *, l2: float = 1e-6, max_iter: int = 200) -> tuple[float, float]:
    z = np.asarray(z, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if z.size != y.size or z.size == 0:
        raise ValueError("equal non-empty z/y are required")
    center, scale = float(z.mean()), float(z.std())
    if scale <= 0:
        return 0.0, 0.0
    x = (z - center) / scale
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    penalty = np.diag([0.0, l2])
    for _ in range(max_iter):
        eta = np.clip(X @ w, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        variance = np.maximum(p * (1 - p), 1e-9)
        hessian = X.T @ (X * variance[:, None]) + penalty
        gradient = X.T @ (y - p) - penalty @ w
        step = np.linalg.solve(hessian, gradient)
        w += step
        if np.max(np.abs(step)) < 1e-10:
            break
    b = float(w[1] / scale)
    a = float(w[0] - w[1] * center / scale)
    return a, b


def _pava(x, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(x, kind="stable")
    xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]
    unique, inverse = np.unique(xs, return_inverse=True)
    sums = np.bincount(inverse, weights=ys)
    counts = np.bincount(inverse).astype(float)
    values = list((sums / counts).astype(float))
    weights = list(counts)
    starts = list(range(len(values)))
    ends = list(range(len(values)))
    i = 0
    while i < len(values) - 1:
        if values[i] > values[i + 1]:
            weight = weights[i] + weights[i + 1]
            value = (values[i] * weights[i] + values[i + 1] * weights[i + 1]) / weight
            values[i : i + 2] = [value]
            weights[i : i + 2] = [weight]
            ends[i : i + 2] = [ends[i + 1]]
            starts[i : i + 2] = [starts[i]]
            i = max(i - 1, 0)
        else:
            i += 1
    fitted = np.empty(len(unique), dtype=float)
    for value, start, end in zip(values, starts, ends):
        fitted[start : end + 1] = value
    return unique, fitted, counts


def isotonic_crossing(z, y, *, expected_slope_sign: int) -> float | None:
    if expected_slope_sign not in (-1, 1):
        raise ValueError("expected_slope_sign must be +/-1")
    oriented = expected_slope_sign * np.asarray(z, dtype=float)
    x, fitted, _ = _pava(oriented, y)
    if fitted.min() > 0.5 or fitted.max() < 0.5:
        return None
    above = np.flatnonzero(fitted >= 0.5)
    if not len(above):
        return None
    i = int(above[0])
    if i == 0:
        crossing = x[0]
    elif fitted[i] == fitted[i - 1]:
        crossing = 0.5 * (x[i - 1] + x[i])
    else:
        crossing = x[i - 1] + (0.5 - fitted[i - 1]) * (x[i] - x[i - 1]) / (fitted[i] - fitted[i - 1])
    return float(expected_slope_sign * crossing)


def prompt_cluster_bootstrap(
    z, y, pid, *, n_boot: int = 20_000, seed: int = 335,
    expected_slope_sign: int | None = None,
) -> dict:
    if n_boot < 20_000:
        raise ValueError("V3.3.5c requires at least 20,000 bootstrap replicates")
    z = np.asarray(z, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=int).reshape(-1)
    pid = np.asarray(pid).reshape(-1)
    prompts = np.unique(pid)
    groups = [np.flatnonzero(pid == prompt) for prompt in prompts]
    index = np.random.default_rng(seed).integers(0, len(prompts), size=(n_boot, len(prompts)), dtype=np.int32)
    tau = np.full(n_boot, np.nan)
    slope = np.full(n_boot, np.nan)
    tau_isotonic = np.full(n_boot, np.nan)
    for replicate, sampled in enumerate(index):
        obs = np.concatenate([groups[i] for i in sampled])
        a, b = fit_logistic(z[obs], y[obs])
        slope[replicate] = b
        if abs(b) > 1e-12:
            tau[replicate] = -a / b
        if expected_slope_sign is not None:
            crossing = isotonic_crossing(
                z[obs], y[obs], expected_slope_sign=expected_slope_sign
            )
            if crossing is not None:
                tau_isotonic[replicate] = crossing
    iso_finite = np.isfinite(tau_isotonic)
    return {
        "n_boot": n_boot,
        "n_prompts": int(len(prompts)),
        "tau_ci95": [float(np.nanquantile(tau, 0.025)), float(np.nanquantile(tau, 0.975))],
        "slope_ci95": [float(np.nanquantile(slope, 0.025)), float(np.nanquantile(slope, 0.975))],
        "finite_tau_fraction": float(np.mean(np.isfinite(tau))),
        "isotonic_tau_ci95": (
            [float(np.nanquantile(tau_isotonic, 0.025)), float(np.nanquantile(tau_isotonic, 0.975))]
            if iso_finite.any() else [float("nan"), float("nan")]
        ),
        "finite_isotonic_tau_fraction": float(np.mean(iso_finite)),
        "tau_samples": tau,
        "slope_samples": slope,
        "isotonic_tau_samples": tau_isotonic,
        "index": index,
    }


def _dose_bin_rates(u, y, safe_sign: int) -> tuple[np.ndarray, np.ndarray]:
    dose = safe_sign * np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    bins = np.unique(dose)
    return bins, np.asarray([y[dose == value].mean() for value in bins])


def monotonicity_checks(u, y, *, safe_sign: int, reversal_tolerance: float = 0.15) -> dict:
    bins, rates = _dose_bin_rates(u, y, safe_sign)
    rho, pvalue = spearmanr(safe_sign * np.asarray(u, float), np.asarray(y, float))
    differences = np.diff(rates)
    major_reversals = int(np.sum(differences < -float(reversal_tolerance)))
    return {
        "spearman_r": float(rho),
        "spearman_p": float(pvalue),
        "spearman_sign_correct": bool(np.isfinite(rho) and rho > 0),
        "major_adjacent_reversals": major_reversals,
        "no_major_reversal": major_reversals == 0,
        "dose_bins_safe_oriented": bins.tolist(),
        "refusal_rates": rates.tolist(),
    }


@dataclass(frozen=True)
class BoundaryFit:
    intercept: float
    slope: float
    tau_logistic: float | None
    tau_isotonic: float | None
    sigma_p0: float
    bootstrap: dict
    checks: dict
    verdict: str


def fit_behavioral_boundary(
    *,
    z_after,
    z_clean,
    outcome,
    pid,
    u,
    coherence,
    degeneration,
    safe_sign: int,
    confirm_direction_agrees: bool,
    n_boot: int = 20_000,
    seed: int = 335,
    coherence_gate: float = 0.95,
    degeneration_rate_gate: float = 0.05,
    crossing_agreement_sigma: float = 1.0,
) -> BoundaryFit:
    z = np.asarray(z_after, float).reshape(-1)
    clean_z = np.asarray(z_clean, float).reshape(-1)
    y = np.asarray(outcome, int).reshape(-1)
    u = np.asarray(u, float).reshape(-1)
    coh = np.asarray(coherence, float).reshape(-1)
    deg = np.asarray(degeneration, bool).reshape(-1)
    if not (len(z) == len(clean_z) == len(y) == len(u) == len(coh) == len(deg) == len(np.asarray(pid))):
        raise ValueError("all observation arrays must have equal length")
    a, b = fit_logistic(z, y)
    tau = float(-a / b) if abs(b) > 1e-12 else None
    iso = isotonic_crossing(z, y, expected_slope_sign=safe_sign)
    pid_array = np.asarray(pid)
    _, first_prompt_rows = np.unique(pid_array, return_index=True)
    sigma = float(np.std(clean_z[first_prompt_rows], ddof=1))
    boot = prompt_cluster_bootstrap(
        z, y, pid, n_boot=n_boot, seed=seed, expected_slope_sign=safe_sign
    )
    mono = monotonicity_checks(u, y, safe_sign=safe_sign)
    lo, hi = boot["tau_ci95"]
    iso_lo, iso_hi = boot["isotonic_tau_ci95"]
    slo, shi = boot["slope_ci95"]
    _, rates = _dose_bin_rates(u, y, safe_sign)
    bracketed = bool(np.min(rates) < 0.5 < np.max(rates))
    slope_direction = bool(np.sign(b) == safe_sign)
    slope_ci_excludes_zero = bool(slo > 0) if safe_sign > 0 else bool(shi < 0)
    crossing_agrees = bool(
        tau is not None
        and iso is not None
        and sigma > 0
        and abs(tau - iso) <= crossing_agreement_sigma * sigma
    )
    checks = {
        "both_outcomes_observed": bool(np.unique(y).size == 2),
        "transition_0_5_bracketed": bracketed,
        "slope_direction_correct": slope_direction,
        "slope_ci_excludes_zero": slope_ci_excludes_zero,
        "tau_inside_realised_z": bool(tau is not None and z.min() <= tau <= z.max()),
        "tau_ci_finite": bool(np.isfinite([lo, hi]).all()),
        "tau_ci_width_le_3_sigma": bool(sigma > 0 and hi - lo <= 3 * sigma),
        "isotonic_tau_inside_realised_z": bool(iso is not None and z.min() <= iso <= z.max()),
        "isotonic_tau_ci_finite": bool(np.isfinite([iso_lo, iso_hi]).all()),
        "isotonic_tau_ci_width_le_3_sigma": bool(sigma > 0 and iso_hi - iso_lo <= 3 * sigma),
        "isotonic_bootstrap_stable": bool(boot["finite_isotonic_tau_fraction"] >= 0.95),
        "coherence_gate": bool(float(np.mean(coh)) >= coherence_gate),
        "degeneration_gate": bool(float(np.mean(deg)) <= degeneration_rate_gate),
        "monotonicity": bool(mono["spearman_sign_correct"] and mono["no_major_reversal"]),
        "dev_confirm_crossing_direction_agrees": bool(confirm_direction_agrees),
        "logistic_isotonic_agree": crossing_agrees,
        "mean_coherence": float(np.mean(coh)),
        "degeneration_rate": float(np.mean(deg)),
        "tau_ci_width_sigma": float((hi - lo) / sigma) if sigma > 0 else float("inf"),
        "isotonic_tau_ci_width_sigma": float((iso_hi - iso_lo) / sigma) if sigma > 0 else float("inf"),
        "beta_std": float(b * sigma),
        "monotonicity_detail": mono,
    }
    logistic_keys = [
        "both_outcomes_observed",
        "transition_0_5_bracketed",
        "slope_direction_correct",
        "slope_ci_excludes_zero",
        "tau_inside_realised_z",
        "tau_ci_finite",
        "tau_ci_width_le_3_sigma",
        "coherence_gate",
        "degeneration_gate",
        "monotonicity",
        "dev_confirm_crossing_direction_agrees",
        "logistic_isotonic_agree",
    ]
    if all(bool(checks[key]) for key in logistic_keys):
        verdict = "B1_P0_BEHAVIORAL_BOUNDARY_IDENTIFIED"
    elif (
        iso is not None
        and checks["both_outcomes_observed"]
        and checks["transition_0_5_bracketed"]
        and checks["coherence_gate"]
        and checks["degeneration_gate"]
        and checks["monotonicity"]
        and checks["dev_confirm_crossing_direction_agrees"]
        and checks["isotonic_tau_inside_realised_z"]
        and checks["isotonic_tau_ci_finite"]
        and checks["isotonic_tau_ci_width_le_3_sigma"]
        and checks["isotonic_bootstrap_stable"]
    ):
        verdict = "B2_NONPARAMETRIC_BOUNDARY_ONLY"
    else:
        verdict = "B3_BOUNDARY_UNIDENTIFIABLE"
    clean_boot = {
        k: v for k, v in boot.items()
        if k not in {"tau_samples", "slope_samples", "isotonic_tau_samples", "index"}
    }
    return BoundaryFit(a, b, tau, iso, sigma, clean_boot, checks, verdict)
