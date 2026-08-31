"""V3.3.3 -- the BEHAVIORAL refusal boundary, tau_beh.

FOUR DISTINCT QUANTITIES, never interchanged:

  tau_mid   midpoint of the harmful/harmless PROJECTION distributions. This is
            what V3.2/V3.3.2 estimated. It is a property of representations.
  tau_beh   the projection value at which the model's BEHAVIOUR crosses from
            compliance to refusal. This is a property of behaviour, and it is
            what a guard is actually supposed to bracket.
  U_EST     sampling uncertainty of an estimated boundary parameter.
  U_PHASE   systematic prefill -> generation shift.

tau_mid and tau_beh answer different questions and there is no reason they
coincide. V3.3.3 estimates tau_beh directly and reports the gap.

DOSE-RESPONSE, AND WHY IT IS NOT CIRCULAR. The refusal direction v is a unit
vector, so adding lambda*v at the hooked layer shifts the projection by exactly
lambda:  z' = <h + lambda v, v> = z + lambda. Sweeping lambda therefore sweeps
the projection through the transition without any fitted quantity entering the
predictor. The regression is P(refusal | z') against the REALISED z', not
against the dose index.

IDENTIFIABILITY GATE. tau_beh is returned only if all six hold; otherwise
TAU_BEH_UNIDENTIFIABLE, and no downstream claim may substitute tau_mid.
"""
from __future__ import annotations
import numpy as np

__all__ = ["fit_logistic", "fit_isotonic", "isotonic_crossing",
           "identifiability", "tau_beh_bootstrap", "TAU_BEH_UNIDENTIFIABLE"]

TAU_BEH_UNIDENTIFIABLE = "TAU_BEH_UNIDENTIFIABLE"


def fit_logistic(z, y, *, l2: float = 1e-4, iters: int = 200):
    """logit P(y=1 | z) = a + b z, by Newton-IRLS with a light ridge.

    Returns (a, b). A tiny ridge keeps the fit finite under perfect separation
    rather than letting b run to infinity, which would make -a/b meaningless.
    """
    z = np.asarray(z, float).ravel()
    y = np.asarray(y, float).ravel()
    X = np.column_stack([np.ones_like(z), z])
    w = np.zeros(2)
    for _ in range(iters):
        eta = X @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        W = np.maximum(p * (1 - p), 1e-9)
        H = X.T @ (X * W[:, None]) + l2 * np.eye(2)
        g = X.T @ (y - p) - l2 * w
        step = np.linalg.solve(H, g)
        w = w + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(w[0]), float(w[1])


def fit_isotonic(z, y):
    """Pool-adjacent-violators isotonic fit; robustness check on the logistic.

    Returns (z_sorted, p_fitted) with p non-decreasing in z.
    """
    z = np.asarray(z, float).ravel()
    y = np.asarray(y, float).ravel()
    o = np.argsort(z)
    zs, ys = z[o], y[o].astype(float)
    vals = list(ys)
    wts = [1.0] * len(ys)
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1] + 1e-15:
            tot = wts[i] + wts[i + 1]
            merged = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / tot
            vals[i:i + 2] = [merged]; wts[i:i + 2] = [tot]
            i = max(i - 1, 0)
        else:
            i += 1
    out = np.concatenate([[v] * int(round(w)) for v, w in zip(vals, wts)])
    return zs, out[:len(zs)]


def isotonic_crossing(z, y, level: float = 0.5):
    zs, ps = fit_isotonic(z, y)
    above = np.where(ps >= level)[0]
    if len(above) == 0 or above[0] == 0:
        return None
    i = above[0]
    p0, p1, z0, z1 = ps[i - 1], ps[i], zs[i - 1], zs[i]
    if abs(p1 - p0) < 1e-12:
        return float(0.5 * (z0 + z1))
    return float(z0 + (level - p0) * (z1 - z0) / (p1 - p0))


def identifiability(z, y, a: float, b: float, tau: float, ci=None,
                    *, min_slope: float = 0.05, min_per_class: int = 10,
                    dose_bins=None, max_ci_width_sigma=None, sigma=None) -> dict:
    """The identifiability gate. All checks must pass or tau_beh is withheld.

    `transition_inside_range` on its own is too weak: tau can fall inside the
    numeric span of z while the OBSERVED refusal proportion never crosses 0.5,
    in which case the 50% point is pure extrapolation. When `dose_bins` is
    supplied the stricter `transition_observed` check requires the binned
    refusal proportion to actually bracket 0.5, and `ci_width_reasonable`
    requires the bootstrap interval to be narrower than a stated number of
    sigma. Both make identification HARDER, never easier.
    """
    z = np.asarray(z, float).ravel()
    y = np.asarray(y, float).ravel()
    checks = {
        "both_classes_present": bool(y.sum() >= min_per_class
                                     and (len(y) - y.sum()) >= min_per_class),
        "slope_nondegenerate": bool(abs(b) >= min_slope),
        "slope_direction_sensible": bool(b > 0),   # more projection -> more refusal
        "transition_inside_range": bool(np.isfinite(tau)
                                        and z.min() <= tau <= z.max()),
        "ci_finite": bool(ci is None or (np.isfinite(ci[0]) and np.isfinite(ci[1]))),
        "fit_not_flat": False,
    }
    p = 1.0 / (1.0 + np.exp(-np.clip(a + b * z, -30, 30)))
    checks["fit_not_flat"] = bool(p.max() - p.min() >= 0.30)
    if dose_bins is not None:
        props = np.asarray([np.mean(y[np.asarray(dose_bins) == d])
                            for d in np.unique(dose_bins)], float)
        checks["transition_observed"] = bool(props.min() < 0.5 < props.max())
        checks["observed_refusal_range"] = [float(props.min()), float(props.max())]
    if ci is not None and max_ci_width_sigma is not None and sigma:
        w = (ci[1] - ci[0]) / sigma
        checks["ci_width_reasonable"] = bool(w <= max_ci_width_sigma)
        checks["ci_width_sigma"] = float(w)
    checks["all_pass"] = bool(all(
        v for k, v in checks.items()
        if k not in ("all_pass", "observed_refusal_range", "ci_width_sigma")))
    return checks


def tau_beh_bootstrap(z, y, prompt_id, *, n_boot: int = 20000, seed: int = 0):
    """Prompt-clustered bootstrap. The resampling unit is the PROMPT: every
    dose of a resampled prompt travels with it, and the FULL model is refit in
    each replicate."""
    z = np.asarray(z, float).ravel()
    y = np.asarray(y, float).ravel()
    pid = np.asarray(prompt_id)
    groups = {p: np.where(pid == p)[0] for p in np.unique(pid)}
    keys = list(groups)
    rng = np.random.default_rng(seed)
    a0, b0 = fit_logistic(z, y)
    tau0 = -a0 / b0 if abs(b0) > 1e-12 else np.nan
    taus = np.full(n_boot, np.nan)
    slopes = np.full(n_boot, np.nan)
    for i in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups[keys[j]] for j in pick])
        a, b = fit_logistic(z[idx], y[idx])
        if abs(b) > 1e-12:
            taus[i] = -a / b
            slopes[i] = b
    ok = np.isfinite(taus)
    return {"tau_hat": float(tau0), "slope_hat": float(b0),
            "intercept_hat": float(a0),
            "tau_ci95": [float(np.nanquantile(taus, .025)),
                         float(np.nanquantile(taus, .975))],
            "slope_ci95": [float(np.nanquantile(slopes, .025)),
                           float(np.nanquantile(slopes, .975))],
            "n_prompts": len(keys), "n_obs": int(len(z)),
            "boot_ok_fraction": float(ok.mean()),
            "tau_samples": taus[ok]}
