"""V3.3.2 PHASE 7/8 -- three DIFFERENT quantities, never summed.

V3.3.1 DEFECT. Its "eps_cal" applied two coordinate transforms to the SAME
prompt projections and took the difference. That is a deterministic systematic
offset with no sampling variability -- it is not estimator uncertainty, and the
n_max claim built on it does not stand.

  U_EST    ESTIMATOR UNCERTAINTY. How far the boundary LOCATION could move if
           the calibration prompts were resampled. Prompt-clustered bootstrap,
           harmful and harmless resampled independently. This is the only one
           that feeds the guard theorem, and it is estimated with NO reference
           to rho.

  U_PHASE  SYSTEMATIC BIAS, not uncertainty. Where the generation-phase
           midpoint sits when read in prompt-phase coordinates -- the error the
           V3.2 controller made by construction. Reported separately and never
           called uncertainty.

  U_STATE  Dispersion of the state itself within a generation. Not a property
           of the estimator, so it must not be used as the guard input.
"""
from __future__ import annotations
import numpy as np
from .calibration import phase_calibration, threat_coordinate

__all__ = ["u_est_bootstrap", "u_phase_bias", "u_state_dispersion"]


def u_est_bootstrap(z_harmful, z_harmless, *, gamma: float = 0.7,
                    n_boot: int = 20000, seed: int = 0) -> dict:
    """PRIMARY delta_abs. Prompt-clustered bootstrap of the boundary location.

    One observation per prompt (PHASE G1), so resampling rows IS resampling
    prompt clusters. Each replicate's tau is expressed as a shift in the
    REFERENCE coordinate and mapped to threat-coordinate width:

        Delta_m^(b) = (tau^(b) - tau_hat) / sigma_hat
        delta_r^(b) = | sigmoid(-gamma * Delta_m^(b)) - 1/2 |

    delta_r is a width in r-units, which is what a guard has to absorb, and it
    does not involve rho anywhere.
    """
    zh = np.asarray(z_harmful, float).ravel()
    zb = np.asarray(z_harmless, float).ravel()
    ref = phase_calibration(zh, zb)
    tau_hat, sig_hat = ref["tau"], ref["sigma"]
    rng = np.random.default_rng(seed)
    ih = rng.integers(0, len(zh), size=(n_boot, len(zh)))
    ib = rng.integers(0, len(zb), size=(n_boot, len(zb)))
    tau_b = 0.5 * (zh[ih].mean(axis=1) + zb[ib].mean(axis=1))
    sig_b = np.sqrt(0.5 * (zh[ih].var(axis=1, ddof=1) + zb[ib].var(axis=1, ddof=1)))
    dm = (tau_b - tau_hat) / sig_hat
    dr = np.abs(threat_coordinate(dm * sig_hat + tau_hat, tau_hat, sig_hat, gamma) - 0.5)
    q = {f"q{p}": float(np.quantile(dr, p / 100)) for p in (50, 75, 90, 95)}
    return {"reference": ref, "n_boot": int(n_boot),
            "delta_abs_quantiles": q,
            "delta_abs_mean": float(dr.mean()),
            "tau_ci95": [float(np.quantile(tau_b, .025)), float(np.quantile(tau_b, .975))],
            "sigma_ci95": [float(np.quantile(sig_b, .025)), float(np.quantile(sig_b, .975))],
            "delta_abs_samples": dr}


def u_phase_bias(cal_P: dict, cal_G: dict, gamma: float = 0.7) -> dict:
    """SYSTEMATIC BIAS. Where tau_G sits in PROMPT coordinates."""
    m_phase = (cal_G["tau"] - cal_P["tau"]) / cal_P["sigma"]
    return {"m_phase_sigma": float(m_phase),
            "delta_phase_r": float(abs(threat_coordinate(
                cal_G["tau"], cal_P["tau"], cal_P["sigma"], gamma) - 0.5)),
            "scale_ratio_sigma_G_over_P": float(cal_G["sigma"] / cal_P["sigma"]),
            "IS_BIAS_NOT_UNCERTAINTY": True}


def u_state_dispersion(margins_per_prompt) -> dict:
    """SECONDARY. Within-generation state spread; NOT an estimator property."""
    per = [float(np.std(np.asarray(m, float))) for m in margins_per_prompt
           if len(np.asarray(m, float)) > 1]
    return {"median_within_prompt_sd": float(np.median(per)) if per else np.nan,
            "n_prompts": len(per),
            "IS_NOT_CALIBRATION_UNCERTAINTY": True}
