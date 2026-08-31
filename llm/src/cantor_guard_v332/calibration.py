"""V3.3.2 PHASE 5/6 -- a real two-class, phase-specific calibration.

V3.3.1 DEFECT. The "generation calibration" was

    tau_g = tau_p + mean_margin(attacked harmful generations) * sigma_p
    sigma_g = std_margin(same) * sigma_p

which is the mean and spread of ONE class, not the midpoint of two. A
calibration that never sees the harmless class cannot locate a boundary
between them, so tau_g was not a boundary at all.

Here every phase gets the same estimator the prompt phase always used:

    tau_phi   = 1/2 [ E(z | harmful, phi) + E(z | harmless, phi) ]
    sigma_phi = sqrt( 1/2 [ Var(z | harmful, phi) + Var(z | harmless, phi) ] )

PHASES
    P    last prompt token (the position the refusal decision is made at)
    G1   FIRST decode forward -- exactly one state per prompt, so no
         output-length weighting and a clean prompt-cluster bootstrap
    G1_4, G5_8   token-bin means, secondary only

Calibration prompts are CLEAN. Attack templates are an evaluation
perturbation, not a class definition (V3.3.1 fed only attacked harmful text).
"""
from __future__ import annotations
import numpy as np
from cantor_guard.probes import RefusalDirections

__all__ = ["phase_calibration", "PHASES", "to_directions", "threat_coordinate"]

PHASES = ("P", "G1", "G1_4", "G5_8")


def phase_calibration(z_harmful: np.ndarray, z_harmless: np.ndarray) -> dict:
    """Two-class midpoint calibration from 1-D projections."""
    zh = np.asarray(z_harmful, float).ravel()
    zb = np.asarray(z_harmless, float).ravel()
    if len(zh) < 2 or len(zb) < 2:
        raise ValueError("need >=2 observations in each class")
    tau = 0.5 * (zh.mean() + zb.mean())
    sigma = float(np.sqrt(0.5 * (zh.var(ddof=1) + zb.var(ddof=1))))
    pooled = max(sigma, 1e-12)
    return {"tau": float(tau), "sigma": pooled,
            "mean_harmful": float(zh.mean()), "mean_harmless": float(zb.mean()),
            "separability": float((zh.mean() - zb.mean()) / pooled),
            "n_harmful": int(len(zh)), "n_harmless": int(len(zb))}


def to_directions(v: np.ndarray, layer: int, cal: dict) -> RefusalDirections:
    return RefusalDirections(layers=[layer],
                             v=np.asarray(v, np.float32).reshape(1, -1),
                             tau=np.array([cal["tau"]], np.float32),
                             sigma=np.array([cal["sigma"]], np.float32),
                             method="diff_means_phase")


def threat_coordinate(z, tau: float, sigma: float, gamma: float = 0.7):
    """r = sigmoid(-gamma * m), m = (z - tau)/sigma. Same map as V2/V3."""
    m = (np.asarray(z, float) - tau) / sigma
    return 1.0 / (1.0 + np.exp(np.clip(gamma * m, -60, 60)))
