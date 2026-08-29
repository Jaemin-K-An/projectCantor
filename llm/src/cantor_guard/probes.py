"""Refusal-direction estimation and calibration.

Two literature-backed estimators (harness §18):

  A. difference-in-means  -- Arditi et al. (2024), "Refusal in Language Models
     Is Mediated by a Single Direction" (NeurIPS 2024). This is the estimator
     that paper validates causally, so it is our default.
  B. regularised logistic-probe normal -- a discriminative alternative.

Neither is used until it passes the causal test in `validate_causal_direction`
(harness §19 / STOP CONDITION A): a direction that merely CLASSIFIES
harmfulness is not necessarily a direction that CAUSES refusal.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class RefusalDirections:
    """Per-layer unit directions plus the calibration constants for the margin.

    Sign convention (fixed here, tested in `test_threat_monotone`):
        larger <h, v_ref>  ==  more refusal / safer.
    """
    layers: list[int]
    v: np.ndarray          # [n_layers, d_model], each row unit norm
    tau: np.ndarray        # [n_layers] decision threshold on the projection
    sigma: np.ndarray      # [n_layers] calibration scale
    method: str

    def margin(self, acts: np.ndarray) -> np.ndarray:
        """m = (<h,v> - tau)/sigma. `acts` is [n, n_layers, d]; returns [n, n_layers]."""
        z = np.einsum("nld,ld->nl", acts, self.v)
        return (z - self.tau[None, :]) / self.sigma[None, :]

    def projection(self, acts: np.ndarray) -> np.ndarray:
        return np.einsum("nld,ld->nl", acts, self.v)


def _unit(x, axis=-1, eps=1e-12):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def diff_in_means_direction(acts_harmful: np.ndarray,
                            acts_harmless: np.ndarray) -> np.ndarray:
    """Arditi et al. estimator: mean(harmful) - mean(harmless), per layer.

    NOTE THE SIGN. This vector points from harmless toward harmful. Refusal is
    what the model does ON harmful prompts, so this is the direction whose
    ADDITION induces refusal -- consistent with our convention that a larger
    projection means more refusal.
    """
    d = acts_harmful.mean(axis=0) - acts_harmless.mean(axis=0)   # [n_layers, d]
    return _unit(d, axis=-1)


def logistic_probe_direction(acts_harmful: np.ndarray, acts_harmless: np.ndarray,
                             *, C: float = 1.0, seed: int = 0) -> np.ndarray:
    """Normal of an L2-regularised logistic boundary, per layer."""
    from sklearn.linear_model import LogisticRegression
    n_layers = acts_harmful.shape[1]
    out = np.zeros((n_layers, acts_harmful.shape[2]), dtype=np.float32)
    X_all = np.concatenate([acts_harmful, acts_harmless], axis=0)
    y = np.concatenate([np.ones(len(acts_harmful)), np.zeros(len(acts_harmless))])
    for l in range(n_layers):
        X = X_all[:, l, :]
        mu, sd = X.mean(0), X.std(0) + 1e-6
        clf = LogisticRegression(C=C, max_iter=3000, random_state=seed)
        clf.fit((X - mu) / sd, y)
        out[l] = _unit(clf.coef_[0] / sd)
    return out


def calibrate(acts_harmful: np.ndarray, acts_harmless: np.ndarray,
              v: np.ndarray, layers: list[int], method: str) -> RefusalDirections:
    """Set tau to the midpoint of the two class projection means and sigma to
    their pooled std, so `m` is a signed, scale-free distance to the boundary."""
    zh = np.einsum("nld,ld->nl", acts_harmful, v)
    zb = np.einsum("nld,ld->nl", acts_harmless, v)
    tau = 0.5 * (zh.mean(0) + zb.mean(0))
    sigma = np.sqrt(0.5 * (zh.var(0) + zb.var(0))) + 1e-8
    return RefusalDirections(layers=list(layers), v=v.astype(np.float32),
                             tau=tau.astype(np.float32),
                             sigma=sigma.astype(np.float32), method=method)


def separability(acts_harmful: np.ndarray, acts_harmless: np.ndarray,
                 v: np.ndarray) -> np.ndarray:
    """Per-layer Cohen's d between the two classes along `v` (diagnostic only:
    high separability does NOT imply causal control -- see probes docstring)."""
    zh = np.einsum("nld,ld->nl", acts_harmful, v)
    zb = np.einsum("nld,ld->nl", acts_harmless, v)
    pooled = np.sqrt(0.5 * (zh.var(0) + zb.var(0))) + 1e-12
    return (zh.mean(0) - zb.mean(0)) / pooled
