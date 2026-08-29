"""A frozen harmfulness detector, shared identically by every controller.

Why this exists (harness §20): a controller that always pushes toward refusal
is a trivial over-refusal machine. Gating on harmfulness prevents that, but the
gate must not be part of what is being compared -- so ONE detector is fitted on
CALIBRATION, frozen, and handed to every controller including the baselines.

Two regimes are reported separately:
  Experiment A (oracle)     : the benchmark label is used as the gate. Isolates
                              the controller mechanism. Benign utility cost is
                              zero BY CONSTRUCTION, so no Pareto claim can be
                              made from it.
  Experiment B (deployable) : this learned detector is used. Its false
                              positives are what create a real benign utility
                              cost, which is what the Pareto analysis needs.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class HarmDetector:
    layer: int
    w: np.ndarray          # unit-norm direction in residual space
    mu: float              # threshold on the projection
    scale: float
    threshold: float       # decision threshold on the calibrated score

    def score(self, acts_layer: np.ndarray) -> np.ndarray:
        """acts_layer: [n, d] residuals at `self.layer` -> calibrated score."""
        return (acts_layer @ self.w - self.mu) / self.scale

    def predict(self, acts_layer: np.ndarray) -> np.ndarray:
        return self.score(acts_layer) > self.threshold

    def provenance(self) -> dict:
        return {"harm_detector_layer": int(self.layer),
                "harm_detector_threshold": float(self.threshold)}


def fit_harm_detector(acts_harmful: np.ndarray, acts_harmless: np.ndarray,
                      layer_index: int, layer: int, *,
                      target_fpr: float = 0.10, seed: int = 0) -> HarmDetector:
    """Fit at one layer on CALIBRATION data; threshold set to a target false
    positive rate on the harmless calibration set (not on dev or test)."""
    from sklearn.linear_model import LogisticRegression
    Xh = acts_harmful[:, layer_index, :]
    Xb = acts_harmless[:, layer_index, :]
    X = np.concatenate([Xh, Xb]); y = np.concatenate([np.ones(len(Xh)), np.zeros(len(Xb))])
    mu_, sd_ = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=1.0, max_iter=3000, random_state=seed)
    clf.fit((X - mu_) / sd_, y)
    w = clf.coef_[0] / sd_
    w = w / (np.linalg.norm(w) + 1e-12)
    zh, zb = Xh @ w, Xb @ w
    mu = 0.5 * (zh.mean() + zb.mean())
    scale = np.sqrt(0.5 * (zh.var() + zb.var())) + 1e-8
    sb = (zb - mu) / scale
    thr = float(np.quantile(sb, 1 - target_fpr))
    return HarmDetector(layer=layer, w=w.astype(np.float32), mu=float(mu),
                        scale=float(scale), threshold=thr)
