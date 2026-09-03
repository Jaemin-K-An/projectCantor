"""L2-regularized linear behavioural sensor trained on CLEAN P0 residuals.

Labels come from the model's ACTUAL OUTPUT BEHAVIOUR on safety-relevant
prompts (y=1 safe/refusal, y=0 harmful compliance), never from the
harmful-vs-harmless prompt distinction -- that shortcut is what produced the
old actuator direction ``v`` and is exactly what V3.4.0 is trying to avoid.

Only an L2 logistic probe is permitted for the primary analysis, so that the
sensor keeps a transparent residual-space distance certificate (Theorem S).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from .sensor_distance import SensorHyperplane

# Frozen before any confirmatory fit.  Selection happens on D_sensor_tune only.
C_GRID = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def _check_labels(y) -> np.ndarray:
    labels = np.asarray(y, dtype=int).reshape(-1)
    if set(np.unique(labels).tolist()) - {0, 1}:
        raise ValueError("sensor labels must be binary 0/1")
    if labels.min() == labels.max():
        raise ValueError("sensor training needs both behaviour classes")
    return labels


def fit_sensor(H, y, *, C: float, max_iter: int = 5000) -> SensorHyperplane:
    """Fit one L2 logistic probe.  No unregularized fit is ever allowed."""
    X = np.asarray(H, dtype=float)
    labels = _check_labels(y)
    if X.ndim != 2 or X.shape[0] != labels.size:
        raise ValueError("H must be [n,d] aligned with y")
    if not np.isfinite(float(C)) or float(C) <= 0:
        raise ValueError("C must be finite and positive")
    model = LogisticRegression(
        penalty="l2", C=float(C), solver="lbfgs", max_iter=max_iter, class_weight="balanced"
    )
    model.fit(X, labels)
    return SensorHyperplane(model.coef_.reshape(-1), float(model.intercept_[0]))


def _null_brier(y_train, y_eval) -> float:
    base = float(np.mean(np.asarray(y_train, dtype=float)))
    labels = np.asarray(y_eval, dtype=float).reshape(-1)
    return float(np.mean((labels - base) ** 2))


def sensor_metrics(sensor: SensorHyperplane, H, y, *, y_train=None) -> dict:
    """Discrimination and calibration on a held-out split.

    ``balanced accuracy`` is evaluated at the hyperplane itself (d=0), not at a
    tuned threshold, because d=0 is the decision boundary the Cantor geometry
    is centred on.
    """
    labels = np.asarray(y, dtype=int).reshape(-1)
    d = np.atleast_1d(sensor.distance(H))
    p = 1.0 / (1.0 + np.exp(-np.atleast_1d(sensor.logit(H))))
    single_class = labels.min() == labels.max()
    out = {
        "n": int(labels.size),
        "base_rate": float(labels.mean()),
        "auroc": float("nan") if single_class else float(roc_auc_score(labels, d)),
        "pr_auc": float("nan") if single_class else float(average_precision_score(labels, d)),
        "balanced_accuracy_at_zero": float(balanced_accuracy_score(labels, (d > 0).astype(int)))
        if not single_class
        else float("nan"),
        "accuracy_at_zero": float(np.mean((d > 0).astype(int) == labels)),
        "brier": float(brier_score_loss(labels, np.clip(p, 0, 1))),
        "null_brier": _null_brier(labels if y_train is None else y_train, labels),
        "mean_distance_y1": float(d[labels == 1].mean()) if (labels == 1).any() else float("nan"),
        "mean_distance_y0": float(d[labels == 0].mean()) if (labels == 0).any() else float("nan"),
    }
    if not single_class:
        tn, fp, fn, tp = confusion_matrix(labels, (d > 0).astype(int), labels=[0, 1]).ravel()
        out["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        out.update(_calibration(labels, np.atleast_1d(sensor.logit(H))))
    out["beats_null_brier"] = bool(out["brier"] < out["null_brier"])
    return out


def _calibration(labels, logit) -> dict:
    """Cox calibration: regress the label on the frozen sensor logit."""
    x = np.asarray(logit, dtype=float).reshape(-1, 1)
    if not np.isfinite(x).all() or np.ptp(x) <= 0:
        return {"calibration_slope": float("nan"), "calibration_intercept": float("nan")}
    try:
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000)
        model.fit(x, labels)
    except Exception:
        return {"calibration_slope": float("nan"), "calibration_intercept": float("nan")}
    return {
        "calibration_slope": float(model.coef_[0][0]),
        "calibration_intercept": float(model.intercept_[0]),
    }


def bootstrap_auroc_ci(sensor: SensorHyperplane, H, y, *, n_boot: int, seed: int) -> dict:
    """Prompt-cluster (here: row = prompt) bootstrap CI for the AUROC."""
    labels = np.asarray(y, dtype=int).reshape(-1)
    d = np.atleast_1d(sensor.distance(H))
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, labels.size, size=(int(n_boot), labels.size))
    values = []
    for row in idx:
        yr = labels[row]
        if yr.min() == yr.max():
            continue
        values.append(roc_auc_score(yr, d[row]))
    if not values:
        return {"auroc_ci95": [float("nan"), float("nan")], "n_effective": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "auroc_ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "n_effective": int(arr.size),
    }


@dataclass
class SensorGate:
    """Frozen confirmatory gate; thresholds are set before D_sensor_confirm."""

    auroc_ci_lower_min: float = 0.60
    balanced_accuracy_min: float = 0.65
    require_beats_null_brier: bool = True
    require_positive_calibration_slope: bool = True
    checks: dict = field(default_factory=dict)

    def evaluate(self, metrics: dict, ci: dict) -> dict:
        lower = float(ci["auroc_ci95"][0])
        slope = float(metrics.get("calibration_slope", float("nan")))
        checks = {
            "auroc_ci_lower_above_min": bool(np.isfinite(lower) and lower > self.auroc_ci_lower_min),
            "balanced_accuracy_at_zero_ok": bool(
                np.isfinite(metrics["balanced_accuracy_at_zero"])
                and metrics["balanced_accuracy_at_zero"] >= self.balanced_accuracy_min
            ),
            "beats_null_brier": bool(metrics["beats_null_brier"]) if self.require_beats_null_brier else True,
            "calibration_slope_positive_finite": bool(np.isfinite(slope) and slope > 0)
            if self.require_positive_calibration_slope
            else True,
        }
        passed = all(checks.values())
        return {
            "checks": checks,
            "passed": passed,
            "verdict": "SENSOR_GATE_PASS" if passed else "SENSOR_GATE_FAIL",
            "thresholds": {
                "auroc_ci_lower_min": self.auroc_ci_lower_min,
                "balanced_accuracy_min": self.balanced_accuracy_min,
            },
        }
