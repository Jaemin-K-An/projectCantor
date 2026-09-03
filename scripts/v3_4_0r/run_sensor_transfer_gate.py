"""Does the FROZEN V3.4.0 sensor transfer to the new harmful population?

Nothing is refitted. The gate was written into configs/v3_4_0r/controller.json
before this ran. If it fails, V3.4.0R stops: every downstream number would be
uninterpretable.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from cantor_guard_v340.behavioral_sensor import bootstrap_auroc_ci, sensor_metrics  # noqa: E402

from _common import CONFIG, RESULTS, V340, frozen_sensor, read_json, write_json  # noqa: E402


def calibration_fit(score: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Descriptive calibration intercept/slope; never applied to the sensor."""
    score = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=float)

    def loss(params):
        p = np.clip(expit(params[0] + params[1] * score), 1e-12, 1 - 1e-12)
        return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

    fit = minimize(loss, np.array([0.0, 1.0]), method="BFGS")
    if not fit.success and not np.all(np.isfinite(fit.x)):
        return float("nan"), float("nan")
    return float(fit.x[0]), float(fit.x[1])


def main() -> None:
    gate_cfg = read_json(CONFIG / "controller.json")["SENSOR_TRANSFER_GATE"]
    sensor = frozen_sensor()
    table = pd.read_csv(RESULTS / "raw" / "clean_D_sensor_transfer_r.csv")
    H = np.load(RESULTS / "cache" / "h_D_sensor_transfer_r.npy")
    y = table.y_safe.to_numpy(dtype=int)

    metrics = sensor_metrics(sensor, H, y, y_train=y)
    ci = bootstrap_auroc_ci(sensor, H, y, n_boot=20000, seed=34000)
    d = np.atleast_1d(sensor.distance(H))
    calibration_intercept, calibration_slope = calibration_fit(d, y)
    checks = {
        "auroc_at_least_min": bool(metrics["auroc"] >= float(gate_cfg["gate"]["auroc_min"])),
        "auroc_ci_lower_above_min": bool(ci["auroc_ci95"][0] > float(gate_cfg["gate"]["auroc_ci_lower_min"])),
        "balanced_accuracy_ok": bool(metrics["balanced_accuracy_at_zero"]
                                     >= float(gate_cfg["gate"]["balanced_accuracy_at_zero_min"])),
        "both_classes_present": bool(0 < y.mean() < 1),
    }
    passed = all(checks.values())

    # How far has the population moved from the sensor's training distribution?
    old = pd.read_csv(V340 / "raw" / "clean_D_sensor_train.csv")
    old_H = np.load(V340 / "cache" / "h_D_sensor_train.npy")
    old_d = np.atleast_1d(sensor.distance(old_H))
    new_d = d
    payload = {
        "frozen_sensor": True, "nothing_refitted": True,
        "n": int(y.size), "base_rate_safe": float(y.mean()),
        "auroc": metrics["auroc"], "auroc_ci95": ci["auroc_ci95"],
        "pr_auc": metrics["pr_auc"],
        "balanced_accuracy_at_zero": metrics["balanced_accuracy_at_zero"],
        "brier": metrics["brier"], "null_brier": metrics["null_brier"],
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "population_shift": {
            "training_population": "declare-lab/HarmfulQA",
            "new_population": "LLM-LAT/harmful-dataset",
            "train_mean_d": float(old_d.mean()), "train_sd_d": float(old_d.std(ddof=1)),
            "train_safe_rate": float(old.y_safe.mean()),
            "new_mean_d": float(new_d.mean()), "new_sd_d": float(new_d.std(ddof=1)),
            "new_safe_rate": float(y.mean()),
            "mean_shift_in_train_sd": float((new_d.mean() - old_d.mean()) / old_d.std(ddof=1)),
        },
        "gate": gate_cfg["gate"], "checks": checks, "passed": passed,
        "transport_verdict": "ST1_PASS" if passed else "ST2_FAIL",
        "sensor_scope": "SENS2_REFUSAL_SENSOR_ONLY",
        "verdict": "ST1_PASS" if passed else "ST2_FAIL",
    }
    write_json(RESULTS / "tables" / "sensor_transfer.json", payload)
    print(f"n={payload['n']}  safe rate={payload['base_rate_safe']:.3f}")
    print(f"  AUROC        {metrics['auroc']:.4f}  95% CI {ci['auroc_ci95'][0]:.4f}..{ci['auroc_ci95'][1]:.4f}")
    print(f"  PR-AUC       {metrics['pr_auc']:.4f}")
    print(f"  bal.acc@d=0  {metrics['balanced_accuracy_at_zero']:.4f}")
    print(f"  Brier        {metrics['brier']:.4f} (null {metrics['null_brier']:.4f})")
    print(f"  calibration  intercept={calibration_intercept:+.4f} slope={calibration_slope:.4f}")
    shift = payload["population_shift"]
    print(f"\n  population shift: mean d {shift['train_mean_d']:+.3f} -> {shift['new_mean_d']:+.3f} "
          f"({shift['mean_shift_in_train_sd']:+.2f} training SD); "
          f"safe rate {shift['train_safe_rate']:.3f} -> {shift['new_safe_rate']:.3f}")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nVERDICT: {payload['verdict']}")


if __name__ == "__main__":
    main()
