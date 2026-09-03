"""Phases 3-4 -- fit the sensor on train, pick C on tune, confirm on confirm.

D_sensor_confirm is read exactly once, after C is fixed.  The gate thresholds
were frozen in configs/v3_4_0/sensor.json before this script ran, and the
decision boundary is the learned hyperplane itself (d=0), so no post-hoc
threshold is estimated anywhere.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "llm/src"))
from cantor_guard_v340.behavioral_sensor import (  # noqa: E402
    C_GRID,
    SensorGate,
    bootstrap_auroc_ci,
    fit_sensor,
    sensor_metrics,
)

from _common import CONFIG, RESULTS, read_json, write_json  # noqa: E402


def load(split: str):
    table = pd.read_csv(RESULTS / "raw" / f"clean_{split}.csv")
    H = np.load(RESULTS / "cache" / f"h_{split}.npy")
    if len(table) != H.shape[0]:
        raise SystemExit(f"{split}: residual/label length mismatch")
    return H, table.y_safe.to_numpy(dtype=int), table


def main() -> None:
    cfg = read_json(CONFIG / "sensor.json")
    H_tr, y_tr, _ = load("D_sensor_train")
    H_tu, y_tu, _ = load("D_sensor_tune")
    sweep = []
    for C in C_GRID:
        sensor = fit_sensor(H_tr, y_tr, C=C)
        train_m = sensor_metrics(sensor, H_tr, y_tr)
        tune_m = sensor_metrics(sensor, H_tu, y_tu, y_train=y_tr)
        sweep.append({
            "C": float(C), "w_norm": sensor.w_norm,
            "train_auroc": train_m["auroc"], "tune_auroc": tune_m["auroc"],
            "tune_balanced_accuracy_at_zero": tune_m["balanced_accuracy_at_zero"],
            "tune_brier": tune_m["brier"], "tune_null_brier": tune_m["null_brier"],
            "generalization_gap": train_m["auroc"] - tune_m["auroc"],
        })
        print(f"C={C:<8g} |w|={sensor.w_norm:8.4f} train_auroc={train_m['auroc']:.3f} "
              f"tune_auroc={tune_m['auroc']:.3f} gap={sweep[-1]['generalization_gap']:+.3f}", flush=True)
    best = max(sweep, key=lambda row: row["tune_auroc"])
    C_star = float(best["C"])
    print(f"\nselected C={C_star} on D_sensor_tune (tune AUROC {best['tune_auroc']:.3f})")

    sensor = fit_sensor(H_tr, y_tr, C=C_star)
    np.save(RESULTS / "cache" / "sensor_w.npy", sensor.w)
    H_cf, y_cf, _ = load("D_sensor_confirm")
    confirm = sensor_metrics(sensor, H_cf, y_cf, y_train=y_tr)
    ci = bootstrap_auroc_ci(sensor, H_cf, y_cf,
                            n_boot=int(cfg["GATE"]["bootstrap"]["n_boot"]),
                            seed=int(cfg["GATE"]["bootstrap"]["seed"]))
    gate = SensorGate(
        auroc_ci_lower_min=float(cfg["GATE"]["auroc_ci_lower_min"]),
        balanced_accuracy_min=float(cfg["GATE"]["balanced_accuracy_at_zero_min"]),
    ).evaluate(confirm, ci)
    verdict = "SENS2_REFUSAL_SENSOR_ONLY" if gate["passed"] else "SENS3_SENSOR_NOT_GENERALIZABLE"
    payload = {
        "C_grid": list(C_GRID), "sweep": sweep, "C_selected": C_star,
        "selected_on": "D_sensor_tune", "w_norm": sensor.w_norm, "b": sensor.b,
        "d_model": int(sensor.w.size), "n_train": int(y_tr.size),
        "train_base_rate": float(y_tr.mean()),
        "confirm": confirm, "confirm_auroc_ci": ci, "gate": gate,
        "verdict": verdict,
        "label_scope": "SENS2_REFUSAL_SENSOR_ONLY (semantic evaluator gate failed)",
    }
    write_json(RESULTS / "tables" / "sensor_confirm.json", payload)
    print(f"\nCONFIRM n={confirm['n']} base_rate={confirm['base_rate']:.3f}")
    print(f"  AUROC        {confirm['auroc']:.4f}   95% CI {ci['auroc_ci95'][0]:.4f}..{ci['auroc_ci95'][1]:.4f}")
    print(f"  PR-AUC       {confirm['pr_auc']:.4f}")
    print(f"  bal.acc@d=0  {confirm['balanced_accuracy_at_zero']:.4f}")
    print(f"  Brier        {confirm['brier']:.4f}  (null {confirm['null_brier']:.4f})")
    print(f"  calib slope  {confirm.get('calibration_slope', float('nan')):.4f}")
    for name, ok in gate["checks"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
