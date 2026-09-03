"""Phase 6 -- the central V3.4.0 ablation, on D_sensor_confirm.

OLD (V3.3.5c): the coordinate is the projection on the ACTUATOR direction,
    z_v = <h, v>, with a scalar threshold fitted by 1-D logistic regression.
NEW (V3.4.0):  the coordinate is signed distance to a hyperplane learned from
    behaviour, d_w = (w^T h + b)/||w||, whose boundary is d=0 by construction.

Both are fitted on D_sensor_train only and read D_sensor_confirm once, so the
comparison is like-for-like: same prompts, same labels, same held-out split.
No intervention is applied anywhere in this script.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "llm/src"))
from cantor_guard_v340.behavioral_sensor import sensor_metrics  # noqa: E402
from cantor_guard_v340.sensor_distance import SensorHyperplane  # noqa: E402

from _common import CONFIG, RESULTS, read_json, write_json  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load(split: str):
    table = pd.read_csv(RESULTS / "raw" / f"clean_{split}.csv")
    return np.load(RESULTS / "cache" / f"h_{split}.npy"), table.y_safe.to_numpy(dtype=int)


def paired_bootstrap(y, d_new, d_old, *, n_boot: int, seed: int) -> dict:
    """One resample matrix reused for both coordinates, preserving pairing."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, y.size, size=(int(n_boot), y.size))
    diffs = []
    for row in idx:
        yr = y[row]
        if yr.min() == yr.max():
            continue
        diffs.append(roc_auc_score(yr, d_new[row]) - roc_auc_score(yr, d_old[row]))
    arr = np.asarray(diffs, dtype=float)
    return {
        "auroc_difference_mean": float(arr.mean()),
        "auroc_difference_ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "fraction_new_better": float(np.mean(arr > 0)),
        "n_effective": int(arr.size),
    }


def main() -> None:
    protocol = read_json(ROOT / "configs/v3_3_5c/behavioral_protocol.json")
    v = np.load(ROOT / protocol["direction_file"]).astype(float).reshape(-1)
    v = v / np.linalg.norm(v)

    H_tr, y_tr = load("D_sensor_train")
    H_cf, y_cf = load("D_sensor_confirm")

    # OLD coordinate: 1-D logistic on the actuator projection, fitted on train.
    z_tr, z_cf = H_tr @ v, H_cf @ v
    old_1d = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000, class_weight="balanced")
    old_1d.fit(z_tr.reshape(-1, 1), y_tr)
    a, b = float(old_1d.intercept_[0]), float(old_1d.coef_[0][0])
    old_sensor = SensorHyperplane(b * v, a)  # identical decision rule, in residual space
    tau_old = -a / b if b != 0 else float("nan")

    sensor_cfg = read_json(CONFIG / "sensor.json")
    w = np.load(RESULTS / "cache" / "sensor_w.npy")
    fit = read_json(RESULTS / "tables" / "sensor_confirm.json")
    new_sensor = SensorHyperplane(w, float(fit["b"]))

    new_m = sensor_metrics(new_sensor, H_cf, y_cf, y_train=y_tr)
    old_m = sensor_metrics(old_sensor, H_cf, y_cf, y_train=y_tr)
    paired = paired_bootstrap(
        y_cf,
        np.atleast_1d(new_sensor.distance(H_cf)),
        np.atleast_1d(old_sensor.distance(H_cf)),
        n_boot=int(sensor_cfg["GATE"]["bootstrap"]["n_boot"]),
        seed=int(sensor_cfg["GATE"]["bootstrap"]["seed"]),
    )
    cos_wv = float(new_sensor.coupling(v))
    payload = {
        "split": "D_sensor_confirm", "n": int(y_cf.size), "base_rate": float(y_cf.mean()),
        "old_projection": {
            "coordinate": "z_v = <h, v> with 1-D logistic threshold fitted on D_sensor_train",
            "tau": tau_old, "slope": b, **{k: old_m[k] for k in
                ("auroc", "pr_auc", "balanced_accuracy_at_zero", "brier", "null_brier")},
            "calibration_slope": old_m.get("calibration_slope"),
        },
        "new_sensor": {
            "coordinate": "d_w = (w^T h + b)/||w||, boundary d=0 by construction",
            **{k: new_m[k] for k in
                ("auroc", "pr_auc", "balanced_accuracy_at_zero", "brier", "null_brier")},
            "calibration_slope": new_m.get("calibration_slope"),
        },
        "paired_bootstrap_auroc_new_minus_old": paired,
        "cos_w_v": cos_wv,
        "angle_w_v_deg": float(np.degrees(np.arccos(np.clip(cos_wv, -1, 1)))),
        "interpretation": "A large AUROC gap with a small |cos(w,v)| is the direct "
                          "signature of the V3.4.0 hypothesis: the direction that "
                          "senses behavioural state is not the direction that actuates it.",
    }
    write_json(RESULTS / "tables" / "sensor_vs_old_projection.json", payload)
    print(f"{'metric':<28}{'OLD <h,v>':>12}{'NEW d_w':>12}")
    for key in ("auroc", "pr_auc", "balanced_accuracy_at_zero", "brier"):
        print(f"{key:<28}{old_m[key]:>12.4f}{new_m[key]:>12.4f}")
    lo, hi = paired["auroc_difference_ci95"]
    print(f"\npaired AUROC (new - old) = {paired['auroc_difference_mean']:+.4f} "
          f"[{lo:+.4f}, {hi:+.4f}]  new better in {paired['fraction_new_better']:.1%} of resamples")
    print(f"cos(w,v) = {cos_wv:+.4f}  angle = {payload['angle_w_v_deg']:.1f} deg")


if __name__ == "__main__":
    main()
