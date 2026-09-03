"""Phase 5 -- is the learned hyperplane stable, or an artifact of 180 prompts?

In 896 dimensions with n=180 no coefficient is individually identified, so
per-coefficient stability is the wrong question (section 17).  What matters is
whether the ORIENTATION of the hyperplane and its PREDICTIONS are reproducible
across resampled training sets.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "llm/src"))
from cantor_guard_v340.behavioral_sensor import fit_sensor  # noqa: E402
from cantor_guard_v340.sensor_distance import SensorHyperplane  # noqa: E402

from _common import CONFIG, RESULTS, read_json, write_json  # noqa: E402


def main() -> None:
    cfg = read_json(CONFIG / "sensor.json")
    fit = read_json(RESULTS / "tables" / "sensor_confirm.json")
    C = float(fit["C_selected"])
    n_boot, seed = int(cfg["stability"]["n_boot"]), int(cfg["stability"]["seed"])

    train = pd.read_csv(RESULTS / "raw" / "clean_D_sensor_train.csv")
    H_tr = np.load(RESULTS / "cache" / "h_D_sensor_train.npy")
    y_tr = train.y_safe.to_numpy(dtype=int)
    confirm = pd.read_csv(RESULTS / "raw" / "clean_D_sensor_confirm.csv")
    H_cf = np.load(RESULTS / "cache" / "h_D_sensor_confirm.npy")
    y_cf = confirm.y_safe.to_numpy(dtype=int)

    full = SensorHyperplane(np.load(RESULTS / "cache" / "sensor_w.npy"), float(fit["b"]))
    d_full = np.atleast_1d(full.distance(H_cf))
    decision_full = (d_full > 0).astype(int)

    rng = np.random.default_rng(seed)
    cosines, aurocs, agreements, distances = [], [], [], []
    for _ in range(n_boot):
        rows = rng.integers(0, y_tr.size, size=y_tr.size)
        if y_tr[rows].min() == y_tr[rows].max():
            continue
        boot = fit_sensor(H_tr[rows], y_tr[rows], C=C)
        cosines.append(float(boot.w_hat @ full.w_hat))
        d_boot = np.atleast_1d(boot.distance(H_cf))
        aurocs.append(float(roc_auc_score(y_cf, d_boot)))
        agreements.append(float(np.mean((d_boot > 0).astype(int) == decision_full)))
        distances.append(d_boot)
    distances = np.asarray(distances)
    per_prompt_sd = distances.std(axis=0, ddof=1)

    def q(values):
        arr = np.asarray(values, dtype=float)
        return {"mean": float(arr.mean()),
                "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]}

    payload = {
        "n_boot_effective": int(len(cosines)), "C": C,
        "cosine_to_full_fit": q(cosines),
        "heldout_auroc": q(aurocs),
        "decision_agreement_with_full_fit": q(agreements),
        "confirm_signed_distance_sd": {
            "mean_over_prompts": float(per_prompt_sd.mean()),
            "median": float(np.median(per_prompt_sd)),
            "max": float(per_prompt_sd.max()),
            "relative_to_clean_spread": float(per_prompt_sd.mean() / d_full.std(ddof=1)),
        },
        "note": "Per-coefficient stability is not required and not reported: with "
                "n=180 in 896 dimensions individual weights are unidentified. "
                "Orientation and prediction stability are what the controller needs.",
    }
    write_json(RESULTS / "tables" / "sensor_stability.json", payload)
    for key in ("cosine_to_full_fit", "heldout_auroc", "decision_agreement_with_full_fit"):
        row = payload[key]
        print(f"{key:<36}{row['mean']:.4f}  [{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}]")
    sd = payload["confirm_signed_distance_sd"]
    print(f"{'confirm distance sd (mean)':<36}{sd['mean_over_prompts']:.4f}"
          f"  = {sd['relative_to_clean_spread']:.2f} x clean spread")


if __name__ == "__main__":
    main()
