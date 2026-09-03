"""Secondary robustness check: does the sensor depend on WHICH labeller was used?

The primary sensor predicts labels from LibrAI/longformer-action-ro. If the
result were an artifact of that particular labeller, a sensor trained on the
independent LEX32 lexical labels should not generalise. Both are refitted with
the frozen recipe and evaluated on the same held-out prompts.

This is a secondary analysis. It does not change the primary sensor, which was
frozen before D_final was opened.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "llm/src"))
from cantor_guard_v340.behavioral_sensor import fit_sensor, sensor_metrics  # noqa: E402

from _common import CONFIG, RESULTS, read_json, write_json  # noqa: E402


def main() -> None:
    C = float(read_json(RESULTS / "tables" / "sensor_confirm.json")["C_selected"])
    train = pd.read_csv(RESULTS / "raw" / "clean_D_sensor_train.csv")
    confirm = pd.read_csv(RESULTS / "raw" / "clean_D_sensor_confirm.csv")
    H_tr = np.load(RESULTS / "cache" / "h_D_sensor_train.npy")
    H_cf = np.load(RESULTS / "cache" / "h_D_sensor_confirm.npy")

    out = {}
    sensors = {}
    for name, column in [("action-ro (primary)", "y_safe"), ("LEX32 (independent)", "refusal_proxy")]:
        y_tr, y_cf = train[column].to_numpy(dtype=int), confirm[column].to_numpy(dtype=int)
        s = fit_sensor(H_tr, y_tr, C=C)
        sensors[name] = s
        m = sensor_metrics(s, H_cf, y_cf, y_train=y_tr)
        out[name] = {
            "label_column": column, "train_base_rate": float(y_tr.mean()),
            "confirm_auroc": m["auroc"], "confirm_balanced_accuracy": m["balanced_accuracy_at_zero"],
            "confirm_brier": m["brier"], "null_brier": m["null_brier"],
        }
    a, b = sensors["action-ro (primary)"], sensors["LEX32 (independent)"]
    out["cosine_between_the_two_hyperplanes"] = float(a.w_hat @ b.w_hat)
    # cross-evaluation: does each sensor predict the OTHER labeller's labels?
    out["cross_auroc"] = {
        "action-ro sensor -> LEX32 labels":
            float(roc_auc_score(confirm.refusal_proxy, np.atleast_1d(a.distance(H_cf)))),
        "LEX32 sensor -> action-ro labels":
            float(roc_auc_score(confirm.y_safe, np.atleast_1d(b.distance(H_cf)))),
    }
    out["conclusion"] = (
        "Both labellers yield a sensor that generalises, and the two hyperplanes point "
        "in nearly the same direction, so the result is a property of the residual "
        "state rather than of one labeller."
        if out["cosine_between_the_two_hyperplanes"] > 0.7
        else "The two labellers give materially different hyperplanes; the primary "
             "sensor result is labeller-dependent and must be reported as such.")
    write_json(RESULTS / "tables" / "label_source_robustness.json", out)
    for name in ("action-ro (primary)", "LEX32 (independent)"):
        row = out[name]
        print(f"{name:<24} base={row['train_base_rate']:.3f} "
              f"confirm AUROC={row['confirm_auroc']:.4f} balacc={row['confirm_balanced_accuracy']:.4f}")
    print(f"\ncosine between hyperplanes = {out['cosine_between_the_two_hyperplanes']:.4f}")
    for key, value in out["cross_auroc"].items():
        print(f"  {key}: {value:.4f}")
    print(f"\n{out['conclusion']}")


if __name__ == "__main__":
    main()
