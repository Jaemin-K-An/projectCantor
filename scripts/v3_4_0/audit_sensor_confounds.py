"""Section 53 -- covariate-shift audit, plus a check for trivial confounds.

A held-out AUROC of 0.93 is only interesting if the sensor is reading
behavioural state rather than a nuisance feature that happens to correlate with
it.  Two questions:

  1. Do simple alternatives (residual norm, prompt length, output length)
     explain the same labels?  If one of them matches the probe, the probe is
     not doing the work it claims.
  2. Does the sensor score distribution shift between the splits the controller
     will actually be deployed on?
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "llm/src"))
from cantor_guard_v340.sensor_distance import SensorHyperplane  # noqa: E402

from _common import CONFIG, RESULTS, read_json, write_json  # noqa: E402

SPLITS = ["D_sensor_train", "D_sensor_tune", "D_sensor_confirm", "D_window_cal",
          "D_controller_budget", "D_attack_dev", "D_final_harmful", "D_final_benign"]


def main() -> None:
    freeze = read_json(CONFIG / "PRE_ANALYSIS_FREEZE.json")
    sensor = SensorHyperplane(np.load(RESULTS / "cache" / "sensor_w.npy"), float(freeze["sensor"]["b"]))
    W = float(freeze["geometry"]["W"])

    confirm = pd.read_csv(RESULTS / "raw" / "clean_D_sensor_confirm.csv")
    H = np.load(RESULTS / "cache" / "h_D_sensor_confirm.npy")
    y = confirm.y_safe.to_numpy(dtype=int)
    d = np.atleast_1d(sensor.distance(H))

    prompts = pd.read_csv(RESULTS / "cache" / "D_sensor_confirm.csv")
    prompt_len = prompts.prompt.astype(str).str.split().str.len().to_numpy(dtype=float)
    alternatives = {
        "sensor_distance_d": d,
        "residual_norm_h": confirm.h_norm.to_numpy(dtype=float),
        "prompt_word_count": prompt_len,
        "output_word_count": confirm.output_length.to_numpy(dtype=float),
    }
    baselines = {}
    for name, score in alternatives.items():
        auroc = float(roc_auc_score(y, score))
        baselines[name] = {
            "auroc": auroc, "auroc_oriented": max(auroc, 1 - auroc),
            "corr_with_d": float(np.corrcoef(score, d)[0, 1]),
        }

    shift = {}
    reference = None
    for split in SPLITS:
        path = RESULTS / "cache" / f"h_{split}.npy"
        if not path.exists():
            continue
        scores = np.atleast_1d(sensor.distance(np.load(path)))
        if reference is None:
            reference = scores
        ks = ks_2samp(scores, reference)
        shift[split] = {
            "n": int(scores.size), "mean": float(scores.mean()), "sd": float(scores.std(ddof=1)),
            "q05": float(np.quantile(scores, 0.05)), "q95": float(np.quantile(scores, 0.95)),
            "inside_window_fraction": float(np.mean(np.abs(scores) <= W)),
            "ks_vs_sensor_train": float(ks.statistic), "ks_p": float(ks.pvalue),
        }
    shifted = [s for s, row in shift.items()
               if row["ks_p"] < 0.01 and row["ks_vs_sensor_train"] > 0.3]
    write_json(RESULTS / "tables" / "sensor_confounds.json", {
        "confound_baselines_on_confirm": baselines,
        "sensor_margin_over_best_alternative": float(
            baselines["sensor_distance_d"]["auroc"]
            - max(v["auroc_oriented"] for k, v in baselines.items() if k != "sensor_distance_d")),
        "covariate_shift": shift,
        "shifted_splits": shifted,
        "verdict": "SENSOR_DISTRIBUTION_SHIFT" if shifted else "NO_MATERIAL_SENSOR_SHIFT",
    })
    print(f"{'score':<24}{'AUROC':>8}{'oriented':>10}{'corr with d':>13}")
    for name, row in baselines.items():
        print(f"{name:<24}{row['auroc']:>8.4f}{row['auroc_oriented']:>10.4f}{row['corr_with_d']:>13.4f}")
    print(f"\n{'split':<22}{'n':>5}{'mean d':>9}{'sd':>8}{'inside W':>10}{'KS':>7}{'p':>9}")
    for split, row in shift.items():
        print(f"{split:<22}{row['n']:>5}{row['mean']:>9.3f}{row['sd']:>8.3f}"
              f"{row['inside_window_fraction']:>10.3f}{row['ks_vs_sensor_train']:>7.3f}{row['ks_p']:>9.3g}")
    print(f"\nshifted splits: {shifted or 'none'}")


if __name__ == "__main__":
    main()
