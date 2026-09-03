"""Phase 20 -- does the realised sensor movement match the analytic prediction?

For ATTACK-W the prediction is exact by Theorem S: ``dd = -eps``.
For ATTACK-V it is ``dd = -eps * kappa``.
For the controller it is ``dd = q_ctrl * ||h|| * kappa``.
Any disagreement means the controller and the model have drifted apart.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "llm/src"))
from _common import CONFIG, RESULTS, read_json, write_json  # noqa: E402


def main(split: str = "D_final_harmful") -> None:
    freeze = read_json(CONFIG / "PRE_ANALYSIS_FREEZE.json")
    kappa = float(freeze["coupling"]["kappa"])
    frame = pd.read_csv(RESULTS / "raw" / f"final_{split}.csv")
    attacked = frame[frame.family != "NONE"].copy()
    attacked["dd_attack_realised"] = attacked.d_attacked - attacked.d_clean
    attacked["dd_attack_predicted"] = np.where(
        attacked.family == "ATTACK_W", -attacked.epsilon, -attacked.epsilon * kappa)
    attacked["dd_ctrl_realised"] = attacked.d_corrected - attacked.d_attacked

    by_family = {}
    for family, group in attacked.groupby("family"):
        err = (group.dd_attack_realised - group.dd_attack_predicted).abs()
        by_family[family] = {
            "n": int(len(group)),
            "max_abs_error": float(err.max()), "median_abs_error": float(err.median()),
            "mean_dd_attack": float(group.dd_attack_realised.mean()),
            "mean_dd_controller": float(group.dd_ctrl_realised.mean()),
            "restoration_fraction": float(
                (group.dd_ctrl_realised.mean() / -group.dd_attack_realised.mean())
                if group.dd_attack_realised.mean() != 0 else float("nan")),
        }

    recovered = attacked.groupby(["family", "rho_key"]).apply(
        lambda g: pd.Series({
            "mean_d_clean": g.d_clean.mean(),
            "mean_d_attacked": g.d_attacked.mean(),
            "mean_d_corrected": g.d_corrected.mean(),
            "fraction_corrected_closer_to_clean": float(np.mean(
                (g.d_corrected - g.d_clean).abs() < (g.d_attacked - g.d_clean).abs())),
        }), include_groups=False).reset_index()

    write_json(RESULTS / "tables" / "mechanism.json", {
        "kappa": kappa, "by_family": by_family,
        "sensor_scores_by_family_and_rho": recovered.to_dict(orient="records"),
        "note": "ATTACK-W error is bounded by float32 hook precision; anything larger "
                "would mean the injected perturbation is not what the analysis assumes.",
    })
    for family, row in by_family.items():
        print(f"{family}: n={row['n']} max|analytic-realised|={row['max_abs_error']:.2e} "
              f"mean dd_attack={row['mean_dd_attack']:+.4f} "
              f"mean dd_ctrl={row['mean_dd_controller']:+.4f} "
              f"restoration={row['restoration_fraction']:.3f}")
    print()
    print(recovered.to_string(index=False))


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
