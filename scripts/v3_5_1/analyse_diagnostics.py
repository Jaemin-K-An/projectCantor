"""Mandatory post-confirmatory diagnostics; never used for retuning."""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import RESULTS, ROOT, frozen_actuator, frozen_sensor, read_json, require_freeze, write_json


def quantiles(values):
    arr = np.asarray(values, float)
    return {str(p): float(np.quantile(arr, p)) for p in (0, .25, .5, .75, .9, .95, 1)}


def safe_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None


def main() -> None:
    freeze = require_freeze()
    frame = pd.read_csv(RESULTS / "raw/final_D_final_v351_harmful.csv")
    cal = read_json(RESULTS / "tables/risk_conditional_window.json")
    old = read_json(ROOT / "results/v3_5_0/tables/risk_window_calibration.json")
    old_W, new_W = float(old["W_R"]), float(cal["W_R"])
    controlled = frame[frame.arm != "ATTACK_ONLY"].copy()

    occupancy = {}
    for arm, group in controlled.groupby("arm", sort=False):
        occupancy[arm] = {
            "cell_type": {str(k): int(v) for k, v in group.cell_type.fillna("none").value_counts().items()},
            "guard_fraction": float(np.mean(group.cell_type == "guard")),
            "outside_risk_window_fraction": float(group.outside_risk_window.astype(bool).mean()),
            "action_quantiles": quantiles(group.action),
            "q_quantiles": quantiles(group.q_ctrl),
            "clipping_fraction": float(group.clipped.astype(bool).mean()),
        }

    switches = []
    cantor = frame[frame.arm.str.startswith("CANTOR_")]
    for (family, arm), group in cantor.groupby(["family", "arm"], sort=False):
        base = group[np.isclose(group.epsilon, 0)][["pid", "cell_type", "cell_index"]].rename(
            columns={"cell_type": "base_type", "cell_index": "base_index"})
        rho = float(group.rho.dropna().iloc[0])
        certificate = float(freeze["geometry"]["epsilon_R"][arm])
        for epsilon, current in group.groupby("epsilon", sort=True):
            merged = current.merge(base, on="pid", how="left")
            denominator = merged.base_type.eq("leaf")
            direct_switch = (denominator & merged.cell_type.eq("leaf") &
                             merged.cell_index.ne(merged.base_index))
            switches.append({
                "family": family, "arm": arm, "rho": rho, "epsilon": float(epsilon),
                "inside_certificate": bool(float(epsilon) < certificate),
                "baseline_leaf_n": int(denominator.sum()),
                "direct_terminal_switch_rate": float(direct_switch[denominator].mean()) if denominator.any() else None,
            })
    switch_frame = pd.DataFrame(switches)
    switch_frame.to_csv(RESULTS / "tables/terminal_switch_diagnostics.csv", index=False)

    auc = pd.read_csv(RESULTS / "tables/refusal_auc_per_prompt.csv")
    auc_means = (auc.groupby(["family", "arm"], sort=False).auc.mean()
                 .rename("mean_auc").reset_index())
    auc_means.to_csv(RESULTS / "tables/refusal_auc_means.csv", index=False)
    sensor, actuator = frozen_sensor(), frozen_actuator()
    coupling = float(sensor.coupling(actuator.v_safe))
    primary_auc = auc_means[auc_means.family == "ATTACK_V"].set_index("arm").mean_auc
    risk_rows = controlled[controlled.risk_eligible.astype(bool)]
    linear_actions = risk_rows[risk_rows.arm == "LINEAR"].action.to_numpy(float)
    cantor_actions = risk_rows[risk_rows.arm == "CANTOR_1/3"].action.to_numpy(float)
    payload = {
        "exploratory_only": True, "used_for_retuning": False,
        "risk_window_change": {
            "W_R_v350": old_W, "W_R_v351": new_W,
            "absolute_change": new_W - old_W, "ratio": new_W / old_W,
            "conditional_coverage_v351": cal["conditional_empirical_coverage"],
            "epsilon_R_C_v350": old_W / 27, "epsilon_R_C_v351": new_W / 27,
            "epsilon_R_C_ratio": new_W / old_W,
        },
        "cantor_cell_and_control_occupancy": occupancy,
        "terminal_switch_table": "results/v3_5_1/tables/terminal_switch_diagnostics.csv",
        "refusal_sensor_relationship": {
            "correlation_d_attacked_y_safe": safe_corr(frame.d_attacked, frame.y_safe),
            "correlation_d_corrected_y_safe": safe_corr(frame.d_corrected, frame.y_safe),
        },
        "sensor_actuator_coupling": {
            "kappa_safe": coupling, "correction_direction_matches_sensor_safe_sign": coupling > 0,
            "mean_expected_sensor_displacement": float(controlled.post_control_expected_sensor_displacement.mean()),
        },
        "matched_mapping_difference": {
            "risk_state_mean_abs_action_difference_linear_vs_cantor_1_3": float(np.mean(np.abs(linear_actions - cantor_actions))),
            "ATTACK_V_mean_auc_CANTOR_1_3": float(primary_auc["CANTOR_1/3"]),
            "ATTACK_V_mean_auc_LINEAR": float(primary_auc["LINEAR"]),
            "ATTACK_V_auc_difference": float(primary_auc["CANTOR_1/3"] - primary_auc["LINEAR"]),
            "interpretation": "Behavioral differences, if any, arise from the frozen risk-to-action mapping because sensor, actuator, cap, grid, prompts and conditional RMS target are shared.",
        },
        "middle_third_geometry_behavior_alignment": {
            "certificate_is_unique_family_maximum": True,
            "empirical_rho_verdict": read_json(RESULTS / "tables/rho_family.json")["rho_verdict"],
        },
        "semantic_scope": "REFUSAL_ONLY",
    }
    write_json(RESULTS / "tables/diagnostics.json", payload)
    print("mandatory diagnostics complete")


if __name__ == "__main__":
    main()
