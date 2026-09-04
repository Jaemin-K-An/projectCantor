"""Fit every controller to risk-conditional q RMS=.03 on one common mask."""
from __future__ import annotations

import hashlib
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from cantor_guard_v340.attack import attack_v, attack_w  # noqa: E402
from cantor_guard_v351.linear_risk_controller import LinearRiskController  # noqa: E402
from cantor_guard_v351.risk_budget import fit_eta_risk_conditional, risk_eligibility  # noqa: E402
from cantor_guard_v351.risk_cantor_controller import RiskConditionalCantorController  # noqa: E402
from _common import (CONFIG, Q_CAP, Q_TARGET, RESULTS, RHOS, arm_key,
                     ensure_final_outputs_absent, frozen_actuator, frozen_sensor,
                     read_json, write_json)  # noqa: E402


def attacked_batches(H, sensor, actuator, grid):
    for family in ("ATTACK_V", "ATTACK_W"):
        for epsilon in grid:
            delta = attack_v(actuator, epsilon) if family == "ATTACK_V" else attack_w(sensor, epsilon, sign=-1)
            yield family, epsilon, H + delta[None, :]


def main() -> None:
    ensure_final_outputs_absent()
    output = RESULTS / "tables/budget_calibration.json"
    if output.exists():
        raise SystemExit("risk-conditional budget already calibrated; refuse to refit")
    states = pd.read_csv(RESULTS / "raw/attacked_budget_states.csv")
    H = np.load(RESULTS / "cache/h_D_budget_v351.npy")
    prompts = pd.read_csv(RESULTS / "cache/D_budget_v351.csv")
    cfg = read_json(CONFIG / "controller.json")
    W_R = float(read_json(CONFIG / "risk_coordinate.json")["W_R"])
    grid = list(map(float, read_json(CONFIG / "attack_grid.json")["attack_grid"]))
    sensor, actuator = frozen_sensor(), frozen_actuator()

    common_mask = risk_eligibility(states.d_attacked.to_numpy(float))
    recorded_mask = states.risk_eligible.astype(bool).to_numpy()
    if not np.array_equal(common_mask, recorded_mask):
        raise SystemExit("recorded and recomputed pre-control eligibility masks differ")
    mask_hash = hashlib.sha256(common_mask.astype(np.uint8).tobytes()).hexdigest()

    arms = {
        "LINEAR": LinearRiskController(sensor=sensor, actuator=actuator, W_R=W_R,
                                       eta=1.0, q_cap=Q_CAP),
    }
    arms.update({
        arm_key(rho): RiskConditionalCantorController(
            sensor=sensor, actuator=actuator, W_R=W_R, rho=rho,
            eta=1.0, q_cap=Q_CAP, leaf_actions=cfg["leaf_actions"]
        ) for rho in RHOS
    })

    per_arm, etas, action_rows = {}, {}, []
    all_masks_identical = True
    for name, controller in arms.items():
        pieces = []
        for _family, _epsilon, attacked in attacked_batches(H, sensor, actuator, grid):
            pieces.append(controller.correct(attacked).actions)
        actions = np.concatenate(pieces)
        arm_mask = risk_eligibility(states.d_attacked.to_numpy(float))
        all_masks_identical &= np.array_equal(arm_mask, common_mask)
        eta, metrics = fit_eta_risk_conditional(
            actions, common_mask, target=Q_TARGET, q_cap=Q_CAP)
        if not np.isfinite(eta):
            per_arm[name] = {"attainable": False, **metrics}
            continue
        q_raw = eta * actions
        q_ctrl = np.minimum(q_raw, Q_CAP)
        identity = metrics["risk_q_rms"] * np.sqrt(metrics["risk_eligible_prevalence"])
        metrics.update({
            "attainable": True, "eta": eta, "n_states": len(actions),
            "risk_eligible_n": int(common_mask.sum()),
            "minimum_risk_action": float(actions[common_mask].min()),
            "all_risk_actions_strictly_positive": bool(np.all(actions[common_mask] > 0)),
            "all_safe_actions_exactly_zero": bool(np.all(actions[~common_mask] == 0)),
            "within_relative_3pct": abs(metrics["risk_q_rms"] / Q_TARGET - 1) <= .03,
            "q_cap_ok": metrics["q_max"] <= Q_CAP + 1e-12,
            "global_rms_identity_error": abs(metrics["global_q_rms"] - identity),
            "eligibility_mask_sha256": mask_hash,
        })
        per_arm[name] = metrics
        etas[name] = eta
        arm_rows = states[["pid", "family", "epsilon", "d_attacked", "risk_eligible"]].copy()
        arm_rows["arm"] = name
        arm_rows["action"] = actions
        arm_rows["q_raw"] = q_raw
        arm_rows["q_ctrl"] = q_ctrl
        arm_rows["clipped"] = q_raw > Q_CAP
        action_rows.append(arm_rows)
        print(name, "eta", eta, "risk_rms", metrics["risk_q_rms"],
              "global_rms", metrics["global_q_rms"])

    passed = (all_masks_identical and len(etas) == len(arms) and
              all(row.get("within_relative_3pct") and row.get("q_cap_ok") and
                  row.get("all_risk_actions_strictly_positive") and
                  row.get("all_safe_actions_exactly_zero") and
                  row.get("safe_side_intervention_frequency") == 0
                  for row in per_arm.values()))
    action_table = pd.concat(action_rows, ignore_index=True) if action_rows else pd.DataFrame()
    action_table.to_csv(RESULTS / "raw/budget_controller_actions.csv", index=False)
    payload = {
        "split": "D_budget_v351",
        "distribution": "both frozen attack families at every frozen grid epsilon, including zero",
        "eligibility": "common frozen pre-control d_attacked<0 mask",
        "eligibility_mask_sha256": mask_hash,
        "eligibility_identical_across_all_arms": bool(all_masks_identical),
        "generation_performed": False, "outputs_or_labels_consulted": False,
        "q_target_risk_rms": Q_TARGET, "q_cap": Q_CAP,
        "relative_tolerance": .03,
        "global_rms_is_selection_target": False,
        "finite_set_feasibility_proof": (
            "Every risk-eligible action is strictly positive. As eta tends to infinity, "
            "every eligible q approaches q_cap, hence maximum attainable risk RMS=q_cap=0.05>0.03."
        ),
        "per_arm": per_arm,
        "eta_per_arm": etas if passed else None,
        "verdict": "BUD1_RISK_CONDITIONAL_MATCHED" if passed else "BUD2_RISK_CONDITIONAL_MISMATCH",
    }
    write_json(output, payload)
    cfg["eta_per_arm"] = payload["eta_per_arm"]
    cfg["budget_calibration_artifact"] = str(output.relative_to(ROOT))
    write_json(CONFIG / "controller.json", cfg)
    print(payload["verdict"])
    if not passed:
        raise SystemExit("risk-conditional equal-budget hard gate failed")


if __name__ == "__main__":
    main()
