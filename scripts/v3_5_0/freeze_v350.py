"""Phase 10: seal the prospective V3.5.0 protocol before final generation."""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

ROOT_PATH = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH / "llm/src"))

from cantor_guard_v350.one_sided_cantor import epsilon_r, epsilon_r_cantor
from _common import (ACTUATOR_SHA256, ALPHA, CONFIG, Q_CAP, Q_TARGET, RESULTS,
                     RHOS, SENSOR_SHA256, ROOT, read_json, rho_key, write_json)


def main() -> None:
    existing = read_json(CONFIG / "PRE_ANALYSIS_FREEZE.json")
    if existing.get("status") == "PRE_ANALYSIS_FROZEN": raise SystemExit("V3.5.0 is already frozen")
    final_paths = [RESULTS / "raw/final_D_final_v350_harmful.csv",
                   RESULTS / "raw/utility_D_final_v350_benign.csv"]
    if any(p.exists() for p in final_paths): raise SystemExit("final data already touched; refuse to freeze")
    audit = read_json(RESULTS / "tables/v340r_audit.json")
    leakage = read_json(RESULTS / "tables/historical_leakage_audit.json")
    proof = read_json(RESULTS / "tables/risk_lipschitz_proof.json")
    cal = read_json(RESULTS / "tables/risk_window_calibration.json")
    budget = read_json(RESULTS / "tables/budget_calibration.json")
    if budget["verdict"] != "BUD1_MATCHED":
        write_json(CONFIG / "PRE_ANALYSIS_FREEZE.json", {
            "version": "3.5.0", "status": "NOT_FROZEN_BUDGET_IMPOSSIBLE",
            "reason": "q_target_rms=.03 is mathematically unattainable with q_cap=.05 and the frozen zero-action safe-side policy on the preregistered attacked-state distribution",
            "budget_verdict": budget["verdict"], "q_target_rms": Q_TARGET, "q_cap": Q_CAP,
            "per_arm_maximum_attainable_q_rms": {k: v.get("maximum_attainable_q_rms") for k, v in budget["per_arm"].items()},
            "attack_grid": read_json(CONFIG / "controller.json")["attack_grid"],
            "D_final_v350_touched": False, "controller_final_testing_allowed": False,
        })
        raise SystemExit("budget hard stop: V3.5.0 cannot be frozen or proceed to final generation")
    dev = read_json(RESULTS / "tables/attack_dev.json")
    if not audit["passed"] or not leakage["passed"] or proof["violations"] or not dev["passed"]:
        raise SystemExit("a hard pre-freeze gate failed")
    cfg = read_json(CONFIG / "controller.json"); splits = read_json(CONFIG / "splits.json"); stats = read_json(CONFIG / "statistics.json")
    W_R = float(cal["W_R"])
    payload = {
        "version": "3.5.0", "status": "PRE_ANALYSIS_FROZEN",
        "frozen_before": "D_final_v350 harmful or benign generation",
        "base": "cantor-guard-v3.4.0r", "historical_audit": audit["verdict"],
        "external_dataset": read_json(CONFIG / "external_dataset.json"),
        "splits": {"sizes": splits["sizes"], "blocks": splits["blocks"], "block_sha256": splits["block_sha256"]},
        "sensor": {"model": cfg["model"], "layer": cfg["layer"], "state": cfg["state"],
                   "w_file": cfg["sensor_w"], "b_source": cfg["sensor_b_source"],
                   "sha256": SENSOR_SHA256, "transport": "SENSOR1_EXTERNAL_REFUSAL_SENSOR_INHERITED"},
        "actuator": {"file": cfg["actuator"], "sha256": ACTUATOR_SHA256, "safe_sign": cfg["safe_sign"]},
        "risk_window": {"transform": "x=max(0,-d)", "alpha": ALPHA, "W_R": W_R,
                        "calibration_n": cal["n"], "order_index_one_based": cal["order_index_one_based"],
                        "calibration_empirical_coverage": cal["empirical_coverage"],
                        "safe_rule": "d>=0 -> x=0 -> q=0", "outside_rule": "x>W_R -> action=1"},
        "geometry": {"depth": 3, "rho_family": list(RHOS), "leaf_actions": cfg["leaf_actions"],
                     "guard_rule": cfg["guard_rule"], "outside_risk_action": 1.0,
                     "M3": {rho_key(r): r**2 * (1 - 2*r) for r in RHOS},
                     "epsilon_R": {rho_key(r): epsilon_r(r, W_R) for r in RHOS},
                     "epsilon_R_C": epsilon_r_cantor(W_R), "unique_structural_maximizer": "1/3"},
        "budget": {"split": "D_budget_v350", "q_target_rms": Q_TARGET, "q_cap": Q_CAP,
                   "eta_per_arm": budget["eta_per_arm"], "calibration_verdict": budget["verdict"],
                   "outputs_or_labels_used": False, "clipping_is_selection_gate": False},
        "attacks": {"families": ["ATTACK_W", "ATTACK_V"], "grid_absolute": cfg["attack_grid"],
                    "primary_family": "ATTACK_V", "grid_frozen_before_dev": True},
        "arms": ["CLEAN", "ATTACK_ONLY", "LINEAR", "0.25", "0.28", "0.30", "1/3", "0.36", "0.40", "0.44"],
        "endpoint": {"primary": stats["primary_endpoint"], "semantic_claim_allowed": False,
                     "secondary": "LEX32 refusal proxy"},
        "statistics": stats, "utility_gate": stats["utility_gate"],
        "generation": {"max_new_tokens": cfg["generation_max_new_tokens"], "batch_size": cfg["generation_batch_size"]},
        "success_logic": {"A": "GEO1+BUD1+CTRL1+U1", "B": "SUCCESS_A+LIN1", "C": "SUCCESS_B+RHO1"},
        "D_final_v350_touched": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["freeze_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    write_json(CONFIG / "PRE_ANALYSIS_FREEZE.json", payload)
    print(f"PRE_ANALYSIS_FROZEN W_R={W_R:.9f} epsilon_R,C={W_R/27:.9f}")


if __name__ == "__main__": main()
