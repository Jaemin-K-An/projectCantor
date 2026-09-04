"""Mechanical multi-axis V3.5.1 verdict and SUCCESS A/B/C gates."""
from __future__ import annotations

from _common import CONFIG, RESULTS, read_json, write_json


def load(name):
    return read_json(RESULTS / f"tables/{name}.json")


def main() -> None:
    risk = load("risk_conditional_window")
    lipschitz = load("risk_lipschitz_proof")
    certificate = load("certificate_validation")
    budget = load("final_budget_audit")
    controller = load("controller_effect")
    linear = load("linear_comparison")
    rho = load("rho_family")
    utility = load("utility")
    geometry = ("PASS" if lipschitz["violations"] == 0 and
                certificate["verdict"] == "GEO1_ONE_SIDED_CANTOR_CERTIFICATE_VALID" else "FAIL")
    calibration = "PASS" if risk["verdict"] == "RISK1_CONDITIONAL_WINDOW_CALIBRATED" else "FAIL"
    budget_axis = "PASS" if budget["verdict"] == "BUD1_RISK_CONDITIONAL_MATCHED" else "FAIL"
    ctrl = controller["controller_verdict"]
    lin = linear["linear_verdict"]
    rho_axis = rho["rho_verdict"]
    utility_axis = "PASS" if utility["verdict"] == "U1_PASS" else "FAIL"
    success_a = (geometry == "PASS" and calibration == "PASS" and budget_axis == "PASS" and
                 utility_axis == "PASS" and ctrl == "CTRL1_CANTOR_EFFECTIVE")
    success_b = success_a and lin == "LIN1_CANTOR_BEATS_LINEAR"
    success_c = success_b and rho_axis == "RHO1_CANTOR_EMPIRICAL_GAIN"
    overall = ("SUCCESS_C_MIDDLE_THIRD_EMPIRICAL_OPTIMUM" if success_c else
               "SUCCESS_B_CANTOR_ADDS_VALUE" if success_b else
               "SUCCESS_A_CANTOR_CONTROLLER" if success_a else
               "CONFIRMATORY_CRITERIA_NOT_MET")
    payload = {
        "GEOMETRY": geometry, "RISK_CALIBRATION": calibration,
        "SENSOR": "PASS_FROZEN_EXTERNAL_REFUSAL_SENSOR_INHERITED",
        "ACTUATOR": "PASS_FROZEN_CAUSAL_P0_ACTUATOR_INHERITED",
        "BUDGET": budget_axis, "CONTROLLER": ctrl, "LINEAR": lin,
        "RHO": rho_axis, "UTILITY": utility_axis,
        "SEMANTIC_SCOPE": "REFUSAL_ONLY", "OVERALL": overall,
        "SUCCESS_A": success_a, "SUCCESS_B": success_b, "SUCCESS_C": success_c,
        "formal_freeze_status": read_json(CONFIG / "PRE_ANALYSIS_FREEZE.json")["status"],
        "D_final_v351_touched": True,
        "invalidated_runs": ["INVALIDATED_initial_historical_leakage_audit.json"],
        "exploratory_analyses": ["W_R bootstrap uncertainty", "post-confirmatory diagnostics"],
        "semantic_safety_claimed": False,
    }
    write_json(RESULTS / "tables/final_verdict.json", payload)
    for key, value in payload.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
