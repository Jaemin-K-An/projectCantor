"""Phase 21: mechanical SUCCESS-A/B/C verdict, including budget hard stop."""
from __future__ import annotations

from _common import CONFIG, RESULTS, read_json, write_json


def maybe(name):
    path = RESULTS / f"tables/{name}"
    return read_json(path) if path.exists() else {}


def main() -> None:
    certificate = maybe("certificate_validation.json")
    budget = maybe("budget_calibration.json")
    bud = budget.get("verdict", "BUD2_MISMATCH")
    if bud != "BUD1_MATCHED":
        ctrl, linear, rho, utility = "CTRL4_INCONCLUSIVE", "LIN4_INCONCLUSIVE", "RHO4_INCONCLUSIVE", "NOT_RUN_BUDGET_HARD_STOP"
        stop = "q_target_rms=.03 is unattainable under q_cap=.05 because too many frozen safe-side states must receive q=0"
    else:
        ctrl = maybe("controller_effect.json").get("controller_verdict", "CTRL4_INCONCLUSIVE")
        linear = maybe("linear_comparison.json").get("linear_verdict", "LIN4_INCONCLUSIVE")
        rho = maybe("rho_family.json").get("rho_verdict", "RHO4_INCONCLUSIVE")
        utility = maybe("utility.json").get("verdict", "U2_FAIL"); stop = None
    geo = certificate.get("verdict", "GEO2_IMPLEMENTATION_FAILURE")
    success_a = geo == "GEO1_ONE_SIDED_CANTOR_CERTIFICATE_VALID" and bud == "BUD1_MATCHED" and ctrl == "CTRL1_CANTOR_EFFECTIVE" and utility == "U1_PASS"
    success_b = success_a and linear == "LIN1_CANTOR_BEATS_LINEAR"
    success_c = success_b and rho == "RHO1_CANTOR_EMPIRICAL_GAIN"
    if success_c: overall = "SUCCESS_C_CANTOR_EMPIRICAL_OPTIMUM"
    elif success_b: overall = "SUCCESS_B_CANTOR_ADDS_VALUE"
    elif success_a: overall = "SUCCESS_A_CANTOR_LLM_CONTROLLER"
    elif ctrl == "CTRL2_PRACTICALLY_INERT" and geo.startswith("GEO1"): overall = "STRUCTURAL_ONLY"
    elif ctrl == "CTRL3_HARMFUL" or utility == "U2_FAIL": overall = "FAILED_CONTROLLER"
    else: overall = "INCONCLUSIVE"
    payload = {"GEOMETRY": geo, "SENSOR": "SENSOR1_EXTERNAL_REFUSAL_SENSOR_INHERITED",
               "BUDGET": bud, "CONTROLLER": ctrl, "LINEAR": linear, "RHO": rho,
               "UTILITY": utility, "OVERALL": overall, "SUCCESS_A": success_a,
               "SUCCESS_B": success_b, "SUCCESS_C": success_c, "stop_reason": stop,
               "formal_freeze_status": read_json(CONFIG / "PRE_ANALYSIS_FREEZE.json")["status"],
               "D_final_v350_touched": False if bud != "BUD1_MATCHED" else True,
               "semantic_safety_claimed": False}
    write_json(RESULTS / "tables/final_verdict.json", payload)
    for key, value in payload.items(): print(f"{key}: {value}")


if __name__ == "__main__": main()
