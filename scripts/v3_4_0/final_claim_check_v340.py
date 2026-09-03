"""Phase 22 -- mechanical verdict.  Every label follows from a stored number.

Nothing here re-reads raw data or applies judgement: each verdict is a function
of the frozen gates and the tables the earlier phases wrote.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "llm/src"))
from _common import CONFIG, RESULTS, read_json, write_json  # noqa: E402


def _maybe(name):
    path = RESULTS / "tables" / name
    return read_json(path) if path.exists() else None


def main() -> None:
    sensor = _maybe("sensor_confirm.json")
    evaluator = _maybe("semantic_evaluator_validation.json")
    actuator = _maybe("actuator_validation.json")
    geom = _maybe("sensor_window_and_coupling.json")
    cert = _maybe("certificate_validation.json")
    budget = _maybe("final_budget_audit.json")
    generation = _maybe("generation_analysis.json")
    utility = _maybe("utility.json")

    # SENSOR
    if sensor is None:
        sensor_v = "SENS3_SENSOR_NOT_GENERALIZABLE"
    elif not sensor["gate"]["passed"]:
        sensor_v = "SENS3_SENSOR_NOT_GENERALIZABLE"
    elif evaluator is not None and evaluator["verdict"] == "SEM1_INDEPENDENT_EVALUATOR_VALID":
        sensor_v = "SENS1_VALID_BEHAVIORAL_SENSOR"
    else:
        sensor_v = "SENS2_REFUSAL_SENSOR_ONLY"

    # SEMANTIC
    semantic_v = evaluator["verdict"] if evaluator else "SEM3_PROXY_ONLY"

    # ACTUATOR / COUPLING / CERTIFICATE
    actuator_v = actuator["verdict"] if actuator else "ACT2_ACTUATOR_NOT_REPLICATED"
    coupling_v = geom["controllability"]["verdict"] if geom else "COUP2_WEAK_SENSOR_ACTUATOR_COUPLING"
    cert_v = cert["verdict"] if cert else "CERT0_NOT_RUN"

    # GENERATION
    if generation is None:
        generation_v = "GEN0_NOT_RUN"
    else:
        primary = generation["by_family"].get("ATTACK_V", {})
        if not primary.get("max_t", {}).get("contrasts"):
            generation_v = "GEN5_INCONCLUSIVE"
        elif primary["all_favour_cantor"]:
            generation_v = "GEN1_CANTOR_SEMANTIC_GAIN" if semantic_v == "SEM1_INDEPENDENT_EVALUATOR_VALID" \
                else "GEN4_REFUSAL_ONLY_RESULT"
        elif primary["any_favours_other"]:
            generation_v = "GEN3_OTHER_RHO_BETTER"
        elif primary["all_within_sesoi"]:
            generation_v = "GEN2_RHO_FAMILY_PRACTICALLY_EQUIVALENT"
        else:
            generation_v = "GEN5_INCONCLUSIVE"

    utility_v = utility["verdict"] if utility else "U0_NOT_RUN"

    # OVERALL
    architecture_ok = (
        sensor_v in ("SENS1_VALID_BEHAVIORAL_SENSOR", "SENS2_REFUSAL_SENSOR_ONLY")
        and actuator_v == "ACT1_CAUSAL_ACTUATOR_REPLICATED"
        and coupling_v == "COUP1_CONTROLLABLE"
        and cert_v == "CERT1_CANTOR_SENSOR_CERTIFICATE_VALID"
        and budget is not None and budget["all_matched"]
        and utility_v == "U1_PASS"
    )
    if sensor_v == "SENS3_SENSOR_NOT_GENERALIZABLE":
        overall = "E_LINEAR_BEHAVIORAL_SENSOR_NOT_SUPPORTED"
    elif actuator_v == "ACT2_ACTUATOR_NOT_REPLICATED":
        overall = "E_LINEAR_BEHAVIORAL_SENSOR_NOT_SUPPORTED"
    elif coupling_v == "COUP2_WEAK_SENSOR_ACTUATOR_COUPLING":
        overall = "F_SENSOR_VALID_ACTUATOR_COUPLING_TOO_WEAK"
    elif not architecture_ok:
        overall = "G_INCONCLUSIVE"
    elif semantic_v == "SEM1_INDEPENDENT_EVALUATOR_VALID" and generation_v == "GEN1_CANTOR_SEMANTIC_GAIN":
        overall = "B_CANTOR_CERTIFICATE_AND_SEMANTIC_GAIN"
    elif semantic_v == "SEM1_INDEPENDENT_EVALUATOR_VALID" and generation_v == "GEN2_RHO_FAMILY_PRACTICALLY_EQUIVALENT":
        overall = "C_CANTOR_CERTIFIED_SEMANTICALLY_EQUIVALENT"
    elif semantic_v == "SEM1_INDEPENDENT_EVALUATOR_VALID" and generation_v == "GEN3_OTHER_RHO_BETTER":
        overall = "D_CANTOR_CERTIFIED_OTHER_RHO_SEMANTICALLY_BETTER"
    else:
        overall = "A_SENSOR_ACTUATOR_CANTOR_CONTROLLER_SUPPORTED"

    payload = {
        "SENSOR": sensor_v, "ACTUATOR": actuator_v, "COUPLING": coupling_v,
        "CERTIFICATE": cert_v, "SEMANTIC": semantic_v, "GENERATION": generation_v,
        "UTILITY": utility_v, "OVERALL": overall,
        "architecture_complete": architecture_ok,
        "semantic_claim_allowed": semantic_v == "SEM1_INDEPENDENT_EVALUATOR_VALID",
        "structural_claim": "For the frozen sensor hyperplane and fixed window W, "
                            "epsilon_h(rho) = 2W rho^2 (1-2rho) is uniquely maximised at rho=1/3",
        "structural_claim_is_conditional_on": [
            "the learned sensor", "model qwen2.5-0.5b-instruct layer 14",
            "the P0 state definition", "the fixed W", "the depth-3 policy geometry"],
        "llm_is_not_claimed_to_be_fractal": True,
    }
    write_json(RESULTS / "tables" / "final_verdict.json", payload)
    for key, value in payload.items():
        if isinstance(value, str) and key.isupper():
            print(f"{key:<12} {value}")
    print(f"\narchitecture complete: {architecture_ok}")


if __name__ == "__main__":
    main()
