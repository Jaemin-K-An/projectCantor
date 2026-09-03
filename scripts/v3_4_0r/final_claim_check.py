"""Phase 21 -- mechanical verdict. Budget validity dominates everything."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v3_4_0r"))
from patch_claim_classifier import overall_verdict  # noqa: E402

from _common import RESULTS, write_json  # noqa: E402


def _maybe(name):
    from _common import read_json

    path = RESULTS / "tables" / name
    return read_json(path) if path.exists() else None


def main() -> None:
    transfer = _maybe("sensor_transfer.json")
    evaluator = _maybe("semantic_evaluator_validation.json")
    budget = _maybe("final_budget_audit.json")
    effect = _maybe("controller_effect.json")
    rho = _maybe("rho_family.json")
    utility = _maybe("utility.json")
    cert = _maybe("certificate_validation.json")

    sensor_v = transfer["verdict"] if transfer else "SENS3_FAILED"
    if evaluator and evaluator["verdict"] == "SEM1_VALID" and sensor_v != "SENS3_FAILED":
        sensor_v = "SENS1_SEMANTICALLY_VALIDATED"
    semantic_v = evaluator["verdict"] if evaluator else "SEM3_PROXY_ONLY"
    actuator_v = "ACT1_REPLICATED"  # revalidated in V3.4.0 and reused frozen
    cert_v = ("CERT1_VALID" if cert and cert["verdict"].startswith("CERT1")
              else ("CERT2_IMPLEMENTATION_FAILURE" if cert else "CERT0_NOT_RUN"))
    budget_v = budget["verdict"] if budget else "BUD2_MISMATCH"
    controller_v = effect["controller_verdict"] if effect else "CTRL3_INCONCLUSIVE"
    cantor_v = rho["cantor_verdict"] if rho else "CANTOR5_INCONCLUSIVE"
    generation_v = rho["generation_verdict"] if rho else "GEN5_INCONCLUSIVE"
    utility_v = utility["verdict"] if utility else "U0_NOT_RUN"

    overall = overall_verdict(certificate=cert_v, budget=budget_v, controller=controller_v,
                              cantor=cantor_v, utility=utility_v)
    payload = {
        "SENSOR": sensor_v, "ACTUATOR": actuator_v, "CERTIFICATE": cert_v,
        "BUDGET": budget_v, "CONTROLLER_EFFECT": controller_v,
        "CANTOR_BEHAVIOR": cantor_v, "GENERATION": generation_v,
        "SEMANTIC": semantic_v, "UTILITY": utility_v, "OVERALL": overall,
        "semantic_claim_allowed": semantic_v == "SEM1_VALID",
        "structural_claim": "rho=1/3 uniquely maximises epsilon_h(rho)=2W rho^2(1-2rho): a "
                            "STRUCTURAL POLICY-SEPARATION OPTIMUM, not an empirical LLM safety optimum",
        "certificate_scope": "certified residual-L2 radius against direct terminal-policy "
                             "switching in the frozen sensor coordinate",
        "llm_is_not_claimed_to_be_fractal": True,
        "population_note": "harmful population changed from HarmfulQA to LLM-LAT because the "
                           "former is exhausted; the frozen sensor was gated for transfer first",
    }
    write_json(RESULTS / "tables" / "final_verdict.json", payload)
    for key, value in payload.items():
        if isinstance(value, str) and key.isupper():
            print(f"{key:<18} {value}")


if __name__ == "__main__":
    main()
