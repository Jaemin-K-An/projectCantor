"""Mechanical, evidence-limited V3.3.5c verdict classifier."""
from __future__ import annotations

import pathlib

from _common import CONFIG, RESULTS, read_json, write_json


def overall_verdict(*, behavioral: str, certificate: str, budgets: str, utility: str, generation: str, evaluator: str, final_ran: bool) -> str:
    anchored = (
        behavioral in {"B1_P0_BEHAVIORAL_BOUNDARY_IDENTIFIED", "B2_NONPARAMETRIC_BOUNDARY_ONLY"}
        and certificate == "C1_CANTOR_P0_CERTIFICATE_VALID"
        and budgets == "BUDGET_MATCHED"
        and utility == "U1_PASS"
        and final_ran
    )
    if behavioral == "B3_BOUNDARY_UNIDENTIFIABLE":
        return "E_P0_BEHAVIORAL_ANCHOR_NOT_REPLICATED"
    if not anchored:
        return "F_INCONCLUSIVE"
    if evaluator == "S1_VALID" and generation == "G1_CANTOR_SEMANTIC_GAIN":
        return "B_CANTOR_CERTIFICATE_AND_SEMANTIC_GAIN"
    if evaluator == "S1_VALID" and generation == "G2_RHO_FAMILY_PRACTICALLY_EQUIVALENT":
        return "C_CANTOR_CERTIFIED_BUT_SEMANTICALLY_EQUIVALENT"
    if evaluator == "S1_VALID" and generation == "G3_OTHER_RHO_BETTER":
        return "D_CANTOR_CERTIFIED_OTHER_RHO_SEMANTICALLY_BETTER"
    return "A_CANTOR_BEHAVIORALLY_ANCHORED_LLM_CONTROLLER_SUPPORTED"


def _optional(path: pathlib.Path, default: dict) -> dict:
    return read_json(path) if path.exists() else default


def main() -> None:
    temporal = read_json(RESULTS / "tables/temporal_correction.json")
    boundary_path = RESULTS / "tables/p0_behavioral_boundary.json"
    boundary = _optional(boundary_path, {"verdict": "NOT_RUN"})
    evaluator = read_json(CONFIG / "evaluator.json")
    certificate = _optional(RESULTS / "tables/certificate_validation.json", {"verdict": "C3_WINDOW_APPLICABILITY_FAILURE"})
    budgets = _optional(RESULTS / "tables/final_budget_audit.json", {"status": "NOT_RUN"})
    generation_path = RESULTS / "tables/generation_analysis.json"
    if generation_path.exists():
        generation = read_json(generation_path)
    elif boundary.get("verdict") == "B3_BOUNDARY_UNIDENTIFIABLE":
        generation = {"verdict": "G6_NOT_RUN_BEHAVIORAL_GATE"}
    else:
        generation = {"verdict": "G5_INCONCLUSIVE"}
    utility = _optional(RESULTS / "tables/utility.json", {"verdict": "U3_NOT_RUN"})
    final_ran = (RESULTS / "raw/final_p0_cantor.csv").exists()
    behavioral_label = boundary.get("verdict", "B3_BOUNDARY_UNIDENTIFIABLE")
    if behavioral_label == "NOT_RUN":
        behavioral_label = "B3_BOUNDARY_UNIDENTIFIABLE"
    overall = overall_verdict(
        behavioral=behavioral_label,
        certificate=certificate["verdict"], budgets=budgets["status"],
        utility=utility["verdict"], generation=generation["verdict"],
        evaluator=evaluator["status"], final_ran=final_ran,
    )
    payload = {
        "TEMPORAL_CORRECTION": temporal["verdict"],
        "BEHAVIORAL": behavioral_label,
        "SEMANTIC_EVALUATOR": evaluator["status"],
        "CERTIFICATE": certificate["verdict"],
        "GENERATION": generation["verdict"],
        "UTILITY": utility["verdict"],
        "OVERALL": overall,
        "structural_claim": "rho=1/3 uniquely maximises 2W*rho^2*(1-2rho) in the fixed affine depth-3 family",
        "semantic_claim_allowed": evaluator["status"] == "S1_VALID" and generation["verdict"] in {"G1_CANTOR_SEMANTIC_GAIN", "G2_RHO_FAMILY_PRACTICALLY_EQUIVALENT", "G3_OTHER_RHO_BETTER"},
        "historical_global_accumulation_proven": False,
    }
    write_json(RESULTS / "tables/final_verdict.json", payload)
    for key, value in payload.items():
        if key.isupper():
            print(f"{key:20s} {value}")


if __name__ == "__main__":
    main()
