"""Mechanical V3.4.0R verdict, including the external-window stop path."""
from __future__ import annotations

from _common import RESULTS, read_json, write_json
from patch_claim_classifier import overall_verdict


def maybe(name: str):
    path = RESULTS / "tables" / name
    return read_json(path) if path.exists() else None


def main() -> None:
    transfer = maybe("sensor_transfer.json") or {}
    window = maybe("external_window.json") or {}
    transport = transfer.get("transport_verdict", transfer.get("verdict", "ST2_FAIL"))
    if window.get("verdict") == "ST3_WINDOW_SHIFT":
        transport = "ST3_WINDOW_SHIFT"

    if transport in {"ST2_FAIL", "ST3_WINDOW_SHIFT"}:
        certificate = "NOT_RUN_EXTERNAL_TRANSPORT_GATE"
        budget = "NOT_RUN_EXTERNAL_TRANSPORT_GATE"
        controller = "CTRL4_INCONCLUSIVE"
        baseline = "BASE4_INCONCLUSIVE"
        rho = "RHO4_INCONCLUSIVE"
        utility = "NOT_RUN_EXTERNAL_TRANSPORT_GATE"
    else:
        certificate_row = maybe("certificate_validation.json") or {}
        budget_row = maybe("final_budget_audit.json") or {}
        controller_row = maybe("controller_effect.json") or {}
        baseline_row = maybe("linear_baseline.json") or {}
        rho_row = maybe("rho_family.json") or {}
        utility_row = maybe("utility.json") or {}
        certificate = certificate_row.get("verdict", "CERT2_IMPLEMENTATION_FAILURE")
        budget = budget_row.get("verdict", "BUD2_MISMATCH")
        controller = controller_row.get("controller_verdict", "CTRL4_INCONCLUSIVE")
        baseline = baseline_row.get("verdict", "BASE4_INCONCLUSIVE")
        rho = rho_row.get("rho_verdict", "RHO4_INCONCLUSIVE")
        utility = utility_row.get("verdict", "U2_FAIL")

    overall = overall_verdict(
        sensor_transport=transport,
        certificate=certificate,
        budget=budget,
        controller=controller,
        baseline=baseline,
        rho=rho,
        utility=utility,
    )
    payload = {
        "SENSOR_TRANSPORT": transport,
        "SENSOR_SCOPE": "SENS2_REFUSAL_SENSOR_ONLY",
        "ACTUATOR": "ACT1_FROZEN_REPLICATED",
        "CERTIFICATE": certificate,
        "BUDGET": budget,
        "CONTROLLER": controller,
        "BASELINE": baseline,
        "RHO": rho,
        "SEMANTIC": "SEM3_PROXY_ONLY",
        "UTILITY": utility,
        "OVERALL": overall,
        "structural_claim": "For fixed W, epsilon_h(rho)=2W*rho^2*(1-2rho) is uniquely maximised at rho=1/3.",
        "structural_claim_scope": "residual policy-transition certificate only",
        "empirical_behavioral_optimum_claimed": False,
        "semantic_safety_guarantee_claimed": False,
        "stop_reason": ("fixed-W coverage 0.8667 < 0.90" if transport == "ST3_WINDOW_SHIFT" else None),
        "invalid_q025_final_used": False
    }
    write_json(RESULTS / "tables/final_verdict.json", payload)
    for key, value in payload.items():
        if key.isupper():
            print(f"{key:<18} {value}")


if __name__ == "__main__":
    main()
