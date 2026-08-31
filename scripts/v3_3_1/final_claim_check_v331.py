"""V3.3.1 AUTOMATIC CLAIM CLASSIFIER -- frozen before the LLM sweep.

Two verdicts are emitted and NEVER merged (harness section 44):

  LEVEL 1 MATHEMATICAL. Does the guard-resolution theory hold? This is decided
  by proofs and their symbolic/numerical verification, not by any LLM.

  LEVEL 2 EMPIRICAL. Does the LLM operating point put rho* near 1/3, and does
  1/3 sit near the measured Pareto knee?

Verdicts:
  A_CANTOR_BALANCED_OPTIMUM_THEORETICAL
  B_CANTOR_BALANCED_OPTIMUM_EMPIRICALLY_SUPPORTED
  C_GENERAL_GUARD_OPTIMUM_NOT_CANTOR
  D_THEORY_VALID_EMPIRICAL_INCONCLUSIVE
  E_THEORY_INVALID
"""
from __future__ import annotations
import sys, json, hashlib, pathlib, argparse

NEAR_THIRD_ABS = 0.03          # frozen tolerance (harness section 51)
RHO_CANTOR = 1.0 / 3.0


def classify(theory: dict, emp: dict | None) -> tuple[str, str, dict]:
    d = {}
    # ---- LEVEL 1: the mathematics ----
    checks = {
        "G_rho_star": theory.get("G_ok", False),
        "BGR_unique_max": theory.get("BGR_ok", False),
        "P_monotonicity": theory.get("P_ok", False),
        "R_exact": theory.get("R_exact", False),
        "counterexamples_present": theory.get("counterexamples", False),
    }
    d["level1_checks"] = checks
    if not all(checks.values()):
        return ("E_THEORY_INVALID",
                f"a mathematical check failed: "
                f"{[k for k, v in checks.items() if not v]}", d)

    if emp is None or not emp.get("available", False):
        return ("A_CANTOR_BALANCED_OPTIMUM_THEORETICAL",
                "theorems G/BGR/P/R proved and independently verified; no "
                "admissible LLM measurement was available, so no empirical "
                "claim is made", d)

    # ---- LEVEL 2: the LLM operating point ----
    # The theory's prediction from MEASURED calibration uncertainty.
    n_max = emp.get("max_useful_depth_best_rho")
    depth = emp.get("depth_tested")
    d["max_useful_depth"] = n_max
    d["depth_tested"] = depth
    if n_max is not None and depth is not None and depth > n_max:
        return ("D_THEORY_VALID_EMPIRICAL_INCONCLUSIVE",
                f"the calibration uncertainty admits a useful depth of only "
                f"n<={n_max}, while the controller under test runs n={depth}; "
                "levels beyond the noise floor cannot discriminate any rho, so "
                "this measurement cannot test the guard optimum", d)

    rho_emp = emp.get("rho_empirical_optimum")
    if rho_emp is None:
        return ("D_THEORY_VALID_EMPIRICAL_INCONCLUSIVE",
                "no empirical optimum could be located", d)
    d["rho_empirical_optimum"] = rho_emp
    d["distance_from_one_third"] = abs(rho_emp - RHO_CANTOR)
    if abs(rho_emp - RHO_CANTOR) <= NEAR_THIRD_ABS:
        return ("B_CANTOR_BALANCED_OPTIMUM_EMPIRICALLY_SUPPORTED",
                f"the empirical optimum rho={rho_emp:.4f} lies within "
                f"{NEAR_THIRD_ABS} of 1/3, so the measured operating point is "
                "compatible with the balanced guard requirement kappa=1", d)
    return ("C_GENERAL_GUARD_OPTIMUM_NOT_CANTOR",
            f"the guard-resolution theory holds, but the empirical optimum "
            f"rho={rho_emp:.4f} is {abs(rho_emp-RHO_CANTOR):.4f} from 1/3, so "
            "this LLM regime does not weight guard and refinement equally", d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--theory", default="results/v3_3_1/tables/theory_gate.json")
    ap.add_argument("--empirical", default="results/v3_3_1/tables/empirical_gate.json")
    ap.add_argument("--out", default="results/v3_3_1/tables/verdict_v331.json")
    ap.add_argument("--allow-unsealed", action="store_true")
    a = ap.parse_args()

    me = pathlib.Path(__file__)
    my_sha = hashlib.sha256(me.read_bytes()).hexdigest()
    seal = pathlib.Path("configs/v3_3_1/PRE_ANALYSIS_FREEZE.json")
    if seal.exists():
        sealed = json.loads(seal.read_text()).get("classifier_sha256")
        if sealed and sealed != my_sha and not a.allow_unsealed:
            raise SystemExit(f"CLASSIFIER MODIFIED SINCE FREEZE\n  sealed {sealed}"
                             f"\n  actual {my_sha}")
    theory = json.loads(pathlib.Path(a.theory).read_text())
    ep = pathlib.Path(a.empirical)
    emp = json.loads(ep.read_text()) if ep.exists() else None
    v, why, detail = classify(theory, emp)
    out = {"verdict": v, "reason": why, "classifier_sha256": my_sha,
           "near_third_tolerance": NEAR_THIRD_ABS, "detail": detail}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n########  V3.3.1 VERDICT: {v}  ########\n{why}\n")
    print(json.dumps(detail, indent=2)[:1800])
