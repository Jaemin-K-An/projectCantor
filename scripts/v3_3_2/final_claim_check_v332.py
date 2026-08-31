"""V3.3.2 AUTOMATIC CLAIM CLASSIFIER.

Three verdicts, reported separately and never merged:

  MATH      always reported; it does not depend on any LLM
  EMPIRICAL what the independently measured absolute uncertainty implies
  MECHANISM whether the old null series is explained -- NOT inherited from
            V3.3.1, which built its claim on an invalid estimator
"""
from __future__ import annotations
import sys, json, hashlib, pathlib, argparse

RHO_CANTOR = 1.0 / 3.0
NEAR = 0.03                      # frozen Cantor neighbourhood, +-0.03
SESOI = 0.03


def classify_math(theory: dict) -> tuple[str, str]:
    need = ["G_ok", "BGR_ok", "P_ok", "R_exact", "AG_ok", "AU_ok"]
    miss = [k for k in need if not theory.get(k, False)]
    if miss:
        return "MATH_INVALID", f"failed: {miss}"
    return ("A_MATH_CANTOR_BALANCED_OPTIMUM",
            "rho*(kappa)=1/(2+kappa) with kappa=1 giving 1/3 uniquely; "
            "absolute-guard theorems AG/AU verified, including the n>3 "
            "counterexample where Cantor does NOT maximise G_n")


def classify_empirical(bridge: dict) -> tuple[str, str, dict]:
    d = {}
    if not bridge.get("phase_hook_verified", False):
        return ("F_BRIDGE_INVALID",
                "phase-aware hook not verified: prefill and decode must use "
                "their own calibrations", d)
    if not bridge.get("two_class_calibration", False):
        return ("F_BRIDGE_INVALID",
                "generation calibration is not a two-class midpoint", d)
    if not bridge.get("delta_rho_independent", False):
        return ("F_BRIDGE_INVALID",
                "delta_abs is not estimated independently of rho", d)
    if not bridge.get("final_untouched", False):
        return ("E_EMPIRICAL_INCONCLUSIVE", "no untouched final set", d)

    ci = bridge.get("rho_pred_ci95")
    med = bridge.get("rho_pred_median")
    d["rho_pred_ci95"], d["rho_pred_median"] = ci, med
    flat = bridge.get("surface_flat", None)
    d["surface_flat"] = flat
    d["cantor_on_pareto_front"] = bridge.get("cantor_on_pareto_front")

    if ci is None or med is None:
        return ("E_EMPIRICAL_INCONCLUSIVE",
                "no admissible predicted ratio (uncertainty exceeded the widest "
                "guard at every tested depth)", d)
    inside = (ci[0] >= RHO_CANTOR - NEAR) and (ci[1] <= RHO_CANTOR + NEAR)
    d["ci_inside_cantor_band"] = inside
    if inside and flat is False:
        return ("B_ABSOLUTE_UNCERTAINTY_SELECTS_CANTOR",
                f"predicted ratio CI {ci} lies inside "
                f"[{RHO_CANTOR-NEAR:.3f}, {RHO_CANTOR+NEAR:.3f}] and the guard "
                "surface is not flat", d)
    if inside and flat is not False:
        return ("D_EMPIRICAL_FLAT",
                "predicted ratio is compatible with 1/3 but the guard surface "
                "is flat, so no ratio is distinguishable", d)
    if flat is True:
        return ("D_EMPIRICAL_FLAT", "guard surface flat across rho", d)
    return ("C_ABSOLUTE_UNCERTAINTY_SELECTS_OTHER_RHO",
            f"the theory predicts the measured boundary, but the measured "
            f"uncertainty implies rho ~ {med:.3f} (95% CI {ci}), outside the "
            f"pre-registered Cantor band; this model does not sit in the "
            f"balanced kappa=1 regime", d)


def classify_mechanism(mech: dict) -> tuple[str, str]:
    if mech.get("interaction_significant", False):
        return ("M_SUPPORTED",
                "phase miscalibration and excessive depth measurably attenuated "
                "layout differentiation")
    if mech.get("fine_guards_below_uncertainty", False):
        return ("M_PLAUSIBLE",
                "fine-scale guards are narrower than the measured calibration "
                "uncertainty, which is a possible explanation of the earlier "
                "null results, but no interaction evidence establishes it as "
                "the cause")
    return ("M_NOT_SUPPORTED", "prediction and empirical pattern do not agree")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--theory", default="results/v3_3_2/tables/theory_gate.json")
    ap.add_argument("--bridge", default="results/v3_3_2/tables/bridge_gate.json")
    ap.add_argument("--mechanism", default="results/v3_3_2/tables/mechanism_gate.json")
    ap.add_argument("--out", default="results/v3_3_2/tables/verdict_v332.json")
    ap.add_argument("--allow-unsealed", action="store_true")
    a = ap.parse_args()
    me = pathlib.Path(__file__)
    my_sha = hashlib.sha256(me.read_bytes()).hexdigest()
    seal = pathlib.Path("configs/v3_3_2/PRE_ANALYSIS_FREEZE.json")
    if seal.exists():
        s = json.loads(seal.read_text()).get("classifier_sha256")
        if s and s != my_sha and not a.allow_unsealed:
            raise SystemExit(f"CLASSIFIER MODIFIED SINCE FREEZE\n sealed {s}\n actual {my_sha}")
    T = json.loads(pathlib.Path(a.theory).read_text())
    B = json.loads(pathlib.Path(a.bridge).read_text())
    M = json.loads(pathlib.Path(a.mechanism).read_text())
    vm, wm = classify_math(T)
    ve, we, de = classify_empirical(B)
    vk, wk = classify_mechanism(M)
    out = {"MATH_VERDICT": vm, "math_reason": wm,
           "EMPIRICAL_VERDICT": ve, "empirical_reason": we, "empirical_detail": de,
           "MECHANISM_VERDICT": vk, "mechanism_reason": wk,
           "classifier_sha256": my_sha, "cantor_band": [RHO_CANTOR - NEAR, RHO_CANTOR + NEAR]}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n########  MATH:      {vm}\n  {wm}")
    print(f"\n########  EMPIRICAL: {ve}\n  {we}")
    print(f"\n########  MECHANISM: {vk}\n  {wk}")
