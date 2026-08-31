"""V3.3.3 AUTOMATIC CLAIM CLASSIFIER -- five separate verdicts, never merged."""
from __future__ import annotations
import sys, json, hashlib, pathlib, argparse
RHO_CANTOR = 1.0 / 3.0


def v_math(m):
    need = ["G_n_formula", "argmax_formula", "n3_is_cantor", "n_gt_3_counterexample"]
    miss = [k for k in need if not m.get(k, False)]
    return (("MATH_INVALID", f"failed {miss}") if miss else
            ("M1_CANTOR_UNIQUE_DEPTH3_GUARD_MAXIMISER",
             "G_n'(rho)=rho^(n-2)[(n-1)-2n rho] gives rho_max=(n-1)/(2n); n=3 "
             "yields exactly 1/3, and for n!=3 the maximiser is NOT 1/3"))


def v_behavior(b):
    if not b.get("identified", False):
        return ("G_BEHAVIORAL_BOUNDARY_UNIDENTIFIABLE",
                "tau_beh could not be validly estimated; tau_mid must NOT be "
                "substituted, and any claim needing behavioural-boundary "
                "uncertainty stops here")
    return ("M2_BEHAVIORAL_BOUNDARY_IDENTIFIED",
            f"tau_beh={b['tau_beh']:.4f} CI95 {b['tau_ci95']}; it differs from "
            f"the projection midpoint by {abs(b['gap_sigma']):.2f} sigma")


def v_system_b(s):
    if not s.get("adversarial", False):
        return ("SB_INVALID", "crossing distances are not adversarial/worst-case")
    r = s.get("max_ratio_deviation")
    if r is None:
        return ("SB_INCONCLUSIVE", "no crossing measurement")
    if r <= s.get("tolerance", 0.01):
        return ("SB_QUANTITATIVE_CONSISTENCY",
                f"measured minimum crossing distance matches G_n to within "
                f"{r:.2%}, inside the pre-declared {s['tolerance']:.0%} tolerance")
    if s.get("ordering_correct", False):
        return ("SB_QUALITATIVE_CONSISTENCY",
                "ordering matches but the magnitude does not meet tolerance")
    return ("SB_INCONSISTENT", "measured crossing contradicts the guard geometry")


def v_system_a(a):
    if not a.get("budget_matched", False):
        return ("F_INCONCLUSIVE", "actual intervention budgets not matched")
    if not a.get("endpoint_attainable", False):
        return ("F_INCONCLUSIVE", "endpoint degenerate on the frozen grid")
    ci = a.get("cantor_vs_alt_ci")
    if ci is None:
        return ("F_INCONCLUSIVE", "no admissible comparison")
    s = a.get("sesoi", 0.02)
    if ci[0] > 0:
        return ("A_CANTOR_GENERATION_POSITIVE",
                f"rho=1/3 exceeds the pre-specified alternative, CI {ci} excludes 0")
    if ci[1] < 0:
        return ("B_OTHER_RHO_GENERATION_POSITIVE",
                f"the pre-specified alternative exceeds rho=1/3, CI {ci}")
    if ci[0] > -s and ci[1] < s:
        return ("C_PRACTICALLY_EQUIVALENT",
                f"the whole CI {ci} lies inside the frozen SESOI +-{s}")
    return ("F_INCONCLUSIVE",
            f"CI {ci} neither excludes 0 nor fits inside +-{s}")


def overall(vm, vb, vsb, va):
    if vb[0].startswith("G_"):
        return ("MATH_STANDS_BEHAVIORAL_BRIDGE_UNRESOLVED",
                "the depth-3 theorem stands; the behavioural bridge is unresolved")
    if va[0] == "A_CANTOR_GENERATION_POSITIVE":
        return ("CONSISTENT_MATH_AND_GENERATION",
                "mathematical and generation evidence are mutually consistent; "
                "this is not a universal proof")
    if va[0] == "B_OTHER_RHO_GENERATION_POSITIVE":
        return ("MATH_OPTIMAL_BUT_GENERATION_PREFERS_OTHER_RHO",
                "1/3 is mathematically optimal for the depth-3 guard objective, "
                "but generation favours a different guard/refinement tradeoff")
    if va[0] == "C_PRACTICALLY_EQUIVALENT":
        return ("CONTRACTION_RATIO_EMPIRICALLY_WEAK",
                "the exact contraction ratio is empirically weak in the tested "
                "intervention regime despite the mathematical distinction")
    return ("INCONCLUSIVE_GENERATION", va[1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", default="results/v3_3_3/tables/gates.json")
    ap.add_argument("--out", default="results/v3_3_3/tables/verdict_v333.json")
    ap.add_argument("--allow-unsealed", action="store_true")
    a = ap.parse_args()
    me = pathlib.Path(__file__); my = hashlib.sha256(me.read_bytes()).hexdigest()
    seal = pathlib.Path("configs/v3_3_3/PRE_ANALYSIS_FREEZE.json")
    if seal.exists():
        s = json.loads(seal.read_text()).get("classifier_sha256")
        if s and s != my and not a.allow_unsealed:
            raise SystemExit(f"CLASSIFIER MODIFIED SINCE FREEZE\n sealed {s}\n actual {my}")
    G = json.loads(pathlib.Path(a.gates).read_text())
    vm, vb = v_math(G["math"]), v_behavior(G["behavior"])
    vsb, va = v_system_b(G["system_b"]), v_system_a(G["system_a"])
    ov = overall(vm, vb, vsb, va)
    out = {"MATHEMATICS": vm[0], "math_reason": vm[1],
           "BEHAVIORAL_BOUNDARY": vb[0], "behavior_reason": vb[1],
           "SYSTEM_B": vsb[0], "system_b_reason": vsb[1],
           "SYSTEM_A": va[0], "system_a_reason": va[1],
           "OVERALL": ov[0], "overall_reason": ov[1],
           "classifier_sha256": my}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2))
    REASON = {"MATHEMATICS": "math_reason",
              "BEHAVIORAL_BOUNDARY": "behavior_reason",
              "SYSTEM_B": "system_b_reason",
              "SYSTEM_A": "system_a_reason",
              "OVERALL": "overall_reason"}
    for k, rk in REASON.items():
        print(f"\n########  {k}: {out[k]}\n  {out[rk]}")
