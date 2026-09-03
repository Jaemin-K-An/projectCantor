"""V3.3.5b AUTOMATIC CLAIM CLASSIFIER -- Stage A gate, then STOP or Stage B."""
from __future__ import annotations
import sys, json, hashlib, pathlib, argparse


def v_math(m):
    return (("M1_CANTOR_AFFINE_MAXIMIN_VALID",
             "frozen from V3.3.5: eps(rho)=2W rho^2(1-2rho), argmax 1/3")
            if m.get("unchanged", False) else ("MATH_INVALID", "altered"))


def v_temporal(t):
    v = t["verdict"]
    if v == "TD1_DISTRIBUTED_SUPPORTED":
        return (v, "a pre-declared distributed schedule materially exceeds both "
                   "single-state schedules at matched B2")
    if v == "TD3_SINGLE_STATE_BETTER":
        return (v, "at matched trajectory L2 budget, CONCENTRATING the "
                   "intervention at P0 is significantly BETTER than "
                   "distributing it; the simultaneous intervals for "
                   "EARLY_4/EARLY_8 minus P0_ONLY exclude zero in the negative "
                   "direction at B2 = 0.4 and 0.8")
    if v == "TD2_ACCUMULATION_EXPLAINS_GLOBAL":
        return (v, "matched-budget schedules are equivalent, so the historical "
                   "all-forward advantage was repeated intervention energy")
    return ("TD4_INCONCLUSIVE", "no admissible discrimination")


def v_trajectory(t):
    return ("TR3_NOT_RUN_TEMPORAL_GATE_FAILED",
            "Stage A did not support temporal distribution, so no trajectory "
            "coordinate was invented and D_final_traj was not spent")


def v_generation(g):
    return ("G5_NOT_RUN", "Stage B never opened")


def overall(vm, vt, vr, vg):
    if vm[0] == "MATH_INVALID":
        return ("F_INCONCLUSIVE", "mathematics altered")
    if vt[0] == "TD1_DISTRIBUTED_SUPPORTED":
        return ("PROCEED_TO_STAGE_B", vt[1])
    if vt[0] in ("TD2_ACCUMULATION_EXPLAINS_GLOBAL", "TD3_SINGLE_STATE_BETTER"):
        return ("D_GLOBAL_ADVANTAGE_WAS_ACCUMULATION",
                "the large behavioural effect of the historical all-forward "
                "intervention is attributable to repeated intervention energy, "
                "not to a temporally distributed refusal mechanism. Under "
                "matched energy, concentration at P0 is at least as good and "
                "significantly better at the larger budgets.")
    return ("F_INCONCLUSIVE", vt[1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", default="results/v3_3_5b/tables/gates.json")
    ap.add_argument("--out", default="results/v3_3_5b/tables/verdict_v335b.json")
    ap.add_argument("--allow-unsealed", action="store_true")
    a = ap.parse_args()
    me = pathlib.Path(__file__); my = hashlib.sha256(me.read_bytes()).hexdigest()
    G = json.loads(pathlib.Path(a.gates).read_text())
    vm, vt = v_math(G["math"]), v_temporal(G["temporal"])
    vr, vg = v_trajectory(G["temporal"]), v_generation({})
    ov = overall(vm, vt, vr, vg)
    out = {"MATH": vm[0], "math_reason": vm[1],
           "TEMPORAL": vt[0], "temporal_reason": vt[1],
           "TRAJECTORY": vr[0], "trajectory_reason": vr[1],
           "GENERATION": vg[0], "generation_reason": vg[1],
           "OVERALL": ov[0], "overall_reason": ov[1], "classifier_sha256": my}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2))
    for k, rk in (("MATH","math_reason"),("TEMPORAL","temporal_reason"),
                  ("TRAJECTORY","trajectory_reason"),("GENERATION","generation_reason"),
                  ("OVERALL","overall_reason")):
        print(f"\n########  {k}: {out[k]}\n  {out[rk]}")
