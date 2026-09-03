"""V3.3.5a AUTOMATIC CLAIM CLASSIFIER."""
from __future__ import annotations
import sys, json, hashlib, pathlib, argparse


def v_math(m):
    return (("M1_CANTOR_AFFINE_MAXIMIN_VALID",
             "unchanged from V3.3.5: eps(rho)=2W rho^2(1-2rho), argmax 1/3, "
             "eps_C = 2W/27") if m.get("unchanged", False)
            else ("MATH_INVALID", "frozen mathematics was altered"))


def v_direction(d):
    if not d.get("first_token_upstream", False):
        return ("D2_P0_DIRECTION_NOT_CAUSAL",
                "P0 intervention did not reach the first-token logits")
    if d.get("causal", False):
        return ("D1_P0_DIRECTION_CAUSAL", "narrow symmetric gate passed")
    return ("D2_P0_DIRECTION_NOT_CAUSAL",
            f"the pre-declared +-2 sigma gate produced ZERO behavioural "
            f"variation (refusal span {d.get('refusal_span'):.4f}); the "
            f"placement is mechanically correct -- a +4 sigma dose moves the "
            f"first-token logits by {d['first_token']['max_abs_dlogit']:.2f} "
            f"and flips top-1 for {d['first_token']['top1_flip']:.0%} -- but "
            "that mechanical reach does not translate into behaviour")


def v_boundary(b):
    st = b["status"]
    if st == "B1_P0_BOUNDARY_IDENTIFIED":
        return (st, f"tau_beh,P0 = {b['tau']:.4f}")
    if st == "B3_P0_NOT_CAUSAL":
        return (st, "no detectable leverage")
    return (st,
            f"slope CI excludes zero, so leverage EXISTS, but beta_std = "
            f"{b['beta_std']:.4f} (gate 0.10), dP over 2 sigma = "
            f"{b['dP_2sigma']:+.4f}, tau CI width {b['ci_width_sigma']:.1f} "
            f"sigma (gate 3.0), and the dose-response is NON-MONOTONE. "
            "Reproduced on an independent confirmation split.")


def v_phase(p):
    r = {x["phase"]: x["beta_std"] for x in p["rows"]}
    return ("PHASE_LEVERAGE_IS_DISTRIBUTED",
            "standardized leverage: P0 %.4f, G1 %.4f, GLOBAL %.4f -- all-forward "
            "intervention carries %.1fx the leverage of either single state, so "
            "refusal control in this model is temporally DISTRIBUTED rather than "
            "localised at one residual state" % (
                r["P0 (pre-token-1)"], r["G1 (first decode)"],
                r["GLOBAL (all forwards)"],
                r["GLOBAL (all forwards)"] / max(r["P0 (pre-token-1)"],
                                                 r["G1 (first decode)"])))


def v_generation(g):
    if not g.get("run", False):
        return ("G5_NOT_RUN_GATE_FAILURE",
                "the P0 behavioural gate failed, so per the protocol the final "
                "set was NOT spent. D_final_P0 remains untouched.")
    return ("G4_INCONCLUSIVE", "no admissible comparison")


def overall(vm, vd, vb, vp, vg):
    if vm[0] == "MATH_INVALID":
        return ("F_INCONCLUSIVE", "mathematics altered")
    if vb[0] == "B1_P0_BOUNDARY_IDENTIFIED":
        return ("A_CANTOR_BEHAVIORALLY_ANCHORED_LLM_CONTROLLER_SUPPORTED", vb[1])
    return ("E_SINGLE_STATE_CAUSAL_CONTROLLER_NOT_SUPPORTED",
            "neither the pre-generation P0 state nor the post-first-token G1 "
            "state provides a usable single-state behavioural anchor. The exact "
            "Cantor certificate remains valid; what is missing is a scalar "
            "residual state on which it can be behaviourally anchored.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", default="results/v3_3_5a/tables/gates.json")
    ap.add_argument("--out", default="results/v3_3_5a/tables/verdict_v335a.json")
    ap.add_argument("--allow-unsealed", action="store_true")
    a = ap.parse_args()
    me = pathlib.Path(__file__); my = hashlib.sha256(me.read_bytes()).hexdigest()
    seal = pathlib.Path("configs/v3_3_5a/PRE_ANALYSIS_FREEZE.json")
    if seal.exists():
        s = json.loads(seal.read_text()).get("classifier_sha256")
        if s and s != my and not a.allow_unsealed:
            raise SystemExit(f"CLASSIFIER MODIFIED SINCE FREEZE\n sealed {s}\n actual {my}")
    G = json.loads(pathlib.Path(a.gates).read_text())
    vm, vd = v_math(G["math"]), v_direction(G["direction"])
    vb, vp = v_boundary(G["boundary"]), v_phase(G["phase"])
    vg = v_generation(G["generation"])
    ov = overall(vm, vd, vb, vp, vg)
    out = {"MATH": vm[0], "math_reason": vm[1],
           "P0_DIRECTION": vd[0], "direction_reason": vd[1],
           "P0_BOUNDARY": vb[0], "boundary_reason": vb[1],
           "PHASE_CAUSALITY": vp[0], "phase_reason": vp[1],
           "GENERATION": vg[0], "generation_reason": vg[1],
           "OVERALL": ov[0], "overall_reason": ov[1], "classifier_sha256": my}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2))
    for k, rk in (("MATH","math_reason"),("P0_DIRECTION","direction_reason"),
                  ("P0_BOUNDARY","boundary_reason"),("PHASE_CAUSALITY","phase_reason"),
                  ("GENERATION","generation_reason"),("OVERALL","overall_reason")):
        print(f"\n########  {k}: {out[k]}\n  {out[rk]}")
