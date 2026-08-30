"""V3.2 POST-HOC sensitivity -- NOT part of the sealed pre-analysis.

Labelled post-hoc, run after the sealed verdict, and it does NOT replace it.

The gains were fitted on D_budget and hit the target there (13/13 within
+-1.8%). On D_test the realised C_rms drifted for three families:

    T2_global_smooth  0.0167 (-16%)
    T3_wide_central   0.0145 (-27%)
    T8_minimax        0.0192  (-4%)

That is a genuine generalisation finding and it is reported as such. But it
also means those particular comparisons are no longer budget-matched on the
data the comparison is read from: T3 spent a quarter less intervention than
Cantor did. The pre-analysis plan's exclusion rule keys on the D_budget fit, so
the sealed classifier still used them.

This script re-runs the comparison keeping only controls whose realised C_rms
on D_TEST is within +-3% of Cantor's own realised C_rms, and reports whether
the verdict would change. The sealed verdict stands either way; this says how
sensitive it is.
"""
import sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
from cantor_guard_v32.cluster_stats import cluster_bootstrap_by_goal, tost_equivalence

SESOI, CANTOR = 0.03, "T7_cantor"
KEYS = ["attack", "delta", "eps", "pid"]
MATCHED = ["T5_shuffled", "T6_center_anchored", "T4_periodic", "T3_wide_central"]

rows = []
for model in ("qwen2.5-0.5b-instruct", "olmo2-1b-instruct"):
    src = pathlib.Path(f"results/v3_2/raw/v32_final_{model}.csv")
    if not src.exists():
        continue
    df = pd.read_csv(src)
    scores = [c for c in ("safe_lex32", "safe_ext") if c in df.columns]
    cfg = json.loads(pathlib.Path(f"configs/v3_2/frozen_{model}.json").read_text())
    tgt = cfg["target_C_rms"]
    real = df.groupby("family").C_rms.mean()
    c_ref = real[CANTOR]
    # Anchor on the PRE-REGISTERED TARGET, not on Cantor's own realised value.
    # Anchoring on Cantor is wrong when Cantor is itself the drifting arm: on
    # Model B every control sits at the target and Cantor is +8%, so a
    # Cantor-anchored filter would discard all four controls and answer nothing.
    keep = {f for f, v in real.items()
            if f == "T0_none" or abs(v - tgt) / tgt <= 0.03}
    dropped = {f: round(float(v), 5) for f, v in real.items() if f not in keep}
    print(f"\n=== {model} ===")
    print(f"target C_rms {tgt}; Cantor realised {c_ref:.5f} "
          f"({100*(c_ref-tgt)/tgt:+.1f}%)")
    print(f"kept (within +-3% of TARGET): {sorted(keep - {'T0_none'})}")
    print(f"DROPPED for D_test budget drift: {dropped}")
    if CANTOR in dropped:
        direction = "MORE" if c_ref > tgt else "LESS"
        print(f"  NOTE: Cantor itself drifted ({100*(c_ref-tgt)/tgt:+.1f}%). It "
              f"spent {direction} realised budget than the matched controls, so "
              f"any Cantor advantage here would be bought, not free.")

    for sc in scores:
        g = df.groupby(["family"] + KEYS, as_index=False)[sc].mean()
        piv = g.pivot_table(index=KEYS, columns="family", values=sc)
        eq_all, used = [], []
        for alt in MATCHED:
            if alt not in keep or alt not in piv or CANTOR not in piv:
                continue
            m = piv[[CANTOR, alt]].dropna().reset_index().rename(
                columns={CANTOR: "score_a", alt: "score_b"})
            st = cluster_bootstrap_by_goal(m, "score_a", "score_b", n_boot=20000, seed=7)
            eq = tost_equivalence(st, SESOI)["equivalent"]
            eq_all.append(eq); used.append(alt)
            print(f"  {sc:11s} {alt:20s} d={st['mean_diff']:+.4f} "
                  f"[{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}] eq={eq}")
            rows.append({"model": model, "scorer": sc, "control": alt,
                         "mean_diff": st["mean_diff"], "ci_lo": st["ci_lo"],
                         "ci_hi": st["ci_hi"], "equivalent": eq,
                         "posthoc_budget_filtered": True})
        v = ("C_PRACTICALLY_EQUIVALENT" if eq_all and all(eq_all)
             else "NOT_ALL_EQUIVALENT")
        print(f"  -> {sc}: {v}  (controls used: {used})")

pd.DataFrame(rows).to_csv(
    "results/v3_2/tables/posthoc_budget_sensitivity.csv", index=False)
print("\nPOST-HOC. Does not replace the sealed verdict.")
