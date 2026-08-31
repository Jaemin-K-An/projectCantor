"""V3.3.1 PHASE 16-18 -- goal-clustered analysis of the rho sweep.

Inference unit is the GOAL, as established in V3.2: the 50 held-out goals are
what we generalise to, not the (attack, Delta, eps) cells.
"""
import sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
from cantor_guard_v32.cluster_stats import cluster_bootstrap_by_goal, tost_equivalence
from cantor_guard_v331.guard_geometry import RHO_CANTOR

SESOI = 0.03
TAB = pathlib.Path("results/v3_3_1/tables")
CAL = json.loads((TAB / "phase_calibration_qwen2.5-0.5b-instruct.json").read_text())
df = pd.read_csv("results/v3_3_1/raw/v331_llm_rho_qwen2.5-0.5b-instruct.csv")
ut = pd.read_csv("results/v3_3_1/raw/v331_llm_utility_qwen2.5-0.5b-instruct.csv")
KEYS = ["attack", "delta", "eps", "pid"]
n_max = CAL["max_useful_depth"]["q50"]["n_max_over_all_rho"]
print(f"measured max useful depth (median uncertainty) = {n_max}\n")

rows = []
for depth, g in df.groupby("depth"):
    piv = g.pivot_table(index=KEYS, columns="rho", values="safe")
    rhos = sorted(piv.columns)
    ref = min(rhos, key=lambda r: abs(r - RHO_CANTOR))
    print(f"=== depth {depth} ({'FEASIBLE' if depth <= n_max else 'BELOW NOISE FLOOR'}) ===")
    for r in rhos:
        if r == ref:
            continue
        m = piv[[ref, r]].dropna().reset_index().rename(
            columns={ref: "score_a", r: "score_b"})
        st = cluster_bootstrap_by_goal(m, "score_a", "score_b", n_boot=20000, seed=7)
        eq = tost_equivalence(st, SESOI)["equivalent"]
        sig = st["ci_lo"] > 0 or st["ci_hi"] < 0
        print(f"  cantor vs rho={r:.4f}: d={st['mean_diff']:+.4f} "
              f"[{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}] eq={eq} sig={sig}")
        rows.append({"depth": depth, "rho_other": r, "mean_diff": st["mean_diff"],
                     "ci_lo": st["ci_lo"], "ci_hi": st["ci_hi"],
                     "equivalent": eq, "significant": sig,
                     "feasible_depth": depth <= n_max})
    fam = g.groupby("rho").safe.mean()
    print(f"  spread across rho = {fam.max()-fam.min():.4f}"
          f"   argmax rho = {fam.idxmax():.4f}")
    rows[-1]["spread"] = float(fam.max() - fam.min())

out = pd.DataFrame(rows); out.to_csv(TAB / "rho_cluster_comparisons.csv", index=False)

print("\n=== utility (benign prompts) ===")
print(ut.groupby(["depth", "rho"])[["false_refusal", "coherence", "mean_words"]]
      .mean().round(4).to_string())

# empirical optimum, per depth, on the safety axis with utility as constraint
emp = {}
for depth, g in df.groupby("depth"):
    fam = g.groupby("rho").safe.mean()
    emp[int(depth)] = {"argmax_rho": float(fam.idxmax()),
                       "spread": float(fam.max() - fam.min()),
                       "any_significant": bool(
                           out[(out.depth == depth)].significant.any())}
feasible = [d for d in emp if d <= n_max]
gate = {"available": True,
        "depth_tested": int(min(emp)) if feasible else int(min(emp)),
        "max_useful_depth_best_rho": int(n_max),
        "rho_empirical_optimum": (emp[min(feasible)]["argmax_rho"]
                                  if feasible else None),
        "per_depth": emp,
        "note": ("rho_empirical_optimum is taken from the deepest FEASIBLE "
                 "depth; depths above the noise floor cannot discriminate rho")}
(TAB / "empirical_gate.json").write_text(json.dumps(gate, indent=2))
print("\nempirical gate:", json.dumps(gate, indent=2))
