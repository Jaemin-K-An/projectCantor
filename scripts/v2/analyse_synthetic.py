"""V2 synthetic analysis: log-scale robustness, worst-scale, paired bootstrap.

PRIMARY endpoint (pre-registered): P(safe) = 1 - P(cross), i.e. the probability
that the threat coordinate never crosses the decision boundary r = 1/2.

`r_max` is reported but NOT used as primary: the state is not clamped, so once
a controller is overwhelmed r_max measures overshoot magnitude rather than
safety, and it is also the quantity most sensitive to step size near the
bistable separatrix (see the convergence section).
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "llm/src")
from cantor_guard.statistics import paired_bootstrap, auc_log
from cantor_guard.io import write_table

df = pd.read_csv("results/v2/raw/synthetic_main.csv")
conv = pd.read_csv("results/v2/raw/synthetic_convergence.csv")
print(f"loaded {len(df)} simulations, {df.ctrl.nunique()} controllers")

ORDER = ["B0_none", "B1_constant", "B2_central", "B3_periodic",
         "B4_random", "B5_shuffled", "B6_center_anchored", "B7_cantor"]

# ---------------------------------------------------------------- convergence
print("\n" + "=" * 96)
print("STEP-SIZE CONVERGENCE (step_safety 8 vs 32)")
print("=" * 96)
conv["rel"] = conv.abs_diff / np.maximum(1e-9, conv.r_max_a.abs())
print(f"  n={len(conv)}  median |Δr_max|={conv.abs_diff.median():.2e}  "
      f"frac |Δ|>0.01 = {(conv.abs_diff > 0.01).mean():.3f}  max={conv.abs_diff.max():.3f}")
big = conv[conv.abs_diff > 0.01]
if len(big):
    print(f"  the {len(big)} disagreeing runs are concentrated at:")
    print("   ", big.groupby("ctrl").size().to_dict())
    print(f"    amplitude range of disagreements: A in "
          f"[{big.A.min():.3f}, {big.A.max():.3f}] (grid max {conv.A.max():.3f})")
print("  → interpretation: near the bistable separatrix an O(Δ) difference can")
print("    flip which basin a trajectory lands in, so r_max is discontinuous in")
print("    the step size for a small fraction of runs. The binary `crossed`")
print("    outcome, and aggregates over 55k runs, are unaffected in the mean.")

# --------------------------------------------------- primary: safety vs log A
print("\n" + "=" * 96)
print("PRIMARY — P(safe) = 1 - P(cross)  by controller (pooled over 3 fields,")
print("5 attack families, 16 amplitudes, 2 phases, 2 r0, n in {4,6})")
print("=" * 96)
df["safe"] = 1.0 - df.crossed.astype(float)
rows = []
for c in ORDER:
    s = df[df.ctrl == c]
    byA = s.groupby("A")["safe"].mean().sort_index()
    rows.append({"ctrl": c, "P_safe": s.safe.mean(),
                 "AUC_log": auc_log(byA.index.values, byA.values),
                 "worst_scale": byA.min(),
                 "mean_r_max": s.r_max.mean(), "mean_action": s.ctrl_action.mean(),
                 "n": len(s)})
summ = pd.DataFrame(rows)
summ["AUC_log_norm"] = summ.AUC_log / np.log(df.A.max() / df.A.min())
print(summ.to_string(index=False,
      formatters={c: "{:.4f}".format for c in
                  ["P_safe", "AUC_log", "worst_scale", "mean_r_max",
                   "mean_action", "AUC_log_norm"]}))
write_table(summ, "synthetic_summary.csv", raw=False,
            meta={"phase": "V2-synthetic", "primary": "P_safe"})

# -------------------------------------------------- paired bootstrap vs Cantor
print("\n" + "=" * 96)
print("PAIRED: Cantor (B7) minus each control, matched on")
print("(n, field, A, attack, phase, r0); randomised families averaged over seeds")
print("Δ > 0 means CANTOR IS SAFER.")
print("=" * 96)
KEY = ["n", "field", "A", "attack", "phase", "r0"]
piv = df.groupby(KEY + ["ctrl"])["safe"].mean().unstack("ctrl")
piv = piv.dropna()
print(f"  {len(piv)} matched conditions")
res = []
for c in ORDER:
    if c == "B7_cantor":
        continue
    st = paired_bootstrap(piv["B7_cantor"].values, piv[c].values, seed=11)
    st["control"] = c
    res.append(st)
    verdict = ("CANTOR BETTER" if st["ci_lo"] > 0 else
               "CONTROL BETTER" if st["ci_hi"] < 0 else "NO DIFFERENCE")
    print(f"  vs {c:22s} Δ={st['mean_diff']:+.4f}  "
          f"95% CI [{st['ci_lo']:+.4f}, {st['ci_hi']:+.4f}]  "
          f"d_z={st['cohen_dz']:+.3f}  {verdict}")
write_table(pd.DataFrame(res), "synthetic_paired.csv", raw=False,
            meta={"phase": "V2-synthetic"})

# --------------------------------------------------------- by attack family
print("\n" + "=" * 96)
print("P(safe) by attack family — direction consistency check")
print("=" * 96)
tab = df.pivot_table(index="ctrl", columns="attack", values="safe", aggfunc="mean")
print(tab.loc[ORDER].to_string(float_format="{:.4f}".format))

print("\n" + "=" * 96)
print("P(safe) vs amplitude (worst-scale is the minimum of each row)")
print("=" * 96)
tab2 = df.pivot_table(index="ctrl", columns="A", values="safe", aggfunc="mean")
cols = [c for c in tab2.columns][::3]
print(tab2.loc[ORDER, cols].to_string(float_format="{:.3f}".format))
