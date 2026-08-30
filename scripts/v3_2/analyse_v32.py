"""V3.2 PHASE 13/14 -- decompose the effect into its three nested questions.

The programme's question is often stated as one thing but is really three,
nested, and only the last is about Cantor:

  Q_A  does intervening at all help?          T7_cantor  vs  T0_none
  Q_B  does STATE-DEPENDENCE help?            T7_cantor  vs  T1_true_constant
  Q_C  does MULTISCALE help beyond one scale? T7_cantor  vs  T2_global_smooth
  Q_D  does the RECURSIVE ORDERING help?      T7_cantor  vs  width/energy-matched

Q_D is the only Cantor-specific question. Reporting the four separately stops a
positive answer to Q_A being read as support for Cantor, which is the confusion
V1 and V2 both fell into.
"""
import argparse, sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
from cantor_guard_v32.cluster_stats import (cluster_bootstrap_by_goal,
                                            naive_cell_bootstrap,
                                            hierarchical_bootstrap,
                                            tost_equivalence)

SESOI = 0.03
KEYS = ["attack", "delta", "eps", "pid"]
CANTOR = "T7_cantor"
QUESTIONS = [
    ("Q_A intervene at all",      "T0_none",           False),
    ("Q_B state-dependence",      "T1_true_constant",  False),
    ("Q_C multiscale vs 1 scale", "T2_global_smooth",  False),
    ("Q_D ordering (shuffled)",   "T5_shuffled",       True),
    ("Q_D ordering (centered)",   "T6_center_anchored", True),
    ("Q_D ordering (periodic)",   "T4_periodic",       True),
    ("Q_D ordering (wide)",       "T3_wide_central",   True),
    ("Q_D ordering (minimax)",    "T8_minimax",        True),
]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
ap.add_argument("--boot", type=int, default=20000)
a = ap.parse_args()

src = pathlib.Path(f"results/v3_2/raw/v32_final_{a.model}.csv")
df = pd.read_csv(src)
scores = [c for c in ("safe_lex32", "safe_ext") if c in df.columns]
print(f"{a.model}: {len(df)} rows, {df.pid.nunique()} goals, "
      f"{df.family.nunique()} families, scorers={scores}\n")

# realised budget on D_test -- a GENERALISATION result, never retuned
cfg = json.loads(pathlib.Path(f"configs/v3_2/frozen_{a.model}.json").read_text())
bud = df.groupby("family").C_rms.mean()
tgt = cfg["target_C_rms"]
print("realised C_rms on D_test (fitted on D_budget, NOT retuned here):")
for f, v in bud.sort_values().items():
    rel = 100 * (v - tgt) / tgt
    print(f"  {f:22s} {v:.5f} ({rel:+5.1f}%) {'ok' if abs(rel)<=3 else 'DRIFT'}")
drift = {f: float(v) for f, v in bud.items()
         if abs(v - tgt) / tgt > 0.03 and f != "T0_none"}

# Families that missed the budget on D_budget are excluded from the matched
# comparisons (PRE_ANALYSIS_PLAN section 7) but still reported.
UNMATCHED = {k.split("|")[0] for k, g in cfg["gains"].items() if not g["matched"]}
if UNMATCHED:
    print(f"\nEXCLUDED from matched comparisons (budget mismatch on D_budget): "
          f"{sorted(UNMATCHED)}")

rows = []
for sc in scores:
    g = df.groupby(["family"] + KEYS, as_index=False)[sc].mean()
    piv = g.pivot_table(index=KEYS, columns="family", values=sc)
    print(f"\n=== {sc} ===")
    print(f"{'question':28s} {'d':>8s} {'cluster 95% CI':>22s} "
          f"{'naive':>20s} {'eq':>4s} {'pow':>5s}")
    for label, alt, matched in QUESTIONS:
        if CANTOR not in piv or alt not in piv:
            continue
        m = piv[[CANTOR, alt]].dropna().reset_index().rename(
            columns={CANTOR: "score_a", alt: "score_b"})
        if m.pid.nunique() < 3:
            continue
        cl = cluster_bootstrap_by_goal(m, "score_a", "score_b",
                                       n_boot=a.boot, seed=7)
        nv = naive_cell_bootstrap(m, "score_a", "score_b", n_boot=a.boot, seed=7)
        hi = hierarchical_bootstrap(m, "score_a", "score_b",
                                    n_boot=max(2000, a.boot // 5), seed=7)
        eq = tost_equivalence(cl, SESOI)["equivalent"]
        from math import sqrt, erf
        sd, n = cl["between_goal_sd"], cl["n_goals"]
        pw = (0.5 * (1 + erf((SESOI * sqrt(n) / sd - 1.959964) / sqrt(2)))
              if sd > 0 else np.nan)
        print(f"{label:28s} {cl['mean_diff']:+8.4f} "
              f"[{cl['ci_lo']:+.4f},{cl['ci_hi']:+.4f}] "
              f"[{nv['ci_lo']:+.4f},{nv['ci_hi']:+.4f}] "
              f"{'Y' if eq else 'n':>4s} {pw:5.2f}")
        rows.append({"model": a.model, "scorer": sc, "question": label,
                     "control": alt, "matched": matched,
                     "budget_excluded": alt in UNMATCHED,
                     "mean_diff": cl["mean_diff"], "ci_lo": cl["ci_lo"],
                     "ci_hi": cl["ci_hi"], "naive_lo": nv["ci_lo"],
                     "naive_hi": nv["ci_hi"], "hier_lo": hi["ci_lo"],
                     "hier_hi": hi["ci_hi"], "equivalent": eq,
                     "significant": bool(cl["ci_lo"] > 0 or cl["ci_hi"] < 0),
                     "between_goal_sd": cl["between_goal_sd"],
                     "n_goals": cl["n_goals"], "n_pairs": cl["n_rows"],
                     "power_for_sesoi": float(pw),
                     "cohen_dz": cl["cohen_dz"]})

out = pd.DataFrame(rows)
T = pathlib.Path("results/v3_2/tables"); T.mkdir(parents=True, exist_ok=True)
out.to_csv(T / f"decomposition_{a.model}.csv", index=False)

# per-family means, both scorers, for the record
fam = df.groupby("family")[scores + ["C_rms"]].mean().sort_values(
    scores[0], ascending=False)
fam.round(5).to_csv(T / f"family_means_{a.model}.csv")
print("\nfamily means:\n", fam.round(4).to_string())
if drift:
    print(f"\nBUDGET DRIFT on D_test (reported, not corrected): {drift}")
print(f"\nwrote {T}/decomposition_{a.model}.csv")
