"""V3.2 PHASE 1 -- quantify the V3.1 pseudoreplication defect.

Re-runs every V3.1 headline comparison twice: once with the V3.1 procedure
(bootstrap over cells) and once with the goal as the resampling unit. Nothing
is re-collected; this is a pure re-analysis of the frozen V3.1 test data.
"""
import sys, json, pathlib
import pandas as pd
sys.path.insert(0, "llm/src")
from cantor_guard_v32.cluster_stats import (
    cluster_bootstrap_by_goal, hierarchical_bootstrap,
    naive_cell_bootstrap, tost_equivalence)

SESOI = 0.03
# NOTE (V3.2 defect D3): in the V3.1 table `seed` is the LAYOUT-INSTANCE index,
# not a generation replicate. The deterministic families carry seed=0 only,
# while T5/T6 carry seeds 1..3 (three random draws of the same family). Keying
# on `seed` would therefore fail to pair the random families at all. The design
# cell is (attack, delta, eps, pid); instances are averaged within a cell so the
# comparison is family-vs-family, which is the question actually asked.
KEYS = ["attack", "delta", "eps", "pid"]
SRC = "results/v3_1/raw/v31_llm_direct_v2_qwen2.5-0.5b-instruct.csv"
REF = "T7_cantor"

d_raw = pd.read_csv(SRC)
counts = d_raw.groupby("family").size()
n_inst = d_raw.groupby("family").seed.nunique()
d = (d_raw.groupby(["family"] + KEYS, as_index=False)
          .agg(safe=("safe", "mean"), n_inst=("seed", "nunique")))
# V3.1 dropped the budget-mismatched families mid-run; keep only families that
# cover the identical design cells as the reference, so every comparison stays
# strictly paired.
ref_cells = set(map(tuple, d[d.family == REF][KEYS].values))
rows = []
for fam in sorted(d.family.unique()):
    if fam == REF:
        continue
    sub = d[d.family == fam]
    cells = set(map(tuple, sub[KEYS].values))
    common = ref_cells & cells
    if len(common) < 100:
        rows.append({"family": fam, "status": "SKIPPED_INSUFFICIENT_PAIRS",
                     "n_pairs": len(common)})
        continue
    a = d[(d.family == REF)].set_index(KEYS).loc[sorted(common), "safe"]
    b = sub.set_index(KEYS).loc[sorted(common), "safe"]
    m = pd.DataFrame({"safe_ref": a.values, "safe_alt": b.values},
                     index=a.index).reset_index()
    nv = naive_cell_bootstrap(m, "safe_ref", "safe_alt", seed=20250830)
    cl = cluster_bootstrap_by_goal(m, "safe_ref", "safe_alt", seed=20250830)
    hi = hierarchical_bootstrap(m, "safe_ref", "safe_alt", n_boot=4000, seed=20250830)
    eq_n, eq_c = tost_equivalence(nv, SESOI), tost_equivalence(cl, SESOI)
    rows.append({
        "family": fam, "status": "OK", "n_pairs": len(common),
        "n_goals": cl["n_goals"], "mean_diff": round(cl["mean_diff"], 5),
        "naive_lo": round(nv["ci_lo"], 5), "naive_hi": round(nv["ci_hi"], 5),
        "cluster_lo": round(cl["ci_lo"], 5), "cluster_hi": round(cl["ci_hi"], 5),
        "hier_lo": round(hi["ci_lo"], 5), "hier_hi": round(hi["ci_hi"], 5),
        "width_ratio": round(cl["half_width"] / nv["half_width"], 3)
                        if nv["half_width"] > 0 else float("nan"),
        "between_goal_sd": round(cl["between_goal_sd"], 5),
        "cohen_dz": round(cl["cohen_dz"], 3),
        "naive_signif": bool(nv["ci_lo"] > 0 or nv["ci_hi"] < 0),
        "cluster_signif": bool(cl["ci_lo"] > 0 or cl["ci_hi"] < 0),
        "naive_equivalent": eq_n["equivalent"],
        "cluster_equivalent": eq_c["equivalent"],
    })

out = pd.DataFrame(rows)
pathlib.Path("results/v3_2/tables").mkdir(parents=True, exist_ok=True)
out.to_csv("results/v3_2/tables/v31_pseudoreplication_audit.csv", index=False)

ok = out[out.status == "OK"]
print("V3.1 rows per family / layout instances per family:")
print(pd.DataFrame({"rows": counts, "instances": n_inst}).to_string(), "\n")
print(ok[["family", "n_pairs", "n_goals", "mean_diff", "naive_lo", "naive_hi",
          "cluster_lo", "cluster_hi", "width_ratio"]].to_string(index=False))
print("\nconclusion flips (naive -> cluster):")
for _, r in ok.iterrows():
    if r.naive_signif != r.cluster_signif:
        print(f"  {r.family}: significant={r.naive_signif} -> {r.cluster_signif}  FLIP")
    if r.naive_equivalent != r.cluster_equivalent:
        print(f"  {r.family}: equivalent={r.naive_equivalent} -> {r.cluster_equivalent}  FLIP")
print("\nbetween-goal SD (median):", round(ok.between_goal_sd.median(), 5))
skipped = out[out.status != "OK"]
if len(skipped):
    print("\nskipped:\n", skipped.to_string(index=False))
