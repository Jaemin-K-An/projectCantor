"""V3.2 PHASE 2/3 -- build and audit the five-way split."""
import sys, json, pathlib
sys.path.insert(0, "llm/src")
import pandas as pd
from cantor_guard.datasets import load_jbb
from cantor_guard_v32.splits import (make_split, leakage_audit, save_split,
                                     DEFAULT_SIZES, BLOCKS)

harmful, benign = load_jbb()
SALT = "cantor-v3.2"
sp = make_split(harmful, salt=SALT, sizes=DEFAULT_SIZES)

print("block sizes:", sp.block.value_counts().reindex(list(BLOCKS)).to_dict())
print("\ncategory balance (rows = block):")
print(pd.crosstab(sp.block, sp.category).reindex(list(BLOCKS)).to_string())

leak = leakage_audit(sp, threshold=0.6)
print(f"\nleakage audit @ jaccard>=0.6 : {len(leak)} cross-block near-duplicates")
if len(leak):
    print(leak.sort_values("jaccard", ascending=False).head(20).to_string(index=False))
for thr in (0.4, 0.5, 0.7):
    print(f"  (jaccard>={thr}: {len(leakage_audit(sp, threshold=thr))} pairs)")

pathlib.Path("results/v3_2/tables").mkdir(parents=True, exist_ok=True)
leak.to_csv("results/v3_2/tables/leakage_audit.csv", index=False)
payload = save_split(sp, "configs/v3_2/split.json", SALT, DEFAULT_SIZES)
print("\nsplit_sha256:", payload["split_sha256"])
# Prompt text stays out of the tracked tree; ids + block only.
sp[["pid", "goal_group", "category", "block"]].to_csv(
    "results/v3_2/tables/split_assignment.csv", index=False)
sp[["pid", "prompt", "category", "block"]].to_csv(
    "results/v3_2/private/split_with_prompts.csv", index=False)
print("BLOCKER" if len(leak) else "OK", "- leakage")
