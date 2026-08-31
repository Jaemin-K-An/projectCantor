"""V3.3.3 -- global prompt-usage registry, D_beh and D_final."""
import sys, json, hashlib, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
from cantor_guard.datasets import load_jbb, load_xstest, prompt_id
from cantor_guard_v32.splits import load_split

SEED, N_BEH, N_FINAL = 20260902, 50, 70
TAB = pathlib.Path("results/v3_3_3/tables"); TAB.mkdir(parents=True, exist_ok=True)
CFG = pathlib.Path("configs/v3_3_3"); CFG.mkdir(parents=True, exist_ok=True)

rows = []
sp = load_split("configs/v3_2/split.json")
harm, ben = load_jbb()
H = harm.set_index("pid")
for blk, pids in sp["blocks"].items():
    for p in pids:
        rows.append({"dataset": "jbb_harmful", "prompt_id": p,
                     "prompt_hash": p, "first_version_used": "V3.2",
                     "purpose": blk, "split": blk,
                     "ever_used_for_tuning": blk != "D_test",
                     "ever_used_for_final": blk == "D_test",
                     "raw_text_committed": False})
xs = load_xstest(); xh = xs[xs.is_harmful].copy(); xh["pid"] = xh.prompt.map(prompt_id)
v332 = set(json.loads((pathlib.Path("configs/v3_3_2/final_split.json")).read_text())["pids"])
for p in xh.pid:
    if p in v332:
        rows.append({"dataset": "xstest_unsafe", "prompt_id": p, "prompt_hash": p,
                     "first_version_used": "V3.3.2", "purpose": "D_final_v332",
                     "split": "D_final_v332", "ever_used_for_tuning": False,
                     "ever_used_for_final": True, "raw_text_committed": False})
for p in xs[~xs.is_harmful].prompt.map(prompt_id):
    rows.append({"dataset": "xstest_safe", "prompt_id": p, "prompt_hash": p,
                 "first_version_used": "V3.1", "purpose": "benign_utility_probe",
                 "split": "benign", "ever_used_for_tuning": True,
                 "ever_used_for_final": False, "raw_text_committed": False})

used = set(x["prompt_id"] for x in rows)
pool = xh[~xh.pid.isin(used)]
print(f"registry: {len(rows)} prompts already used")
print(f"XSTest unsafe available and NEVER used: {len(pool)}")
if len(pool) < N_BEH + 40:
    raise SystemExit("STOP -- insufficient untouched prompts")

rng = np.random.default_rng(SEED)
perm = pool.iloc[rng.permutation(len(pool))]
d_beh = perm.iloc[:N_BEH].sort_values("pid")
d_fin = perm.iloc[N_BEH:N_BEH + N_FINAL].sort_values("pid")
assert not (set(d_beh.pid) & set(d_fin.pid)) and not (set(d_fin.pid) & used)

for df, name in ((d_beh, "D_beh"), (d_fin, "D_final_v333")):
    for p in df.pid:
        rows.append({"dataset": "xstest_unsafe", "prompt_id": p, "prompt_hash": p,
                     "first_version_used": "V3.3.3", "purpose": name, "split": name,
                     "ever_used_for_tuning": name == "D_beh",
                     "ever_used_for_final": name == "D_final_v333",
                     "raw_text_committed": False})
pd.DataFrame(rows).to_csv(TAB / "global_prompt_usage_registry.csv", index=False)

payload = {"seed": SEED, "source": "xstest_v2_unsafe",
           "D_beh": sorted(d_beh.pid.tolist()),
           "D_final": sorted(d_fin.pid.tolist()),
           "n_beh": len(d_beh), "n_final": len(d_fin),
           "D_beh_sha256": hashlib.sha256(json.dumps(sorted(d_beh.pid.tolist())).encode()).hexdigest(),
           "D_final_sha256": hashlib.sha256(json.dumps(sorted(d_fin.pid.tolist())).encode()).hexdigest(),
           "population_note": ("XSTest unsafe; same population as V3.3.2 D_final but "
                               "DISJOINT prompts. External validity is restricted to "
                               "this population, not JBB."),
           "jbb_unused": 0}
(CFG / "splits.json").write_text(json.dumps(payload, indent=2))
d_beh[["pid", "prompt"]].to_csv("results/v3_3_3/cache/d_beh_prompts.csv", index=False)
d_fin[["pid", "prompt"]].to_csv("results/v3_3_3/cache/d_final_prompts.csv", index=False)
print(f"D_beh   {len(d_beh)}  sha {payload['D_beh_sha256'][:16]}")
print(f"D_final {len(d_fin)}  sha {payload['D_final_sha256'][:16]}")
print(f"registry -> {TAB}/global_prompt_usage_registry.csv")
