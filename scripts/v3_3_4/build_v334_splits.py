"""V3.3.4 PHASE 12 -- fresh splits from an untouched public population.

XSTest unsafe is down to 20 unused prompts, so it cannot supply D_final_334.
AdvBench proper is gated. `mlabonne/harmful_behaviors` (416 prompts, ungated,
AdvBench-derived) is used instead, deduplicated by prompt hash against every
prompt this project has ever touched.

POPULATION SHIFT, stated plainly: these are AdvBench-style imperative harmful
instructions, a third distinct population after JBB behaviours and XSTest
unsafe contrast prompts. Results generalise to it, not to the earlier two.
"""
import sys, json, hashlib, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download, HfApi
from cantor_guard.datasets import load_jbb, load_xstest, prompt_id
from cantor_guard_v32.splits import load_split

SEED = 20260904
N = {"D_beh_dev": 40, "D_beh_confirm": 60, "D_budget_new": 30, "D_final_334": 90}
CFG = pathlib.Path("configs/v3_3_4"); TAB = pathlib.Path("results/v3_3_4/tables")

used = set()
for v in load_split("configs/v3_2/split.json")["blocks"].values():
    used |= set(v)
used |= set(json.loads(pathlib.Path("configs/v3_3_2/final_split.json").read_text())["pids"])
s333 = json.loads(pathlib.Path("configs/v3_3_3/splits.json").read_text())
used |= set(s333["D_beh"]) | set(s333["D_final"])
xs = load_xstest(); used |= set(xs.prompt.map(prompt_id))
h, ben = load_jbb(); used |= set(h.pid) | set(ben.prompt.map(prompt_id))
print(f"prompts ever touched by this project: {len(used)}")

api = HfApi(); info = api.dataset_info("mlabonne/harmful_behaviors")
p = hf_hub_download("mlabonne/harmful_behaviors",
                    "data/train-00000-of-00001.parquet",
                    revision=info.sha, repo_type="dataset")
df = pd.read_parquet(p).rename(columns={"text": "prompt"})
df["pid"] = df.prompt.map(prompt_id)
df = df.drop_duplicates("pid")
fresh = df[~df.pid.isin(used)].reset_index(drop=True)
print(f"mlabonne/harmful_behaviors rev {info.sha[:12]}: {len(df)} unique, "
      f"{len(fresh)} never touched")
need = sum(N.values())
if len(fresh) < need:
    raise SystemExit(f"STOP -- need {need}, have {len(fresh)}")

rng = np.random.default_rng(SEED)
perm = fresh.iloc[rng.permutation(len(fresh))].reset_index(drop=True)
blocks, i = {}, 0
for k, n in N.items():
    blocks[k] = sorted(perm.pid[i:i + n].tolist()); i += n
allp = [x for v in blocks.values() for x in v]
assert len(set(allp)) == len(allp) and not (set(allp) & used)

payload = {"source": "mlabonne/harmful_behaviors", "revision": info.sha,
           "seed": SEED, "sizes": N, "blocks": blocks,
           "sha256": {k: hashlib.sha256(json.dumps(v).encode()).hexdigest()
                      for k, v in blocks.items()},
           "population_note": ("AdvBench-style imperative harmful instructions; "
                               "a THIRD distinct population after JBB behaviours "
                               "and XSTest unsafe. Results generalise to it only."),
           "disjoint_from_all_prior": True,
           "xstest_unsafe_remaining": 20}
(CFG/"splits.json").write_text(json.dumps(payload, indent=2))
for k, v in blocks.items():
    perm[perm.pid.isin(v)][["pid", "prompt"]].to_csv(
        f"results/v3_3_4/cache/{k}.csv", index=False)
    print(f"  {k:16s} n={len(v):3d}  sha {payload['sha256'][k][:16]}")
reg = pd.DataFrame([{"dataset": "mlabonne_harmful_behaviors", "prompt_id": x,
                     "block": k, "first_version_used": "V3.3.4",
                     "raw_text_committed": False}
                    for k, v in blocks.items() for x in v])
reg.to_csv(TAB/"v334_prompt_registry.csv", index=False)
print(f"\nwrote {CFG}/splits.json")
