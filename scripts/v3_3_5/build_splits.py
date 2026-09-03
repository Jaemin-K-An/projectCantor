"""V3.3.5 PHASE 12 -- fresh splits from a FOURTH distinct population."""
import sys, json, hashlib, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download, HfApi
from cantor_guard.datasets import load_jbb, load_xstest, prompt_id
from cantor_guard_v32.splits import load_split

SEED = 20260906
N = {"D_beh_g1_dev": 40, "D_beh_g1_confirm": 60, "D_window_cal": 40,
     "D_budget_335": 30, "D_attack_dev": 30, "D_final_335": 90}
CFG = pathlib.Path("configs/v3_3_5"); TAB = pathlib.Path("results/v3_3_5/tables")

used = set()
for v in load_split("configs/v3_2/split.json")["blocks"].values():
    used |= set(v)
used |= set(json.loads(pathlib.Path("configs/v3_3_2/final_split.json").read_text())["pids"])
s3 = json.loads(pathlib.Path("configs/v3_3_3/splits.json").read_text())
used |= set(s3["D_beh"]) | set(s3["D_final"])
for v in json.loads(pathlib.Path("configs/v3_3_4/splits.json").read_text())["blocks"].values():
    used |= set(v)
xs = load_xstest(); used |= set(xs.prompt.map(prompt_id))
h, ben = load_jbb(); used |= set(h.pid) | set(ben.prompt.map(prompt_id))
print(f"prompts touched in V1..V3.3.4: {len(used)}")

api = HfApi(); info = api.dataset_info("declare-lab/HarmfulQA")
p = hf_hub_download("declare-lab/HarmfulQA", "data_for_hub.json",
                    revision=info.sha, repo_type="dataset")
df = pd.read_json(p).rename(columns={"question": "prompt"})
df["pid"] = df.prompt.map(prompt_id)
df = df.drop_duplicates("pid")
fresh = df[~df.pid.isin(used)].reset_index(drop=True)
print(f"declare-lab/HarmfulQA rev {info.sha[:10]}: {len(df)} unique, {len(fresh)} fresh")
need = sum(N.values())
if len(fresh) < need:
    raise SystemExit(f"STOP -- need {need}, have {len(fresh)}")

rng = np.random.default_rng(SEED)
perm = fresh.iloc[rng.permutation(len(fresh))].reset_index(drop=True)
blocks, i = {}, 0
for k, n in N.items():
    blocks[k] = sorted(perm.pid[i:i+n].tolist()); i += n
# benign utility prompts: XSTest safe (probes only, never an endpoint)
benign = list(xs[~xs.is_harmful].prompt)[:50]
allp = [x for v in blocks.values() for x in v]
assert len(set(allp)) == len(allp) and not (set(allp) & used)

payload = {"source": "declare-lab/HarmfulQA", "revision": info.sha, "seed": SEED,
           "sizes": N, "blocks": blocks,
           "sha256": {k: hashlib.sha256(json.dumps(v).encode()).hexdigest()
                      for k, v in blocks.items()},
           "n_benign": len(benign),
           "population_note": ("Question-form harmful queries organised by "
                               "topic/subtopic -- a FOURTH distinct population "
                               "after JBB behaviours, XSTest contrast prompts "
                               "and AdvBench imperatives. External validity is "
                               "restricted to it."),
           "hash_overlap_with_all_prior": 0}
(CFG/"splits.json").write_text(json.dumps(payload, indent=2))
for k, v in blocks.items():
    perm[perm.pid.isin(v)][["pid", "prompt"]].to_csv(
        f"results/v3_3_5/cache/{k}.csv", index=False)
    print(f"  {k:18s} n={len(v):3d}  sha {payload['sha256'][k][:16]}")
pd.DataFrame({"prompt": benign}).to_csv("results/v3_3_5/cache/D_benign_335.csv", index=False)
pd.DataFrame([{"dataset": "harmfulqa", "prompt_id": x, "block": k,
               "first_version_used": "V3.3.5", "raw_text_committed": False}
              for k, v in blocks.items() for x in v]).to_csv(
    TAB/"v335_prompt_registry.csv", index=False)
print(f"  benign             n={len(benign)}\nwrote {CFG}/splits.json")
