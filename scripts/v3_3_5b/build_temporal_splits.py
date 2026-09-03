import sys, json, hashlib, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download, HfApi
from cantor_guard.datasets import load_jbb, load_xstest, prompt_id
from cantor_guard_v32.splits import load_split

SEED = 20260908
N = {"D_temporal_dev": 40, "D_temporal_confirm": 60, "D_traj_cal": 40,
     "D_traj_beh_dev": 40, "D_traj_beh_confirm": 60,
     "D_controller_budget": 30, "D_attack_dev_b": 30, "D_final_traj": 90}
CFG = pathlib.Path("configs/v3_3_5b")
used = set()
for v in load_split("configs/v3_2/split.json")["blocks"].values(): used |= set(v)
used |= set(json.loads(pathlib.Path("configs/v3_3_2/final_split.json").read_text())["pids"])
s3 = json.loads(pathlib.Path("configs/v3_3_3/splits.json").read_text())
used |= set(s3["D_beh"]) | set(s3["D_final"])
for f in ("configs/v3_3_4/splits.json", "configs/v3_3_5/splits.json",
          "configs/v3_3_5a/splits.json"):
    for v in json.loads(pathlib.Path(f).read_text())["blocks"].values(): used |= set(v)
xs = load_xstest(); used |= set(xs.prompt.map(prompt_id))
h, ben = load_jbb(); used |= set(h.pid) | set(ben.prompt.map(prompt_id))
print(f"prompts touched V1..V3.3.5a: {len(used)}")
api = HfApi(); info = api.dataset_info("declare-lab/HarmfulQA")
p = hf_hub_download("declare-lab/HarmfulQA", "data_for_hub.json",
                    revision=info.sha, repo_type="dataset")
df = pd.read_json(p).rename(columns={"question": "prompt"})
df["pid"] = df.prompt.map(prompt_id); df = df.drop_duplicates("pid")
fresh = df[~df.pid.isin(used)].reset_index(drop=True)
print(f"HarmfulQA: {len(fresh)} unused")
if len(fresh) < sum(N.values()): raise SystemExit("STOP -- insufficient")
rng = np.random.default_rng(SEED)
perm = fresh.iloc[rng.permutation(len(fresh))].reset_index(drop=True)
blocks, i = {}, 0
for k, n in N.items():
    blocks[k] = sorted(perm.pid[i:i+n].tolist()); i += n
allp = [x for v in blocks.values() for x in v]
assert len(set(allp)) == len(allp) and not (set(allp) & used)
payload = {"source": "declare-lab/HarmfulQA", "revision": info.sha, "seed": SEED,
           "sizes": N, "blocks": blocks,
           "sha256": {k: hashlib.sha256(json.dumps(v).encode()).hexdigest()
                      for k, v in blocks.items()},
           "disjoint_from_all_prior": True,
           "note": "D_final_P0 from V3.3.5a remains untouched and archived"}
(CFG/"splits.json").write_text(json.dumps(payload, indent=2))
for k, v in blocks.items():
    perm[perm.pid.isin(v)][["pid","prompt"]].to_csv(
        f"results/v3_3_5b/cache/{k}.csv", index=False)
    print(f"  {k:22s} n={len(v):3d}  sha {payload['sha256'][k][:16]}")
pd.DataFrame({"prompt": list(ben.prompt)[:50]}).to_csv(
    "results/v3_3_5b/cache/D_benign_traj.csv", index=False)
