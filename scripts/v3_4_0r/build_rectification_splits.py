"""Phase 2 -- fresh V3.4.0R splits.

HarmfulQA is EXHAUSTED: 28 unused prompts remain against a need above 800.
V3.4.0R therefore draws harmful prompts from LLM-LAT/harmful-dataset, which is
a DIFFERENT POPULATION from the one the frozen sensor was trained on. That is
disclosed here and gated: `D_sensor_transfer_r` measures the frozen sensor on
the new population before anything downstream is allowed to run.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from cantor_guard.datasets import DATASET_REGISTRY, prompt_id  # noqa: E402

from _common import CONFIG, RESULTS, write_json  # noqa: E402

SEED = 20260903
HARMFUL_REPO = "LLM-LAT/harmful-dataset"
HARMFUL_REVISION = "8bfba31bc6d93a5b71808fee5275ef4b6330ed91"
HARMFUL_FILE = "data/train-00000-of-00001.parquet"
HEX16 = re.compile(r"^[0-9a-f]{16}$")

HARMFUL_SIZES = {
    "D_sensor_transfer_r": 150,
    "D_eval_val_r": 200,
    "D_budget_attacked_r": 300,
    "D_final_r_harmful": 120,
}
BENIGN_SIZES = {"D_eval_val_benign_r": 60, "D_final_r_benign": 80}


def prior_prompt_hashes() -> set[str]:
    used: set[str] = set()

    def visit(value):
        if isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)
        elif isinstance(value, str) and HEX16.fullmatch(value):
            used.add(value)

    for path in sorted((ROOT / "configs").rglob("*.json")):
        if "v3_4_0r" in path.parts:
            continue
        try:
            visit(json.loads(path.read_text()))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return used


def main() -> None:
    used = prior_prompt_hashes()
    path = hf_hub_download(HARMFUL_REPO, HARMFUL_FILE, revision=HARMFUL_REVISION,
                           repo_type="dataset", local_files_only=True)
    harmful = pd.read_parquet(path)[["prompt"]].copy()
    harmful["prompt"] = harmful.prompt.astype(str).str.strip()
    harmful["pid"] = harmful.prompt.map(prompt_id)
    harmful["kind"] = "harmful"
    harmful["source"] = HARMFUL_REPO
    harmful = harmful.drop_duplicates("pid")
    harmful = harmful[~harmful.pid.isin(used)].reset_index(drop=True)

    spec = DATASET_REGISTRY["alpaca"]
    bpath = hf_hub_download(spec["repo"], spec["file"], revision=spec["revision"],
                            repo_type="dataset", local_files_only=True)
    raw = pd.read_parquet(bpath)
    raw = raw[raw["input"].fillna("").astype(str).str.len() == 0]
    benign = pd.DataFrame({"prompt": raw.instruction.astype(str),
                           "reference": raw.output.fillna("").astype(str)})
    benign["pid"] = benign.prompt.map(prompt_id)
    benign["kind"] = "benign"
    benign["source"] = spec["repo"]
    benign = benign.drop_duplicates("pid")
    benign = benign[~benign.pid.isin(used)].reset_index(drop=True)

    need_h, need_b = sum(HARMFUL_SIZES.values()), sum(BENIGN_SIZES.values())
    if len(harmful) < need_h or len(benign) < need_b:
        raise SystemExit(f"insufficient fresh data: harmful {len(harmful)}/{need_h}, "
                         f"benign {len(benign)}/{need_b}")

    rng = np.random.default_rng(SEED)
    harmful = harmful.iloc[rng.permutation(len(harmful))].reset_index(drop=True)
    benign = benign.iloc[rng.permutation(len(benign))].reset_index(drop=True)
    cache = RESULTS / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    blocks, cursor = {}, 0
    for name, size in HARMFUL_SIZES.items():
        chosen = harmful.iloc[cursor : cursor + size]
        cursor += size
        blocks[name] = sorted(chosen.pid.tolist())
        chosen.to_csv(cache / f"{name}.csv", index=False)
    cursor = 0
    for name, size in BENIGN_SIZES.items():
        chosen = benign.iloc[cursor : cursor + size]
        cursor += size
        blocks[name] = sorted(chosen.pid.tolist())
        chosen.to_csv(cache / f"{name}.csv", index=False)

    ids = [p for v in blocks.values() for p in v]
    overlap = sorted(set(ids) & used)
    if len(ids) != len(set(ids)) or overlap:
        raise AssertionError("V3.4.0R split leakage detected")

    write_json(CONFIG / "splits.json", {
        "seed": SEED,
        "POPULATION_CHANGE_DISCLOSURE": {
            "why": "HarmfulQA, the population every prior version used and the one the "
                   "frozen sensor was trained on, has only 28 unused prompts left against "
                   "a V3.4.0R requirement above 800. No subset of it can satisfy the "
                   "freshness rule, so the harmful population changes.",
            "old_population": "declare-lab/HarmfulQA",
            "new_population": HARMFUL_REPO,
            "consequence": "The final therefore tests the frozen sensor OUT OF ITS "
                           "TRAINING POPULATION. That is a real confound with the protocol "
                           "repair and must be reported as such.",
            "mitigation": "D_sensor_transfer_r measures the frozen sensor on the new "
                          "population against a preregistered gate BEFORE the budget is "
                          "calibrated or the final is opened. Nothing is refitted.",
        },
        "sources": {"harmful": {"repo": HARMFUL_REPO, "revision": HARMFUL_REVISION},
                    "benign": {"repo": spec["repo"], "revision": spec["revision"]}},
        "sizes": {**HARMFUL_SIZES, **BENIGN_SIZES},
        "blocks": blocks,
        "sha256": {n: hashlib.sha256(json.dumps(i, separators=(",", ":")).encode()).hexdigest()
                   for n, i in blocks.items()},
        "prior_hashes_excluded": len(used),
        "hash_overlap_with_all_prior": len(overlap),
        "within_v340r_overlap": len(ids) - len(set(ids)),
        "fresh_harmful_pool": int(len(harmful)),
        "fresh_benign_pool": int(len(benign)),
    })
    print(f"excluded prior prompt hashes: {len(used)}")
    print(f"fresh pool: harmful {len(harmful)} ({HARMFUL_REPO}), benign {len(benign)}")
    for name, size in {**HARMFUL_SIZES, **BENIGN_SIZES}.items():
        print(f"  {name}: n={size}")
    print("overlap with all prior: 0; within V3.4.0R: 0")


if __name__ == "__main__":
    main()
