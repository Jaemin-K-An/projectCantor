"""Phase 1: fresh nonoverlapping V3.5.0 splits from pinned populations."""
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
from _common import CONFIG, RESULTS, sha256, write_json  # noqa: E402

SEED = 20260904
HARMFUL_REPO = "LLM-LAT/harmful-dataset"
HARMFUL_REVISION = "8bfba31bc6d93a5b71808fee5275ef4b6330ed91"
HARMFUL_FILE = "data/train-00000-of-00001.parquet"
HARMFUL_FILE_SHA = "51a41eaebf21eabec33ea90366d01d5bee7edb933d439c7017ad6e0107a645b1"
HEX16 = re.compile(r"^[0-9a-f]{16}$")
HARMFUL_SIZES = {
    "D_risk_window_cal": 300,
    "D_budget_v350": 300,
    "D_attack_dev_v350": 50,
    "D_final_v350_harmful": 200,
}
BENIGN_SIZES = {"D_final_v350_benign": 80}


def historical_prompt_registry():
    used, texts = set(), {}

    def visit(value):
        if isinstance(value, dict):
            for nested in value.values(): visit(nested)
        elif isinstance(value, list):
            for nested in value: visit(nested)
        elif isinstance(value, str) and HEX16.fullmatch(value):
            used.add(value)

    for path in sorted((ROOT / "configs").rglob("*.json")):
        if "v3_5_0" in path.parts:
            continue
        try: visit(json.loads(path.read_text()))
        except (json.JSONDecodeError, UnicodeDecodeError): pass
    for path in sorted((ROOT / "results").rglob("*.csv")):
        if "v3_5_0" in path.parts:
            continue
        try:
            frame = pd.read_csv(path, usecols=lambda c: c in {"pid", "prompt"})
        except Exception:
            continue
        if "pid" in frame:
            used.update(x for x in frame.pid.dropna().astype(str) if HEX16.fullmatch(x))
        if "prompt" in frame:
            for prompt in frame.prompt.dropna().astype(str):
                pid = prompt_id(prompt)
                used.add(pid)
                texts.setdefault(pid, prompt)
    return used, texts


def main() -> None:
    used, _ = historical_prompt_registry()
    hpath = hf_hub_download(HARMFUL_REPO, HARMFUL_FILE, revision=HARMFUL_REVISION,
                            repo_type="dataset", local_files_only=True)
    if sha256(hpath) != HARMFUL_FILE_SHA:
        raise SystemExit("pinned harmful dataset file hash mismatch")
    harmful = pd.read_parquet(hpath)[["prompt"]].copy()
    harmful["prompt"] = harmful.prompt.astype(str).str.strip()
    harmful["pid"] = harmful.prompt.map(prompt_id)
    harmful["kind"], harmful["source"] = "harmful", HARMFUL_REPO
    harmful = harmful.drop_duplicates("pid")
    harmful = harmful[~harmful.pid.isin(used)].reset_index(drop=True)

    spec = DATASET_REGISTRY["alpaca"]
    bpath = hf_hub_download(spec["repo"], spec["file"], revision=spec["revision"],
                            repo_type="dataset", local_files_only=True)
    raw = pd.read_parquet(bpath)
    raw = raw[raw["input"].fillna("").astype(str).str.len() == 0]
    benign = pd.DataFrame({"prompt": raw.instruction.astype(str)})
    benign["pid"] = benign.prompt.map(prompt_id)
    benign["kind"], benign["source"] = "benign", spec["repo"]
    benign = benign.drop_duplicates("pid")
    benign = benign[~benign.pid.isin(used)].reset_index(drop=True)

    need_h, need_b = sum(HARMFUL_SIZES.values()), sum(BENIGN_SIZES.values())
    if len(harmful) < need_h or len(benign) < need_b:
        raise SystemExit(f"insufficient fresh pool harmful={len(harmful)}/{need_h}, benign={len(benign)}/{need_b}")
    rng = np.random.default_rng(SEED)
    harmful = harmful.iloc[rng.permutation(len(harmful))].reset_index(drop=True)
    benign = benign.iloc[rng.permutation(len(benign))].reset_index(drop=True)
    blocks, cursor = {}, 0
    for name, size in HARMFUL_SIZES.items():
        chosen = harmful.iloc[cursor:cursor + size].copy(); cursor += size
        chosen.to_csv(RESULTS / f"cache/{name}.csv", index=False)
        blocks[name] = sorted(chosen.pid.tolist())
    cursor = 0
    for name, size in BENIGN_SIZES.items():
        chosen = benign.iloc[cursor:cursor + size].copy(); cursor += size
        chosen.to_csv(RESULTS / f"cache/{name}.csv", index=False)
        blocks[name] = sorted(chosen.pid.tolist())
    ids = [pid for block in blocks.values() for pid in block]
    overlap = sorted(set(ids) & used)
    if overlap or len(ids) != len(set(ids)):
        raise AssertionError("historical or within-V3.5.0 prompt leakage")
    write_json(CONFIG / "splits.json", {
        "version": "3.5.0", "seed": SEED,
        "sources": {"harmful": {"repo": HARMFUL_REPO, "revision": HARMFUL_REVISION,
                                  "file_sha256": HARMFUL_FILE_SHA},
                    "benign": {"repo": spec["repo"], "revision": spec["revision"]}},
        "sizes": {**HARMFUL_SIZES, **BENIGN_SIZES}, "blocks": blocks,
        "block_sha256": {name: hashlib.sha256(json.dumps(ids_, separators=(",", ":")).encode()).hexdigest()
                         for name, ids_ in blocks.items()},
        "historical_prompt_hashes_excluded": len(used),
        "historical_exact_overlap": len(overlap), "within_v350_overlap": len(ids) - len(set(ids)),
        "fresh_harmful_pool": int(len(harmful)), "fresh_benign_pool": int(len(benign)),
    })
    print(f"historical hashes excluded: {len(used)}")
    for name, size in {**HARMFUL_SIZES, **BENIGN_SIZES}.items(): print(f"{name}: n={size}")
    print("historical overlap=0; within-V3.5.0 overlap=0")


if __name__ == "__main__":
    main()
