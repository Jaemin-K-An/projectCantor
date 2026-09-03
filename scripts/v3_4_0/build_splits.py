"""Fresh V3.4.0 splits, disjoint from every prompt hash used in V1..V3.3.5c.

The sensor pipeline needs far more prompts than any prior version because the
probe is trained, tuned and confirmed on disjoint prompt sets, and because the
semantic evaluator must be validated on data independent of sensor training.
Sizes are therefore fixed by the remaining fresh supply, and declared here
before any residual is extracted.
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "llm/src"))
from cantor_guard.datasets import DATASET_REGISTRY, prompt_id  # noqa: E402

from _common import CONFIG, RESULTS, write_json  # noqa: E402

SEED = 20260903
HARMFUL_REPO = "declare-lab/HarmfulQA"
HARMFUL_REVISION = "6f1a78aed47d16c0695e4595d0159abc38197bfd"
HARMFUL_FILE = "data_for_hub.json"
HEX16 = re.compile(r"^[0-9a-f]{16}$")

HARMFUL_SIZES = {
    "D_eval_val_harmful": 50,
    "D_sensor_train": 180,
    "D_sensor_tune": 45,
    "D_sensor_confirm": 85,
    "D_actuator_validate": 30,
    "D_window_cal": 30,
    "D_controller_budget": 25,
    "D_attack_dev": 25,
    "D_final_harmful": 80,
}
BENIGN_SIZES = {"D_eval_val_benign": 40, "D_final_benign": 60}


def prior_prompt_hashes(config_root: pathlib.Path = pathlib.Path("configs")) -> set[str]:
    """Every 16-hex prompt id recorded in any pre-V3.4.0 config."""
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

    for path in sorted(config_root.rglob("*.json")):
        if "v3_4_0" in path.parts:
            continue
        try:
            visit(json.loads(path.read_text()))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return used


def _harmful_source() -> pd.DataFrame:
    path = hf_hub_download(
        HARMFUL_REPO, HARMFUL_FILE, revision=HARMFUL_REVISION,
        repo_type="dataset", local_files_only=True,
    )
    raw = pd.read_json(path).rename(columns={"question": "prompt"})
    raw["prompt"] = raw.prompt.astype(str)
    raw["pid"] = raw.prompt.map(prompt_id)
    raw["kind"] = "harmful"
    raw["source"] = HARMFUL_REPO
    return raw.drop_duplicates("pid")[["pid", "prompt", "kind", "source"]]


def _benign_source() -> pd.DataFrame:
    spec = DATASET_REGISTRY["alpaca"]
    path = hf_hub_download(
        spec["repo"], spec["file"], revision=spec["revision"],
        repo_type="dataset", local_files_only=True,
    )
    raw = pd.read_parquet(path)
    raw = raw[raw["input"].fillna("").astype(str).str.len() == 0]
    out = pd.DataFrame({
        "prompt": raw.instruction.astype(str),
        "reference": raw.output.fillna("").astype(str),
    })
    out["pid"] = out.prompt.map(prompt_id)
    out["kind"] = "benign"
    out["source"] = spec["repo"]
    return out.drop_duplicates("pid")[["pid", "prompt", "reference", "kind", "source"]]


def main() -> None:
    used = prior_prompt_hashes()
    harmful = _harmful_source()
    benign = _benign_source()
    harmful = harmful[~harmful.pid.isin(used)].reset_index(drop=True)
    benign = benign[~benign.pid.isin(used)].reset_index(drop=True)
    need_h, need_b = sum(HARMFUL_SIZES.values()), sum(BENIGN_SIZES.values())
    if len(harmful) < need_h or len(benign) < need_b:
        raise SystemExit(f"insufficient fresh data: harmful {len(harmful)}/{need_h}, benign {len(benign)}/{need_b}")
    rng = np.random.default_rng(SEED)
    harmful = harmful.iloc[rng.permutation(len(harmful))].reset_index(drop=True)
    benign = benign.iloc[rng.permutation(len(benign))].reset_index(drop=True)
    cache = RESULTS / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    blocks: dict[str, list[str]] = {}
    cursor = 0
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
    all_ids = [pid for ids in blocks.values() for pid in ids]
    overlap = sorted(set(all_ids) & used)
    if len(all_ids) != len(set(all_ids)) or overlap:
        raise AssertionError("V3.4.0 split leakage detected")
    write_json(CONFIG / "splits.json", {
        "seed": SEED,
        "sources": {
            "harmful": {"repo": HARMFUL_REPO, "revision": HARMFUL_REVISION},
            "benign": {"repo": DATASET_REGISTRY["alpaca"]["repo"],
                       "revision": DATASET_REGISTRY["alpaca"]["revision"]},
        },
        "sizes": {**HARMFUL_SIZES, **BENIGN_SIZES},
        "blocks": blocks,
        "sha256": {n: hashlib.sha256(json.dumps(i, separators=(",", ":")).encode()).hexdigest()
                   for n, i in blocks.items()},
        "prior_hashes_excluded": len(used),
        "hash_overlap_with_all_prior": len(overlap),
        "within_v340_overlap": len(all_ids) - len(set(all_ids)),
        "fresh_harmful_pool": int(len(harmful)),
        "fresh_benign_pool": int(len(benign)),
        "evaluator_validation_independent_of_sensor_training": True,
        "note": "Sensor train/tune/confirm are mutually disjoint; the evaluator "
                "validation blocks are disjoint from every sensor block, so an "
                "evaluator validated on them is independent of the sensor labels.",
    })
    print(f"excluded prior prompt hashes: {len(used)}")
    print(f"fresh pool: harmful {len(harmful)}, benign {len(benign)}")
    for name, size in {**HARMFUL_SIZES, **BENIGN_SIZES}.items():
        print(f"  {name}: n={size}")
    print("overlap with all prior: 0; within V3.4.0: 0")


if __name__ == "__main__":
    main()
