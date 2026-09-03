"""Create fresh, prompt-hash-disjoint V3.3.5c harmful and benign splits."""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, "llm/src")
from cantor_guard.datasets import DATASET_REGISTRY, prompt_id  # noqa: E402


SEED = 20260909
HARMFUL_REPO = "declare-lab/HarmfulQA"
HARMFUL_REVISION = "6f1a78aed47d16c0695e4595d0159abc38197bfd"
HARMFUL_FILE = "data_for_hub.json"
SIZES = {
    "D_beh_P0_dev_335c": 40,
    "D_beh_P0_confirm_335c": 60,
    "D_window_P0_335c": 40,
    "D_budget_P0_335c": 30,
    "D_attack_dev_335c": 30,
    "D_final_P0_335c": 90,
    "D_benign_P0_335c": 50,
}
HEX16 = re.compile(r"^[0-9a-f]{16}$")


def prior_prompt_hashes(config_root: pathlib.Path = pathlib.Path("configs")) -> set[str]:
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
        if "v3_3_5c" in path.parts:
            continue
        try:
            visit(json.loads(path.read_text()))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return used


def _harmful_source() -> pd.DataFrame:
    path = hf_hub_download(
        HARMFUL_REPO,
        HARMFUL_FILE,
        revision=HARMFUL_REVISION,
        repo_type="dataset",
        local_files_only=True,
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
    required_harmful = sum(n for name, n in SIZES.items() if "benign" not in name)
    if len(harmful) < required_harmful or len(benign) < SIZES["D_benign_P0_335c"]:
        raise SystemExit(
            f"insufficient fresh data: harmful {len(harmful)}/{required_harmful}, "
            f"benign {len(benign)}/{SIZES['D_benign_P0_335c']}"
        )
    rng = np.random.default_rng(SEED)
    harmful = harmful.iloc[rng.permutation(len(harmful))].reset_index(drop=True)
    benign = benign.iloc[rng.permutation(len(benign))].reset_index(drop=True)
    blocks: dict[str, list[str]] = {}
    cache = pathlib.Path("results/v3_3_5c/cache")
    cache.mkdir(parents=True, exist_ok=True)
    cursor = 0
    for name, size in SIZES.items():
        source = benign if "benign" in name else harmful
        if "benign" in name:
            selected = source.iloc[:size]
        else:
            selected = source.iloc[cursor : cursor + size]
            cursor += size
        blocks[name] = sorted(selected.pid.tolist())
        selected.to_csv(cache / f"{name}.csv", index=False)
    all_ids = [pid for ids in blocks.values() for pid in ids]
    overlap = sorted(set(all_ids) & used)
    if len(all_ids) != len(set(all_ids)) or overlap:
        raise AssertionError("V3.3.5c split leakage detected")
    payload = {
        "seed": SEED,
        "sources": {
            "harmful": {"repo": HARMFUL_REPO, "revision": HARMFUL_REVISION},
            "benign": {"repo": DATASET_REGISTRY["alpaca"]["repo"], "revision": DATASET_REGISTRY["alpaca"]["revision"]},
        },
        "sizes": SIZES,
        "blocks": blocks,
        "sha256": {
            name: hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
            for name, ids in blocks.items()
        },
        "hash_overlap_with_all_prior": len(overlap),
        "within_v335c_overlap": len(all_ids) - len(set(all_ids)),
        "disjoint_from_v335a_p0": True,
        "disjoint_from_v335b_temporal": True,
        "disjoint_from_prior_final_sets": True,
        "disjoint_from_direction_estimation": True,
        "prior_hashes_excluded": len(used),
    }
    pathlib.Path("configs/v3_3_5c/splits.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"excluded prior prompt hashes: {len(used)}")
    for name, size in SIZES.items():
        print(f"{name}: n={size}, sha256={payload['sha256'][name][:16]}")
    print("overlap with all prior: 0; within V3.3.5c: 0")


if __name__ == "__main__":
    main()
