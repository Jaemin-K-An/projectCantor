"""Build deterministic fresh calibration candidates and inherit untouched finals."""
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
from cantor_guard.datasets import prompt_id  # noqa: E402
from _common import CONFIG, RESULTS, read_json, sha256, write_json  # noqa: E402

SEED = 20260905
REPO = "LLM-LAT/harmful-dataset"
REVISION = "8bfba31bc6d93a5b71808fee5275ef4b6330ed91"
FILE = "data/train-00000-of-00001.parquet"
FILE_SHA = "51a41eaebf21eabec33ea90366d01d5bee7edb933d439c7017ad6e0107a645b1"
HEX16 = re.compile(r"^[0-9a-f]{16}$")


def normalized_tokens(text):
    """Match the pre-registered leakage audit's normalized token view."""
    return frozenset(re.findall(r"[a-z0-9]+", str(text).lower()))


def historical_registry():
    used, texts = set(), {}
    def visit(value):
        if isinstance(value, dict):
            for nested in value.values(): visit(nested)
        elif isinstance(value, list):
            for nested in value: visit(nested)
        elif isinstance(value, str) and HEX16.fullmatch(value): used.add(value)
    for path in sorted((ROOT / "configs").rglob("*.json")):
        if "v3_5_1" in path.parts: continue
        try: visit(json.loads(path.read_text()))
        except Exception: pass
    for path in sorted((ROOT / "results").rglob("*.csv")):
        if "v3_5_1" in path.parts: continue
        try: frame = pd.read_csv(path, usecols=lambda c: c in {"pid", "prompt"})
        except Exception: continue
        if "pid" in frame:
            used.update(x for x in frame.pid.dropna().astype(str) if HEX16.fullmatch(x))
        if "prompt" in frame:
            for prompt in frame.prompt.dropna().astype(str):
                pid = prompt_id(prompt); used.add(pid); texts.setdefault(pid, prompt)
    return used, texts


def main():
    audit = read_json(RESULTS / "tables/v350_failure_audit.json")
    if not audit["v350_final_untouched"]: raise SystemExit("V3.5.0 final set was touched")
    used, historical_texts = historical_registry()
    path = hf_hub_download(REPO, FILE, revision=REVISION, repo_type="dataset", local_files_only=True)
    if sha256(path) != FILE_SHA: raise SystemExit("external data hash mismatch")
    frame = pd.read_parquet(path)[["prompt"]].copy()
    frame["prompt"] = frame.prompt.astype(str).str.strip(); frame["pid"] = frame.prompt.map(prompt_id)
    frame["kind"], frame["source"] = "harmful", REPO
    frame = frame.drop_duplicates("pid"); frame = frame[~frame.pid.isin(used)].reset_index(drop=True)

    # This filter is deliberately applied before the fixed permutation.  The
    # initial, pre-model leakage audit found two >=0.9 Jaccard near-duplicates;
    # removing them here changes neither the seed nor any model-derived choice.
    historical_token_sets = [(pid, normalized_tokens(text)) for pid, text in historical_texts.items()]
    keep, excluded_near = [], []
    for row in frame.itertuples():
        candidate_tokens = normalized_tokens(row.prompt)
        match = None
        for old_pid, old_tokens in historical_token_sets:
            if not candidate_tokens or not old_tokens:
                continue
            length_ratio = min(len(candidate_tokens), len(old_tokens)) / max(len(candidate_tokens), len(old_tokens))
            if length_ratio < 0.9:
                continue
            if len(candidate_tokens & old_tokens) / len(candidate_tokens | old_tokens) >= 0.9:
                match = old_pid
                break
        if match is None:
            keep.append(row.Index)
        else:
            excluded_near.append({"candidate_pid": str(row.pid), "historical_pid": str(match)})
    frame = frame.loc[keep].reset_index(drop=True)
    rng = np.random.default_rng(SEED); frame = frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)
    frame["candidate_order"] = np.arange(len(frame))
    frame.to_csv(RESULTS / "cache/D_risk_cal_candidate_order_v351.csv", index=False)

    inherited = {}
    for old, new in (("D_final_v350_harmful", "D_final_v351_harmful"),
                     ("D_final_v350_benign", "D_final_v351_benign")):
        src = ROOT / f"results/v3_5_0/cache/{old}.csv"; dst = RESULTS / f"cache/{new}.csv"
        final = pd.read_csv(src); final.to_csv(dst, index=False)
        inherited[new] = {"n": len(final), "pids": sorted(final.pid.astype(str)),
                          "source": str(src.relative_to(ROOT)), "source_sha256": sha256(src),
                          "copy_sha256": sha256(dst)}
    candidate_ids = frame.pid.astype(str).tolist()
    payload = {
        "version": "3.5.1", "seed": SEED,
        "external": {"repo": REPO, "revision": REVISION, "file": FILE, "file_sha256": FILE_SHA},
        "historical_hashes_excluded": len(used), "candidate_pool_size": len(frame),
        "pre_model_near_duplicate_filter": {
            "metric": "normalized token-set Jaccard >= 0.9 with token-count ratio >= 0.9",
            "excluded_n": len(excluded_near),
            "excluded_pairs": excluded_near,
            "seed_changed": False,
        },
        "candidate_order_sha256": sha256(RESULTS / "cache/D_risk_cal_candidate_order_v351.csv"),
        "candidate_pid_order_sha256": hashlib.sha256(json.dumps(candidate_ids, separators=(",", ":")).encode()).hexdigest(),
        "risk_calibration": {"K_RISK": 200, "selection": "first 200 d<0 states in fixed candidate order"},
        "budget": {"n": 300, "selection": "first 300 unscanned candidate prompts after risk calibration"},
        "inherited_untouched_finals": inherited,
        "v350_final_output_or_label_seen": False,
        "risk_cal_scanned_pids": None, "risk_cal_selected_pids": None, "budget_pids": None,
    }
    write_json(CONFIG / "splits.json", payload)
    print(f"historical hashes excluded={len(used)} near duplicates excluded={len(excluded_near)} candidate pool={len(frame)}")
    print("inherited untouched final harmful=200 benign=80")


if __name__ == "__main__": main()
