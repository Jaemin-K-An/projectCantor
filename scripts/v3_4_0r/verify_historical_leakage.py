"""Audit exact, normalized, and conservative token-overlap leakage.

Only prompt text is read.  Model outputs and labels are never consulted, and
the report stores hashes rather than harmful prompt text.
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import unicodedata

import pandas as pd

from _common import RESULTS, ROOT, write_json


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def digest(text: str) -> str:
    return hashlib.sha256(str(text).strip().encode()).hexdigest()[:16]


def token_set(text: str) -> frozenset[str]:
    return frozenset(normalise(text).split())


def prompt_rows(paths) -> list[tuple[str, str, pathlib.Path]]:
    out = []
    for path in paths:
        try:
            frame = pd.read_csv(path, usecols=lambda c: c in {"pid", "prompt"})
        except Exception:
            continue
        if "prompt" not in frame:
            continue
        for row in frame.drop_duplicates("prompt").itertuples(index=False):
            text = str(row.prompt)
            pid = str(getattr(row, "pid", digest(text)))
            out.append((pid, text, path))
    return out


def near_pairs(left, right, *, same_collection: bool, threshold: float = 0.90):
    right_tokens = [(pid, token_set(text)) for pid, text, _ in right]
    pairs = []
    seen = set()
    for lpid, ltext, _ in left:
        lt = token_set(ltext)
        if len(lt) < 5:
            continue
        for rpid, rt in right_tokens:
            if same_collection and lpid >= rpid:
                continue
            if lpid == rpid or len(rt) < 5:
                continue
            key = tuple(sorted((lpid, rpid)))
            if key in seen:
                continue
            union = lt | rt
            similarity = len(lt & rt) / len(union) if union else 0.0
            if similarity >= threshold:
                seen.add(key)
                pairs.append({"left_pid": lpid, "right_pid": rpid,
                              "token_jaccard": float(similarity)})
    return pairs


def main() -> None:
    external_paths = sorted((RESULTS / "cache").glob("D_*.csv"))
    historical_paths = [p for p in (ROOT / "results").glob("*/cache/*.csv")
                        if "v3_4_0r" not in p.parts]
    external = prompt_rows(external_paths)
    historical = prompt_rows(historical_paths)
    external_by_pid = {pid: text for pid, text, _ in external}
    historical_by_pid = {pid: text for pid, text, _ in historical}
    exact = sorted(set(external_by_pid) & set(historical_by_pid))

    external_norm = {}
    for pid, text in external_by_pid.items():
        external_norm.setdefault(normalise(text), []).append(pid)
    historical_norm = {}
    for pid, text in historical_by_pid.items():
        historical_norm.setdefault(normalise(text), []).append(pid)
    normalized_historical = []
    for key in set(external_norm) & set(historical_norm):
        for left in external_norm[key]:
            for right in historical_norm[key]:
                if left != right:
                    normalized_historical.append({"external_pid": left, "historical_pid": right})
    normalized_internal = [ids for ids in external_norm.values() if len(set(ids)) > 1]
    near_historical = near_pairs(external, historical, same_collection=False)
    near_internal = near_pairs(external, external, same_collection=True)
    payload = {
        "audit_scope": "prompt text only; no output or outcome labels read",
        "external_files": [str(p.relative_to(ROOT)) for p in external_paths],
        "historical_files_scanned": len(historical_paths),
        "external_unique_exact": len(external_by_pid),
        "historical_unique_exact": len(historical_by_pid),
        "historical_exact_overlap_count": len(exact),
        "historical_exact_overlap_pids": exact,
        "historical_normalized_overlap_count": len(normalized_historical),
        "historical_normalized_overlaps": normalized_historical,
        "external_internal_normalized_duplicate_groups": len(normalized_internal),
        "external_internal_near_duplicate_count": len(near_internal),
        "historical_near_duplicate_count": len(near_historical),
        "near_duplicate_rule": "lowercase NFKC alphanumeric token-set Jaccard >= 0.90, >=5 unique tokens",
        "external_internal_near_duplicates": near_internal,
        "historical_near_duplicates": near_historical,
        "passed_exact_historical_gate": len(exact) == 0,
    }
    write_json(RESULTS / "tables/historical_leakage_audit.json", payload)
    print(f"historical exact overlap: {len(exact)}")
    print(f"historical normalized overlap: {len(normalized_historical)}")
    print(f"external internal normalized duplicate groups: {len(normalized_internal)}")
    print(f"near duplicates: historical={len(near_historical)} internal={len(near_internal)}")
    if exact:
        raise SystemExit("STOP: exact historical prompt leakage")


if __name__ == "__main__":
    main()
