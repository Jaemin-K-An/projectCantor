"""Exact, normalized and high token-Jaccard leakage audit for V3.5.0."""
from __future__ import annotations

import re
import sys

import pandas as pd

from _common import CONFIG, RESULTS, read_json, write_json
from build_v350_splits import historical_prompt_registry


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def tokens(text: str):
    return frozenset(normalize(text).split())


def main() -> None:
    splits = read_json(CONFIG / "splits.json")["sizes"]
    current = []
    for split in splits:
        frame = pd.read_csv(RESULTS / f"cache/{split}.csv")
        current.extend((split, str(pid), str(prompt)) for pid, prompt in zip(frame.pid, frame.prompt))
    used, historical_texts = historical_prompt_registry()
    current_ids = {pid for _, pid, _ in current}
    exact = sorted(current_ids & used)
    current_norm = [(split, pid, normalize(prompt)) for split, pid, prompt in current]
    historical_norm = {normalize(text) for text in historical_texts.values()}
    normalized = sorted(pid for _, pid, text in current_norm if text in historical_norm)
    seen, internal_norm = {}, []
    for split, pid, text in current_norm:
        if text in seen: internal_norm.append((seen[text], pid))
        seen[text] = pid

    hist_sets = [(pid, tokens(text)) for pid, text in historical_texts.items()]
    near_hist = []
    for split, pid, text in current:
        a = tokens(text)
        if not a: continue
        for old_pid, b in hist_sets:
            if not b or min(len(a), len(b)) / max(len(a), len(b)) < 0.9: continue
            if len(a & b) / len(a | b) >= 0.9:
                near_hist.append((pid, old_pid)); break
    near_internal = []
    tokenized = [(pid, tokens(text)) for _, pid, text in current]
    for i, (pid, a) in enumerate(tokenized):
        for other, b in tokenized[:i]:
            if not a or not b or min(len(a), len(b)) / max(len(a), len(b)) < 0.9: continue
            if len(a & b) / len(a | b) >= 0.9: near_internal.append((other, pid))
    passed = not (exact or normalized or internal_norm or near_hist or near_internal)
    payload = {
        "n_selected": len(current), "n_unique_selected": len(current_ids),
        "historical_registry_size": len(used), "historical_text_registry_size": len(historical_texts),
        "historical_exact_overlap": len(exact), "historical_normalized_overlap": len(normalized),
        "historical_token_jaccard_ge_0_9": len(near_hist),
        "internal_normalized_duplicate_pairs": len(internal_norm),
        "internal_token_jaccard_ge_0_9": len(near_internal),
        "passed": passed, "verdict": "LEAK1_PASS" if passed else "LEAK2_FAIL",
    }
    write_json(RESULTS / "tables/historical_leakage_audit.json", payload)
    print(payload)
    if not passed: raise SystemExit("dataset leakage hard stop")


if __name__ == "__main__":
    main()
