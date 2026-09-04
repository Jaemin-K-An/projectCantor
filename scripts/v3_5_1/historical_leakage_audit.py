"""Leakage audit for fresh V3.5.1 development candidates and inherited finals."""
from __future__ import annotations

import re
import pandas as pd

from _common import RESULTS, write_json
from build_candidate_pool import historical_registry


def norm(text): return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))
def toks(text): return frozenset(norm(text).split())


def main():
    pool = pd.read_csv(RESULTS / "cache/D_risk_cal_candidate_order_v351.csv")
    used, old_text = historical_registry()
    exact = set(pool.pid.astype(str)) & used
    old_norm = {norm(x) for x in old_text.values()}; normalized = [p for p in pool.prompt if norm(p) in old_norm]
    hist_sets = [(pid, toks(text)) for pid, text in old_text.items()]; near = []
    for pid, prompt in zip(pool.pid.astype(str), pool.prompt.astype(str)):
        a = toks(prompt)
        for old_pid, b in hist_sets:
            if not a or not b or min(len(a), len(b))/max(len(a), len(b)) < .9: continue
            if len(a & b)/len(a | b) >= .9: near.append((pid, old_pid)); break
    duplicates = int(pool.pid.duplicated().sum())
    passed = not exact and not normalized and not near and duplicates == 0
    payload = {"candidate_pool_n": len(pool), "historical_registry_n": len(used),
               "historical_text_registry_n": len(old_text), "exact_overlap": len(exact),
               "normalized_overlap": len(normalized), "token_jaccard_ge_0_9": len(near),
               "internal_pid_duplicates": duplicates,
               "inherited_final_overlap_authorized": True,
               "passed": passed, "verdict": "LEAK1_PASS" if passed else "LEAK2_FAIL"}
    write_json(RESULTS / "tables/historical_leakage_audit.json", payload)
    print(payload)
    if not passed: raise SystemExit("historical leakage hard stop")


if __name__ == "__main__": main()
