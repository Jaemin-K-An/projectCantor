"""Select the first 300 unscanned candidates for D_budget_v351."""
from __future__ import annotations

import hashlib
import json
import pathlib

import pandas as pd

from _common import CONFIG, RESULTS, ensure_final_outputs_absent, read_json, sha256, write_json

N_BUDGET = 300


def list_hash(values):
    return hashlib.sha256(json.dumps(list(values), separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    ensure_final_outputs_absent()
    if not (CONFIG / "attack_grid.json").exists():
        raise SystemExit("attack grid must be frozen before D_budget_v351")
    splits = read_json(CONFIG / "splits.json")
    scanned = set(splits.get("risk_cal_scanned_pids") or [])
    if not scanned:
        raise SystemExit("risk-calibration scan registry is missing")
    pool = pd.read_csv(RESULTS / "cache/D_risk_cal_candidate_order_v351.csv")
    budget = pool[~pool.pid.astype(str).isin(scanned)].sort_values("candidate_order", kind="stable").head(N_BUDGET).copy()
    if len(budget) != N_BUDGET:
        raise SystemExit(f"D_budget_v351 insufficient: {len(budget)} < {N_BUDGET}")
    harmful_final = set(pd.read_csv(RESULTS / "cache/D_final_v351_harmful.csv").pid.astype(str))
    benign_final = set(pd.read_csv(RESULTS / "cache/D_final_v351_benign.csv").pid.astype(str))
    budget_ids = budget.pid.astype(str).tolist()
    overlaps = {
        "risk_scanned": len(set(budget_ids) & scanned),
        "harmful_final": len(set(budget_ids) & harmful_final),
        "benign_final": len(set(budget_ids) & benign_final),
    }
    if any(overlaps.values()):
        raise SystemExit(f"D_budget_v351 overlap: {overlaps}")
    target = RESULTS / "cache/D_budget_v351.csv"
    budget.to_csv(target, index=False)
    splits["budget_pids"] = budget_ids
    splits["budget_pid_order_sha256"] = list_hash(budget_ids)
    splits["budget_csv_sha256"] = sha256(target)
    splits["budget_overlap_audit"] = overlaps
    write_json(CONFIG / "splits.json", splits)
    print(f"D_budget_v351 n={len(budget)} overlap={overlaps}")


if __name__ == "__main__":
    main()
