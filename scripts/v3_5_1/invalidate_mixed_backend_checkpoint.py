"""Invalidate CPU-origin final blocks before frozen external endpoint scoring.

The long CPU run was checkpointed and resumed on MPS for throughput.  This
left one nonzero ATTACK_V condition split across compute backends.  Although
its binary risk masks agreed exactly, pre-control distances differed by up to
3.36e-5.  Remove every CPU-origin block, preserve it as INVALIDATED, and let
the already-frozen generator refill the exact same cells on MPS.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from _common import RESULTS, read_json, sha256, write_json


def main() -> None:
    raw_path = RESULTS / "raw/final_D_final_v351_harmful.csv"
    private_path = RESULTS / "private/final_D_final_v351_harmful_completions.csv"
    raw, private = pd.read_csv(raw_path, low_memory=False), pd.read_csv(private_path, low_memory=False)
    if "y_safe" in raw or "action_class" in raw:
        raise SystemExit("external refusal endpoint already scored; backend correction forbidden")
    grid = read_json(RESULTS.parent.parent / "configs/v3_5_1/attack_grid.json")["attack_grid"]
    epsilon = float(grid[1])
    first_two = ["ATTACK_ONLY", "LINEAR"]
    raw_eps1 = np.isclose(raw.epsilon.to_numpy(), epsilon, rtol=0.0, atol=1e-15)
    private_eps1 = np.isclose(private.epsilon.to_numpy(), epsilon, rtol=0.0, atol=1e-15)
    affected_raw = ((raw.family == "ATTACK_V") & (raw.epsilon == 0.0)) | \
                   ((raw.family == "ATTACK_W") & (raw.epsilon == 0.0)) | \
                   ((raw.family == "ATTACK_V") & raw_eps1 & raw.arm.isin(first_two))
    affected_private = ((private.family == "ATTACK_V") & (private.epsilon == 0.0)) | \
                       ((private.family == "ATTACK_W") & (private.epsilon == 0.0)) | \
                       ((private.family == "ATTACK_V") & private_eps1 & private.arm.isin(first_two))
    if int(affected_raw.sum()) != 4000 or int(affected_private.sum()) != 4000:
        raise SystemExit(f"unexpected mixed-backend row count: raw={affected_raw.sum()} private={affected_private.sum()}")
    invalid_raw = RESULTS / "raw/INVALIDATED_cpu_backend_final_conditions.csv"
    invalid_private = RESULTS / "private/INVALIDATED_cpu_backend_final_conditions_completions.csv"
    if invalid_raw.exists() or invalid_private.exists():
        raise SystemExit("mixed-backend checkpoint was already invalidated")
    raw.loc[affected_raw].to_csv(invalid_raw, index=False)
    private.loc[affected_private].to_csv(invalid_private, index=False)
    raw.loc[~affected_raw].to_csv(raw_path, index=False)
    private.loc[~affected_private].to_csv(private_path, index=False)

    generation_marker = RESULTS / "tables/final_harmful_generation.json"
    invalid_marker = RESULTS / "tables/INVALIDATED_mixed_backend_final_harmful_generation.json"
    budget_audit = RESULTS / "tables/final_budget_audit.json"
    invalid_budget = RESULTS / "tables/INVALIDATED_mixed_backend_final_budget_audit.json"
    generation_marker.rename(invalid_marker)
    budget_audit.rename(invalid_budget)
    write_json(RESULTS / "tables/mixed_backend_checkpoint_invalidation.json", {
        "status": "INVALIDATED_MIXED_COMPUTE_BACKEND_BEFORE_EXTERNAL_ENDPOINT",
        "reason": "CPU-to-MPS resume split ATTACK_V epsilon=0.5*epsilon_R_C across backends; binary eligibility agreed but pre-control d differed up to 3.36e-5",
        "settings_changed": False,
        "prompts_grid_arms_eta_changed": False,
        "external_refusal_labels_seen": False,
        "lexical_refusal_proxy_was_computed_during_generation": True,
        "invalidated_rows": 4000,
        "conditions": [
            "ATTACK_V epsilon=0 all arms (CPU)",
            "ATTACK_W epsilon=0 all arms (copied from CPU ATTACK_V zero)",
            f"ATTACK_V epsilon={epsilon} ATTACK_ONLY and LINEAR (CPU)",
        ],
        "retained_valid_rows": int((~affected_raw).sum()),
        "required_remediation": "regenerate the exact invalidated cells on MPS under the unchanged PRE_ANALYSIS_FREEZE",
        "invalidated_raw_sha256": sha256(invalid_raw),
        "invalidated_generation_marker": str(invalid_marker.relative_to(RESULTS.parent.parent)),
        "invalidated_budget_audit": str(invalid_budget.relative_to(RESULTS.parent.parent)),
    })
    print(f"invalidated 4000 mixed-backend rows; retained {(~affected_raw).sum()} rows")


if __name__ == "__main__":
    main()
