"""Repair a post-confirmatory diagnostic after checkpoint row reordering.

This is an integrity-only amendment: the affected statistic is exploratory and
is not consumed by any confirmatory gate.  Preserve the original output, join
the matched arms by their experimental keys, and extend the invalidation ledger.
"""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from _common import RESULTS, read_json, require_freeze, sha256, write_json


def main() -> None:
    require_freeze()
    diagnostics_path = RESULTS / "tables/diagnostics.json"
    invalidated_path = RESULTS / "tables/INVALIDATED_row_order_diagnostics.json"
    verdict_path = RESULTS / "tables/final_verdict.json"
    amendment_path = RESULTS / "tables/postfreeze_integrity_amendment.json"
    if invalidated_path.exists() or amendment_path.exists():
        raise SystemExit("post-freeze diagnostic amendment already applied")

    frame = pd.read_csv(
        RESULTS / "raw/final_D_final_v351_harmful.csv", low_memory=False)
    risk = frame[frame.risk_eligible.astype(bool)]
    keys = ["pid", "family", "epsilon"]
    linear = risk[risk.arm == "LINEAR"][keys + ["action"]].rename(
        columns={"action": "linear_action"})
    cantor = risk[risk.arm == "CANTOR_1/3"][keys + ["action"]].rename(
        columns={"action": "cantor_action"})
    matched = linear.merge(cantor, on=keys, validate="one_to_one")
    if len(matched) != len(linear) or len(matched) != len(cantor):
        raise SystemExit("matched diagnostic arms are incomplete")
    corrected = float(np.mean(np.abs(matched.linear_action - matched.cantor_action)))

    shutil.copy2(diagnostics_path, invalidated_path)
    before_hash = sha256(invalidated_path)
    diagnostics = read_json(diagnostics_path)
    reported = diagnostics["matched_mapping_difference"][
        "risk_state_mean_abs_action_difference_linear_vs_cantor_1_3"]
    diagnostics["matched_mapping_difference"][
        "risk_state_mean_abs_action_difference_linear_vs_cantor_1_3"] = corrected
    diagnostics["matched_mapping_difference"].update({
        "matching_keys": keys,
        "matched_risk_states_per_arm": len(matched),
        "integrity_amendment": "post-freeze row-order repair; exploratory only; no retuning",
    })
    write_json(diagnostics_path, diagnostics)

    verdict = read_json(verdict_path)
    invalidated = list(verdict.get("invalidated_runs", []))
    for name in (
        "INVALIDATED_mixed_backend_final_harmful_generation.json",
        "INVALIDATED_mixed_backend_final_budget_audit.json",
        "INVALIDATED_row_order_diagnostics.json",
    ):
        if name not in invalidated:
            invalidated.append(name)
    verdict["invalidated_runs"] = invalidated
    verdict["integrity_amendments"] = [
        "mixed-backend checkpoint regenerated before external endpoint scoring",
        "exploratory matched-action diagnostic corrected by keyed join",
    ]
    write_json(verdict_path, verdict)

    write_json(amendment_path, {
        "status": "POSTFREEZE_INTEGRITY_AMENDMENT_COMPLETE",
        "confirmatory_outputs_or_verdicts_changed": False,
        "settings_prompts_grid_arms_eta_changed": False,
        "external_labels_used_for_retuning": False,
        "affected_output": "exploratory diagnostics only",
        "cause": "checkpoint regeneration changed CSV row order; original diagnostic subtracted unkeyed arm arrays",
        "original_reported_mean_abs_action_difference": reported,
        "corrected_keyed_mean_abs_action_difference": corrected,
        "matching_keys": keys,
        "matched_risk_states_per_arm": len(matched),
        "invalidated_diagnostics_sha256": before_hash,
        "corrected_diagnostics_sha256": sha256(diagnostics_path),
        "overall_verdict": verdict["OVERALL"],
    })
    print(f"corrected exploratory matched-action diagnostic: {reported} -> {corrected}")


if __name__ == "__main__":
    main()
