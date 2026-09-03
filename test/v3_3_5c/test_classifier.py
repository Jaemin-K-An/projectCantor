import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path("scripts/v3_3_5c").resolve()))
from freeze_p0_dose_grid import select_confirm_grid
from final_claim_check_v335c import overall_verdict
from freeze_v335c import verify_freeze


def test_confirm_grid_is_symmetric_contiguous_and_nondegenerate():
    rows = []
    for u in (-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8):
        for i in range(20):
            bad_outer = abs(u) >= 0.8
            rows.append({
                "u": u,
                "coherence": 0.8 if bad_outer else 1.0,
                "degeneration": int(bad_outer),
                "refusal_proxy": int(u + i / 100 > 0),
                "relative_norm_realised": abs(u),
            })
    protocol = {
        "coherence_gate": 0.95,
        "degeneration_rate_gate": 0.05,
        "candidate_u_grid": [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8],
    }
    result = select_confirm_grid(pd.DataFrame(rows), protocol)
    assert result["confirm_u_grid"] == [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
    assert result["status"] == "READY_FOR_CONFIRM"


def test_no_extreme_absolute_lambda_in_behavioral_protocol():
    text = pathlib.Path("configs/v3_3_5c/behavioral_protocol.json").read_text()
    assert '"candidate_u_grid"' in text
    assert "lambda=-100" not in text and '"doses"' not in text


def test_overall_classifier_requires_every_A_gate():
    common = dict(
        behavioral="B1_P0_BEHAVIORAL_BOUNDARY_IDENTIFIED",
        certificate="C1_CANTOR_P0_CERTIFICATE_VALID",
        budgets="BUDGET_MATCHED", utility="U1_PASS",
        generation="G4_PROXY_ONLY", evaluator="S2_PROXY_ONLY", final_ran=True,
    )
    assert overall_verdict(**common) == "A_CANTOR_BEHAVIORALLY_ANCHORED_LLM_CONTROLLER_SUPPORTED"
    for key, bad in (
        ("certificate", "C2_IMPLEMENTATION_FAILURE"),
        ("budgets", "BUDGET_MISMATCH"),
        ("utility", "U2_FAIL"),
        ("final_ran", False),
    ):
        changed = dict(common); changed[key] = bad
        assert overall_verdict(**changed) == "F_INCONCLUSIVE"


def test_failed_fresh_boundary_forces_E():
    assert overall_verdict(
        behavioral="B3_BOUNDARY_UNIDENTIFIABLE",
        certificate="C3_WINDOW_APPLICABILITY_FAILURE",
        budgets="NOT_RUN", utility="U3_NOT_RUN",
        generation="G6_NOT_RUN_BEHAVIORAL_GATE",
        evaluator="S2_PROXY_ONLY", final_ran=False,
    ) == "E_P0_BEHAVIORAL_ANCHOR_NOT_REPLICATED"


def test_final_stage_rejects_nonfrozen_manifest_before_hash_access():
    with pytest.raises(RuntimeError, match="requires status=PRE_ANALYSIS_FROZEN"):
        verify_freeze({"status": "NOT_FROZEN_BEHAVIORAL_GATE_FAILED"})
