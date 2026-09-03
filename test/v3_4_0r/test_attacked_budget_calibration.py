import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CAL = json.loads((ROOT / "results/v3_4_0r/tables/budget_calibration.json").read_text())


def test_calibration_uses_attacked_states():
    assert "ATTACKED" in CAL["distribution"]
    assert CAL["split"] == "D_budget_attacked_r"


def test_calibration_consulted_no_output_or_label():
    assert CAL["generation_performed"] is False
    assert CAL["labels_consulted"] is False


def test_sample_is_large_enough():
    assert CAL["n_prompts"] >= 200
    assert CAL["n_states_per_arm"] >= 200 * 5


def test_every_arm_attains_the_common_target():
    chosen = str(CAL["q_target_selected"])
    for arm, row in CAL["feasibility"][chosen]["per_arm"].items():
        assert row["attainable"]
        assert abs(row["q_rms"] / float(chosen) - 1) <= 0.01, arm
        assert row["q_max"] <= CAL["q_cap"] + 1e-12, arm


def test_selection_rule_is_outcome_independent():
    rule = CAL["selection_rule"].lower()
    assert "largest" in rule and "clip" in rule
    for word in ("auc", "refusal", "safe", "winner", "best rho"):
        assert word not in rule


def test_target_is_the_largest_feasible_candidate():
    assert CAL["q_target_selected"] == max(CAL["feasible_targets"])


def test_infeasible_targets_were_rejected_for_clipping_not_outcome():
    for target, row in CAL["feasibility"].items():
        if not row["feasible"]:
            assert not (row["all_within_1pct"] and row["clip_rate_ok"])


def test_linear_baseline_shares_the_same_budget():
    chosen = str(CAL["q_target_selected"])
    arms = CAL["feasibility"][chosen]["per_arm"]
    assert "LINEAR" in arms
    assert arms["LINEAR"]["q_rms"] == pytest.approx(float(chosen), rel=0.01)
