"""Budget design remains testable even though the external window stops execution."""
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CAL = json.loads((ROOT / "results/v3_4_0r/tables/budget_calibration.json").read_text())


def test_collector_and_fit_are_attacked_state_only():
    source = (ROOT / "scripts/v3_4_0r/fit_attacked_state_budgets.py").read_text()
    assert "ATTACKED states" in source
    assert "attack_w" in source and "attack_v" in source
    assert "labels" not in source.lower() or '"labels_consulted": False' in source


def test_historical_q03_row_was_mathematically_attainable_under_cap():
    """The prior .025 choice came only from an undeclared clipping criterion."""
    row = CAL["feasibility"]["0.03"]
    assert row["all_attainable"] is True
    assert row["all_within_1pct"] is True
    for arm in row["per_arm"].values():
        assert arm["q_rms"] == pytest.approx(0.03, rel=0.01)
        assert arm["q_max"] <= 0.05 + 1e-12


def test_current_protocol_fixes_target_instead_of_searching_it():
    cfg = json.loads((ROOT / "configs/v3_4_0r/controller.json").read_text())
    budget = cfg["NEW_IN_V340R"]["budget_calibration"]
    assert budget["q_target_rms"] == 0.03
    assert "target_grid" not in budget
    assert "not a selection criterion" in budget["selection_rule"]


def test_budget_fit_is_blocked_by_external_window_gate():
    window = json.loads((ROOT / "results/v3_4_0r/tables/external_window.json").read_text())
    assert window["verdict"] == "ST3_WINDOW_SHIFT"
    assert window["stop_controller_final_testing"] is True


def test_bisection_enforces_the_cap():
    import sys
    sys.path.insert(0, str(ROOT / "scripts/v3_4_0r"))
    from fit_attacked_state_budgets import solve_eta

    actions = np.array([0.1, 0.5, 1.0])
    eta = solve_eta(actions, 0.03, 0.05)
    q = np.minimum(eta * actions, 0.05)
    assert np.sqrt(np.mean(q ** 2)) == pytest.approx(0.03)
    assert q.max() <= 0.05
