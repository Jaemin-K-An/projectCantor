import json
import pathlib

import numpy as np
import pytest

RESULTS = pathlib.Path(__file__).resolve().parents[2] / "results/v3_4_0"
CONFIG = pathlib.Path(__file__).resolve().parents[2] / "configs/v3_4_0"


def test_eta_was_fit_on_the_budget_split_only():
    b = json.loads((RESULTS / "tables" / "controller_budgets.json").read_text())
    assert "D_controller_budget" in b["selection_basis"] or "budget" in b["selection_basis"]
    assert "no rho outcome is consulted" in b["selection_basis"]


def test_every_rho_shares_one_target():
    b = json.loads((RESULTS / "tables" / "controller_budgets.json").read_text())
    target = b["q_target_selected"]
    for row in b["per_rho"].values():
        assert row["q_rms"] == pytest.approx(target, rel=1e-6)


def test_target_is_at_or_below_the_frozen_cap():
    b = json.loads((RESULTS / "tables" / "controller_budgets.json").read_text())
    assert b["q_target_selected"] <= b["q_cap"]
    assert b["q_target_selected"] >= b["q_needed_for_one_certificate"]


def test_budget_definition_is_relative(sensor, actuator, freeze, rng):
    from cantor_guard_v340.sensor_actuator_controller import SensorActuatorCantorController

    c = SensorActuatorCantorController(sensor=sensor, actuator=actuator,
                                       W=float(freeze["geometry"]["W"]), rho=1 / 3, eta=0.04,
                                       leaf_actions=freeze["geometry"]["leaf_actions"])
    h = rng.normal(size=(20, sensor.w.size)) * 7
    r = c.correct(h)
    realised = np.linalg.norm(r.delta_h, axis=1) / np.linalg.norm(h, axis=1)
    assert np.allclose(realised, r.q_ctrl)


def test_eta_differs_across_rho_because_action_mix_differs():
    b = json.loads((RESULTS / "tables" / "controller_budgets.json").read_text())
    etas = [row["eta"] for row in b["per_rho"].values()]
    assert len(set(round(e, 6) for e in etas)) > 1


def test_final_tolerance_is_three_percent(freeze):
    assert freeze["budget"]["tolerance"] == pytest.approx(0.03)
