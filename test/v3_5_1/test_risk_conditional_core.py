import inspect
import json

import numpy as np
import pytest

from cantor_guard_v340.actuator import Actuator
from cantor_guard_v340.sensor_distance import SensorHyperplane
from cantor_guard_v351.conformal_window import calibrate_conditional_risk_window
from cantor_guard_v351.risk_budget import fit_eta_risk_conditional, risk_eligibility
from cantor_guard_v351.risk_cantor_controller import POSITIVE_LEAF_ACTIONS, RiskConditionalCantorController
from cantor_guard_v351.one_sided_cantor import epsilon_r, margin_derivative_m3, margin_m3, unique_grid_max
from scripts.v3_5_1 import _common


def objects():
    return SensorHyperplane(np.array([1.,0.]),0.), Actuator(np.array([1.,0.]),1)


def test_conditional_window_excludes_safe_zero_mass():
    d = np.r_[np.full(900, 2.), -np.arange(1, 201)]
    got = calibrate_conditional_risk_window(d, n_risk=200, alpha=.05)
    assert got.order_index_one_based == 191
    assert got.W_R == 191


def test_positive_monotone_leaf_schedule():
    assert POSITIVE_LEAF_ACTIONS == tuple((i+1)/8 for i in range(8))
    assert min(POSITIVE_LEAF_ACTIONS) > 0
    assert np.all(np.diff(POSITIVE_LEAF_ACTIONS) > 0)


def test_safe_zero_risk_positive_outside_max_and_cap():
    sensor, actuator = objects()
    ctrl = RiskConditionalCantorController(sensor=sensor, actuator=actuator, W_R=1,
            rho=1/3, eta=.1, q_cap=.05)
    got = ctrl.correct(np.array([[2.,1.],[-.01,1.],[-2.,1.]]))
    assert got.q_ctrl[0] == 0
    assert got.actions[1] > 0
    assert got.actions[2] == 1 and got.q_ctrl[2] == .05
    assert np.max(got.q_ctrl) <= .05


def test_guard_action_uses_more_conservative_adjacent_leaf():
    sensor, actuator = objects()
    ctrl = RiskConditionalCantorController(sensor=sensor, actuator=actuator, W_R=1,
            rho=1/3, eta=.01, q_cap=.05)
    guard = ctrl.guards[0]
    r = (guard.lo + guard.hi) / 2
    result = ctrl.correct(np.array([[-r, 1.0]]))
    assert result.cell_kind[0] == "guard"
    assert result.actions[0] == ctrl._guard_action(guard)
    left = max((leaf for leaf in ctrl.leaves if leaf.hi <= guard.lo + 1e-15), key=lambda c: c.hi)
    right = min((leaf for leaf in ctrl.leaves if leaf.lo >= guard.hi - 1e-15), key=lambda c: c.lo)
    assert result.actions[0] == max(ctrl.leaf_actions[left.index], ctrl.leaf_actions[right.index])


def test_risk_mask_is_arm_independent_and_precontrol_only():
    d = np.array([1.,-.1,-2.,0.])
    expected = np.array([False,True,True,False])
    for _arm in range(9): assert np.array_equal(risk_eligibility(d), expected)
    assert list(inspect.signature(risk_eligibility).parameters) == ["d_attacked"]


def test_conditional_solver_does_not_match_global_rms():
    mask = np.array([False]*90 + [True]*10)
    actions = np.where(mask, .5, 0.)
    eta, metrics = fit_eta_risk_conditional(actions, mask, target=.03, q_cap=.05)
    assert np.isclose(metrics["risk_q_rms"], .03)
    assert np.isclose(metrics["global_q_rms"], .03*np.sqrt(.1))
    assert metrics["safe_side_intervention_frequency"] == 0
    assert eta > 0


def test_budget_rejects_arm_specific_or_safe_side_actions():
    mask = np.array([False, True])
    with pytest.raises(ValueError, match="safe-side"):
        fit_eta_risk_conditional(np.array([.01, .5]), mask)
    with pytest.raises(ValueError, match="strictly positive"):
        fit_eta_risk_conditional(np.array([0., 0.]), mask)


def test_window_api_cannot_accept_outputs_or_labels():
    parameters = set(inspect.signature(calibrate_conditional_risk_window).parameters)
    assert parameters == {"d", "n_risk", "alpha"}
    with pytest.raises(TypeError):
        calibrate_conditional_risk_window([-1] * 200, labels=[1] * 200)


def test_middle_third_is_unique_maximum_and_certificate_matches_formula():
    rhos = (.25, .28, .30, 1/3, .36, .40, .44)
    assert unique_grid_max(rhos) == 1/3
    assert margin_derivative_m3(1/3 - 1e-6) > 0
    assert margin_derivative_m3(1/3 + 1e-6) < 0
    assert np.isclose(margin_m3(1/3), 1/27)
    assert np.isclose(epsilon_r(1/3, 2.7), 2.7/27)


def test_final_access_fails_closed_before_freeze(tmp_path, monkeypatch):
    config = tmp_path / "config"
    config.mkdir()
    (config / "PRE_ANALYSIS_FREEZE.json").write_text(json.dumps({"status": "NOT_FROZEN"}))
    monkeypatch.setattr(_common, "CONFIG", config)
    with pytest.raises(RuntimeError, match="requires PRE_ANALYSIS_FROZEN"):
        _common.require_freeze()


def test_frozen_sensor_and_actuator_hashes_unchanged():
    assert _common.sha256(_common.ROOT / "results/v3_4_0/cache/sensor_w.npy") == _common.SENSOR_SHA
    assert _common.sha256(_common.ROOT / "results/v3_3_5a/cache/v_p0.npy") == _common.ACTUATOR_SHA
