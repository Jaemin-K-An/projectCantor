import numpy as np
import pytest

from cantor_guard_v340r.controllers import CappedCantorController, LinearThresholdController


def test_linear_shares_sensor_actuator_and_budget(sensor, actuator, freeze):
    W = float(freeze["inherited_frozen"]["W"])
    cap = float(freeze["hard_q_cap"]["q_cap"])
    lin = LinearThresholdController(sensor=sensor, actuator=actuator, W=W, eta=0.04, q_cap=cap)
    assert lin.sensor is sensor and lin.actuator is actuator
    assert lin.kappa == pytest.approx(sensor.coupling(actuator.v_safe))


def test_linear_has_no_recursive_partition(sensor, actuator, freeze):
    W = float(freeze["inherited_frozen"]["W"])
    lin = LinearThresholdController(sensor=sensor, actuator=actuator, W=W, eta=0.04, q_cap=0.05)
    assert lin.rho is None
    assert not hasattr(lin, "leaves")


def test_linear_action_rises_smoothly_with_risk(sensor, actuator, freeze, rng):
    W = float(freeze["inherited_frozen"]["W"])
    lin = LinearThresholdController(sensor=sensor, actuator=actuator, W=W, eta=1.0, q_cap=1.0)
    base = sensor.project_to_hyperplane(rng.normal(size=sensor.w.size))
    states = np.stack([base + t * sensor.w_hat for t in np.linspace(-0.9 * W, 0.9 * W, 40)])
    actions = lin.correct(states).actions
    assert np.all(np.diff(actions) <= 1e-9), "action must decrease as d increases (safer)"
    assert actions.min() >= 0 and actions.max() <= 1


def test_linear_and_cantor_differ_only_by_the_partition(sensor, actuator, freeze, rng):
    W = float(freeze["inherited_frozen"]["W"])
    h = rng.normal(size=(60, sensor.w.size)) * 6
    lin = LinearThresholdController(sensor=sensor, actuator=actuator, W=W, eta=0.04, q_cap=0.05)
    cant = CappedCantorController(sensor=sensor, actuator=actuator, W=W, rho=1 / 3,
                                  eta=0.04, q_cap=0.05)
    a, b = lin.correct(h), cant.correct(h)
    assert np.allclose(a.d_observed, b.d_observed)          # same sensing
    assert not np.allclose(a.actions, b.actions)            # different policy
    unit_a = a.delta_h / np.maximum(np.linalg.norm(a.delta_h, axis=1, keepdims=True), 1e-12)
    unit_b = b.delta_h / np.maximum(np.linalg.norm(b.delta_h, axis=1, keepdims=True), 1e-12)
    nz = (np.linalg.norm(a.delta_h, axis=1) > 0) & (np.linalg.norm(b.delta_h, axis=1) > 0)
    assert np.allclose(unit_a[nz], unit_b[nz])              # same actuation direction


def test_linear_falls_back_outside_the_window(sensor, actuator, freeze, rng):
    W = float(freeze["inherited_frozen"]["W"])
    lin = LinearThresholdController(sensor=sensor, actuator=actuator, W=W, eta=1.0, q_cap=1.0)
    far = sensor.project_to_hyperplane(rng.normal(size=sensor.w.size)) - 50 * W * sensor.w_hat
    rec = lin.records(np.atleast_2d(far))[0]
    assert rec["outside_window"] and rec["action"] == pytest.approx(1.0)
