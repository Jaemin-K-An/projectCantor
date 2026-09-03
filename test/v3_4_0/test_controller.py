import json
import pathlib

import numpy as np
import pytest

from cantor_guard_v340.sensor_actuator_controller import SensorActuatorCantorController

ROOT = pathlib.Path(__file__).resolve().parents[2]


def build(sensor, actuator, freeze, rho, eta=0.03):
    return SensorActuatorCantorController(
        sensor=sensor, actuator=actuator, W=float(freeze["geometry"]["W"]),
        rho=rho, eta=eta, leaf_actions=freeze["geometry"]["leaf_actions"])


def test_controller_consumes_actual_residual_vectors(sensor, actuator, freeze, rng):
    c = build(sensor, actuator, freeze, 1 / 3)
    h = rng.normal(size=(6, sensor.w.size)) * 3
    result = c.correct(h)
    assert result.h_corrected.shape == h.shape
    # a precomputed margin must not be accepted in place of the state
    with pytest.raises(ValueError):
        c.correct(np.atleast_2d(np.atleast_1d(sensor.distance(h))))


def test_r_of_zero_distance_is_one_half(sensor, actuator, freeze):
    for rho in freeze["geometry"]["rho_family"]:
        c = build(sensor, actuator, freeze, float(rho))
        assert c.risk_coordinate(0.0)[0] == pytest.approx(0.5)


def test_higher_risk_means_higher_r(sensor, actuator, freeze):
    c = build(sensor, actuator, freeze, 1 / 3)
    # d>0 is the safe side, so r must decrease as d increases
    assert c.risk_coordinate(1.0)[0] < c.risk_coordinate(-1.0)[0]


def test_depth_three_structure_for_every_rho(sensor, actuator, freeze):
    for rho in freeze["geometry"]["rho_family"]:
        c = build(sensor, actuator, freeze, float(rho))
        assert len(c.leaves) == 8 and len(c.guards) == 7


def test_central_guard_action_is_conservative(sensor, actuator, freeze):
    for rho in freeze["geometry"]["rho_family"]:
        c = build(sensor, actuator, freeze, float(rho))
        centre = c.correct(np.atleast_2d(sensor.project_to_hyperplane(np.ones(sensor.w.size))))
        assert centre.cells[0].kind == "guard"
        # conservative == at least the median leaf action
        assert centre.actions[0] >= float(np.median(c.leaf_actions))


def test_outside_window_falls_back_and_is_never_clipped(sensor, actuator, freeze, rng):
    c = build(sensor, actuator, freeze, 1 / 3)
    W = float(freeze["geometry"]["W"])
    far = sensor.project_to_hyperplane(rng.normal(size=sensor.w.size)) - 50 * W * sensor.w_hat
    rec = c.policy_record(np.atleast_2d(far))[0]
    assert rec["outside_window"] and rec["status"] == "OUTSIDE_WINDOW"
    assert rec["r"] is None                      # NaN, not clipped to 0 or 1
    assert rec["action"] == pytest.approx(1.0)   # conservative fallback


def test_same_sensor_and_window_for_every_rho(freeze):
    geom = freeze["geometry"]
    assert geom["same_W_every_rho"] is True
    assert geom["boundary"] == "d = 0 by construction"
    ctrl = json.loads((ROOT / "configs/v3_4_0/controller.json").read_text())
    for shared in ("w", "b", "v_safe", "W", "depth", "leaf actions"):
        assert shared in ctrl["identical_across_rho"]


def test_correction_uses_the_actuator_not_the_sensor_normal(sensor, actuator, freeze, rng):
    c = build(sensor, actuator, freeze, 1 / 3)
    h = rng.normal(size=(5, sensor.w.size)) * 3
    delta = c.correct(h).delta_h
    unit = delta / np.linalg.norm(delta, axis=1, keepdims=True)
    assert np.allclose(unit, actuator.v_safe[None, :], atol=1e-9)
    assert not np.allclose(unit, sensor.w_hat[None, :], atol=1e-3)


def test_rejects_bad_configuration(sensor, actuator, freeze):
    for kwargs in ({"depth": 2}, {"eta": -1.0}, {"W": 0.0},
                   {"leaf_actions": [0, 1, 0, 1, 0, 1, 0, 1]}, {"outside_action": 2.0}):
        base = {"sensor": sensor, "actuator": actuator, "W": 2.0, "rho": 1 / 3, "eta": 0.03}
        base.update(kwargs)
        with pytest.raises(ValueError):
            SensorActuatorCantorController(**base)
