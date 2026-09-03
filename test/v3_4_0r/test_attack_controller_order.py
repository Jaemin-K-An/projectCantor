import numpy as np
import pytest

from cantor_guard_v340.attack import attack_v, attack_w
from cantor_guard_v340r.controllers import CappedCantorController


def build(sensor, actuator, freeze, eta=0.04):
    return CappedCantorController(sensor=sensor, actuator=actuator,
                                  W=float(freeze["inherited_frozen"]["W"]), rho=1 / 3,
                                  eta=eta, q_cap=float(freeze["hard_q_cap"]["q_cap"]),
                                  leaf_actions=freeze["inherited_frozen"]["leaf_actions"])


def test_attack_w_moves_the_sensor_exactly(sensor):
    for eps in (0.1014, 1.8413, 7.3821):
        assert sensor.delta_distance(attack_w(sensor, eps, sign=-1)) == pytest.approx(-eps)


def test_attack_v_moves_by_epsilon_times_kappa(sensor, actuator):
    kappa = sensor.coupling(actuator.v_safe)
    assert sensor.delta_distance(attack_v(actuator, 2.0)) == pytest.approx(-2.0 * kappa)


def test_controller_sees_the_attacked_state(sensor, actuator, freeze, rng):
    c = build(sensor, actuator, freeze)
    h = rng.normal(size=(30, sensor.w.size)) * 6
    attacked = h + attack_w(sensor, 1.2, sign=-1)[None, :]
    res = c.correct(attacked)
    assert np.allclose(res.d_observed, np.atleast_1d(sensor.distance(attacked)))
    assert not np.allclose(res.d_observed, np.atleast_1d(sensor.distance(h)))


def test_correction_never_pushes_further_toward_risk(sensor, actuator, freeze, rng):
    c = build(sensor, actuator, freeze)
    h = rng.normal(size=(60, sensor.w.size)) * 6
    attacked = h + attack_w(sensor, 1.0, sign=-1)[None, :]
    res = c.correct(attacked)
    moved = np.atleast_1d(sensor.distance(res.h_corrected)) - np.atleast_1d(res.d_observed)
    assert np.all(moved >= -1e-12)


def test_realised_shift_matches_the_analytic_prediction(sensor, actuator, freeze, rng):
    c = build(sensor, actuator, freeze)
    h = rng.normal(size=(40, sensor.w.size)) * 5
    res = c.correct(h)
    realised = np.atleast_1d(sensor.distance(res.h_corrected)) - np.atleast_1d(res.d_observed)
    assert np.allclose(realised, res.delta_d_expected, atol=1e-9)


def test_hook_touches_only_the_prefill_forward():
    import inspect

    from cantor_guard_v340 import p0_generation

    src = inspect.getsource(p0_generation.p0_attack_then_control)
    assert 'if tr["forward"] == 0:' in src
    assert "no P0 intervention leaks into G1+" in inspect.getdoc(p0_generation.p0_attack_then_control)
