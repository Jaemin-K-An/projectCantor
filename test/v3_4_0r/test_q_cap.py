import numpy as np
import pytest

from cantor_guard_v340r.controllers import CappedCantorController, LinearThresholdController


def build(sensor, actuator, freeze, kind="cantor", eta=5.0):
    W = float(freeze["inherited_frozen"]["W"])
    cap = float(freeze["hard_q_cap"]["q_cap"])
    if kind == "cantor":
        return CappedCantorController(sensor=sensor, actuator=actuator, W=W, rho=1 / 3,
                                      eta=eta, q_cap=cap,
                                      leaf_actions=freeze["inherited_frozen"]["leaf_actions"])
    return LinearThresholdController(sensor=sensor, actuator=actuator, W=W, eta=eta, q_cap=cap)


@pytest.mark.parametrize("kind", ["cantor", "linear"])
@pytest.mark.parametrize("eta", [0.01, 0.5, 50.0])
def test_cap_holds_for_every_state(sensor, actuator, freeze, rng, kind, eta):
    """V3.4.0 declared q_cap=0.05 and recorded q_max up to 0.0554."""
    cap = float(freeze["hard_q_cap"]["q_cap"])
    c = build(sensor, actuator, freeze, kind, eta)
    h = rng.normal(size=(400, sensor.w.size)) * rng.uniform(0.5, 40, size=(400, 1))
    res = c.correct(h)
    assert res.q_ctrl.max() <= cap + 1e-12
    assert np.all(res.q_ctrl <= res.q_raw + 1e-12)


def test_realised_norm_equals_q_ctrl(sensor, actuator, freeze, rng):
    c = build(sensor, actuator, freeze, "cantor", 2.0)
    h = rng.normal(size=(50, sensor.w.size)) * 6
    res = c.correct(h)
    realised = np.linalg.norm(res.delta_h, axis=1) / np.linalg.norm(h, axis=1)
    assert np.allclose(realised, res.q_ctrl)


def test_clipping_flag_is_accurate(sensor, actuator, freeze, rng):
    cap = float(freeze["hard_q_cap"]["q_cap"])
    c = build(sensor, actuator, freeze, "cantor", 1.0)
    h = rng.normal(size=(200, sensor.w.size)) * 8
    res = c.correct(h)
    assert np.array_equal(res.clipped, res.q_raw > cap)
    assert np.allclose(res.q_ctrl[res.clipped], cap)


def test_uncapped_v340_controller_would_have_exceeded_it(sensor, actuator, freeze, rng):
    """Demonstrates the defect the cap fixes."""
    from cantor_guard_v340.sensor_actuator_controller import SensorActuatorCantorController

    cap = float(freeze["hard_q_cap"]["q_cap"])
    old = SensorActuatorCantorController(sensor=sensor, actuator=actuator,
                                         W=float(freeze["inherited_frozen"]["W"]),
                                         rho=1 / 3, eta=0.2)
    h = rng.normal(size=(100, sensor.w.size)) * 6
    assert old.correct(h).q_ctrl.max() > cap


def test_rejects_bad_cap(sensor, actuator, freeze):
    W = float(freeze["inherited_frozen"]["W"])
    for bad in (0.0, -0.1, float("nan")):
        with pytest.raises(ValueError):
            CappedCantorController(sensor=sensor, actuator=actuator, W=W, rho=1 / 3,
                                   eta=0.03, q_cap=bad)
