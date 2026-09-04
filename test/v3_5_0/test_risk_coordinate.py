import numpy as np

from cantor_guard_v340.actuator import Actuator
from cantor_guard_v340.sensor_distance import SensorHyperplane
from cantor_guard_v350.conformal_window import calibrate_upper_window, conformal_order_index
from cantor_guard_v350.linear_risk_controller import LinearRiskController
from cantor_guard_v350.one_sided_cantor import epsilon_r, margin_m3
from cantor_guard_v350.risk_cantor_controller import RiskCantorController
from cantor_guard_v350.risk_coordinate import lipschitz_slack, risk_magnitude, risk_ratio


def objects():
    sensor = SensorHyperplane(np.array([1.0, 0.0]), 0.0)
    actuator = Actuator(np.array([1.0, 0.0]), 1)
    return sensor, actuator


def test_one_sided_transform_and_safe_collapse():
    d = np.array([-3.0, -0.5, 0.0, 2.0, 1000.0])
    x = risk_magnitude(d)
    assert np.array_equal(x, [3.0, 0.5, 0.0, 0.0, 0.0])
    assert np.all(x >= 0)


def test_risk_transform_is_one_lipschitz():
    sensor, _ = objects()
    rng = np.random.default_rng(350)
    h = rng.normal(size=(1000, 2))
    delta = rng.normal(size=(1000, 2))
    assert lipschitz_slack(sensor, h, delta).min() >= -1e-12


def test_conformal_index_and_order_statistic():
    x = np.arange(1, 301, dtype=float)
    got = calibrate_upper_window(x, 0.05)
    assert conformal_order_index(300, 0.05) == 286
    assert got.order_index_one_based == 286
    assert got.W_R == 286.0
    assert got.empirical_coverage == 286 / 300


def test_ratio_is_not_clipped():
    r = risk_ratio(np.array([0.0, 1.0, 2.0]), 1.0)
    assert np.array_equal(r[:2], [0.0, 1.0])
    assert np.isnan(r[2])


def test_certificate_formula_and_unique_continuous_max():
    assert np.isclose(epsilon_r(1 / 3, 2.7), 0.1)
    assert np.isclose(margin_m3(1 / 3), 1 / 27)
    assert margin_m3(1 / 3) > margin_m3(0.30)
    assert margin_m3(1 / 3) > margin_m3(0.36)


def test_safe_action_zero_and_outside_max():
    sensor, actuator = objects()
    ctrl = RiskCantorController(sensor=sensor, actuator=actuator, W_R=1.0,
                                rho=1 / 3, eta=0.1, q_cap=0.05)
    got = ctrl.correct(np.array([[2.0, 1.0], [-2.0, 1.0]]))
    assert got.x_risk[0] == 0.0
    assert got.q_ctrl[0] == 0.0
    assert got.cell_kind[1] == "outside"
    assert got.actions[1] == 1.0
    assert got.q_ctrl[1] == 0.05


def test_linear_has_same_safe_and_outside_policy():
    sensor, actuator = objects()
    ctrl = LinearRiskController(sensor=sensor, actuator=actuator, W_R=1.0,
                                eta=0.1, q_cap=0.05)
    got = ctrl.correct(np.array([[2.0, 1.0], [-2.0, 1.0]]))
    assert got.q_ctrl[0] == 0.0
    assert got.q_ctrl[1] == 0.05
