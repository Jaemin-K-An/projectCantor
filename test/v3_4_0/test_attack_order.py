import numpy as np
import pytest

from cantor_guard_v340.attack import attack_grid_absolute, attack_v, attack_w, expected_delta_d
from cantor_guard_v340.sensor_actuator_controller import SensorActuatorCantorController


def test_attack_w_moves_the_sensor_by_exactly_epsilon(sensor):
    for eps in (0.05, 0.1689, 7.38):
        assert abs(expected_delta_d(sensor, attack_w(sensor, eps, sign=-1))) == pytest.approx(eps)
        assert expected_delta_d(sensor, attack_w(sensor, eps, sign=-1)) == pytest.approx(-eps)


def test_attack_v_moves_the_sensor_by_epsilon_times_kappa(sensor, actuator):
    kappa = sensor.coupling(actuator.v_safe)
    for eps in (0.05, 1.84):
        assert expected_delta_d(sensor, attack_v(actuator, eps)) == pytest.approx(-eps * kappa)


def test_the_two_families_are_not_interchangeable(sensor, actuator):
    eps = 1.0
    assert abs(expected_delta_d(sensor, attack_w(sensor, eps))) > \
           abs(expected_delta_d(sensor, attack_v(actuator, eps)))


def test_controller_observes_the_attacked_state_not_the_clean_one(sensor, actuator, freeze, rng):
    c = SensorActuatorCantorController(sensor=sensor, actuator=actuator,
                                       W=float(freeze["geometry"]["W"]), rho=1 / 3, eta=0.03,
                                       leaf_actions=freeze["geometry"]["leaf_actions"])
    h = rng.normal(size=(1, sensor.w.size)) * 4
    attacked = h + attack_w(sensor, 1.5, sign=-1)[None, :]
    assert c.correct(attacked).d_observed[0] == pytest.approx(np.atleast_1d(sensor.distance(attacked))[0])
    assert c.correct(attacked).d_observed[0] != pytest.approx(np.atleast_1d(sensor.distance(h))[0])


def test_correction_opposes_the_attack(sensor, actuator, freeze, rng):
    c = SensorActuatorCantorController(sensor=sensor, actuator=actuator,
                                       W=float(freeze["geometry"]["W"]), rho=1 / 3, eta=0.03,
                                       leaf_actions=freeze["geometry"]["leaf_actions"])
    h = rng.normal(size=(30, sensor.w.size)) * 4
    attacked = h + attack_w(sensor, 1.0, sign=-1)[None, :]
    result = c.correct(attacked)
    moved = np.atleast_1d(sensor.distance(result.h_corrected)) - np.atleast_1d(result.d_observed)
    assert np.all(moved >= -1e-12), "correction must never push further toward risk"


def test_grid_is_absolute_and_shared(freeze):
    grid = freeze["attacks"]["generation_grid_absolute"]
    assert grid[0] == 0.0 and sorted(grid) == grid
    assert freeze["attacks"]["same_values_for_every_rho"] is True
    certs = [c["epsilon_h"] for c in freeze["geometry"]["certificates"].values()]
    assert min(grid[1:]) < min(certs), "grid must reach below every certificate"
    assert max(grid) > max(certs), "grid must reach above every certificate"


def test_grid_builder_brackets_the_certificates():
    grid = attack_grid_absolute([0.106, 0.157, 0.169])
    assert 0.0 in grid and max(grid) > 0.169 and min(g for g in grid if g > 0) < 0.106
