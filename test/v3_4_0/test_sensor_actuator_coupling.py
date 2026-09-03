import json
import pathlib

import numpy as np
import pytest

from cantor_guard_v340.actuator import achievable_sensor_shift, coupling
from cantor_guard_v340.sensor_actuator_controller import SensorActuatorCantorController

RESULTS = pathlib.Path(__file__).resolve().parents[2] / "results/v3_4_0"


def test_kappa_is_the_cosine_between_sensor_normal_and_safe_actuator(sensor, actuator):
    c = coupling(sensor, actuator)
    assert c["kappa"] == pytest.approx(float(sensor.w_hat @ actuator.v_safe))
    assert c["angle_w_v_deg"] == pytest.approx(
        np.degrees(np.arccos(np.clip(c["cos_w_v"], -1, 1))))


def test_controller_step_moves_the_sensor_by_exactly_eta_times_kappa(sensor, actuator, freeze, rng):
    c = SensorActuatorCantorController(sensor=sensor, actuator=actuator,
                                       W=float(freeze["geometry"]["W"]), rho=1 / 3, eta=0.03)
    h = rng.normal(size=(25, sensor.w.size)) * 4
    result = c.correct(h)
    realised = np.atleast_1d(sensor.distance(result.h_corrected)) - np.atleast_1d(result.d_observed)
    assert np.allclose(realised, result.delta_d_expected, atol=1e-9)
    norms = np.linalg.norm(h, axis=1)
    assert np.allclose(result.delta_d_expected, result.q_ctrl * norms * c.kappa)


def test_achievable_shift_formula(sensor, actuator):
    norms = np.array([10.0, 18.4, 25.0])
    got = achievable_sensor_shift(sensor, actuator, norms, 0.05)
    assert np.allclose(got, 0.05 * norms * abs(coupling(sensor, actuator)["kappa"]))


def test_controllability_gate_was_frozen_before_being_evaluated():
    cfg = json.loads((pathlib.Path(__file__).resolve().parents[2]
                      / "configs/v3_4_0/controller.json").read_text())
    gate = cfg["CONTROLLABILITY_GATE"]
    assert gate["fraction_required"] == 0.90 and gate["q_cap"] == 0.05
    assert "2W/27" in gate["criterion"]
    assert cfg["written_before"].startswith("W was calibrated")


def test_controllability_gate_result_is_consistent_with_its_own_criterion():
    geom = json.loads((RESULTS / "tables" / "sensor_window_and_coupling.json").read_text())
    c = geom["controllability"]
    passed = c["fraction_meeting_criterion"] >= c["fraction_required"]
    assert c["passed"] == passed
    assert c["verdict"] == ("COUP1_CONTROLLABLE" if passed else "COUP2_WEAK_SENSOR_ACTUATOR_COUPLING")


def test_no_post_hoc_rotation_of_the_actuator(sensor, actuator):
    """v must stay the historical direction, never a blend of v and w."""
    geom = json.loads((RESULTS / "tables" / "sensor_window_and_coupling.json").read_text())
    assert coupling(sensor, actuator)["kappa"] == pytest.approx(geom["coupling"]["kappa"], abs=1e-9)
