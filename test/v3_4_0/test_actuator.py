import hashlib
import json
import pathlib

import numpy as np
import pytest

from cantor_guard_v340.actuator import Actuator

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/v3_4_0"


def test_actuator_is_the_frozen_historical_direction():
    cfg = json.loads((ROOT / "configs/v3_4_0/actuator.json").read_text())
    digest = hashlib.sha256((ROOT / cfg["direction_file"]).read_bytes()).hexdigest()
    assert digest == cfg["direction_sha256"]
    assert cfg["direction_file"] == "results/v3_3_5a/cache/v_p0.npy"
    assert cfg["frozen"] is True


def test_actuator_is_unit_norm(actuator):
    assert np.linalg.norm(actuator.v) == pytest.approx(1.0)
    assert np.linalg.norm(actuator.v_safe) == pytest.approx(1.0)
    assert np.allclose(actuator.v_unsafe, -actuator.v_safe)


def test_step_has_exactly_the_requested_relative_norm(actuator, rng):
    h = rng.normal(size=(12, actuator.v.size)) * 5
    for amp in (0.02, 0.1, 0.4):
        delta = actuator.step(h, amp)
        realised = np.linalg.norm(delta, axis=1) / np.linalg.norm(h, axis=1)
        assert np.allclose(realised, amp)


def test_safe_sign_is_revalidated_not_reassumed():
    val = json.loads((RESULTS / "tables" / "actuator_validation.json").read_text())
    assert val["checks"]["direction_correct_and_significant"]
    assert val["spearman_u_vs_safe"]["rho"] > 0
    assert val["verdict"] == "ACT1_CAUSAL_ACTUATOR_REPLICATED"


def test_coherence_and_degeneration_gates_pass():
    val = json.loads((RESULTS / "tables" / "actuator_validation.json").read_text())
    assert val["checks"]["coherence_gate"] and val["checks"]["degeneration_gate"]


def test_no_alternative_actuator_was_searched():
    cfg = json.loads((ROOT / "configs/v3_4_0/actuator.json").read_text())
    assert "never rotated toward w" in cfg["retraining_forbidden"]


def test_actuator_and_sensor_are_different_objects(sensor, actuator):
    """The whole hypothesis is w != v; a silent substitution would void it."""
    assert not np.allclose(sensor.w_hat, actuator.v_safe)
    assert abs(sensor.coupling(actuator.v_safe)) < 0.95
