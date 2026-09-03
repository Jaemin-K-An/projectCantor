import json
import pathlib

import numpy as np
import pytest

from cantor_guard_v340.cantor_geometry import epsilon_h, epsilon_h_cantor, margin_m3

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_theorem_s_still_holds(sensor, rng):
    dh = rng.normal(size=(300, sensor.w.size)) * rng.uniform(0.1, 4, size=(300, 1))
    assert np.all(np.abs(sensor.delta_distance(dh)) <= np.linalg.norm(dh, axis=1) + 1e-9)
    for eps in (0.1, 2.0):
        assert abs(sensor.delta_distance(eps * sensor.w_hat)) == pytest.approx(eps)


def test_translation_preserves_the_certificate(sensor, rng):
    """d(h) = d_0(h) - tau leaves every difference unchanged."""
    h, dh = rng.normal(size=(20, sensor.w.size)), rng.normal(size=(20, sensor.w.size))
    tau = 1.234
    d0 = np.atleast_1d(sensor.distance(h))
    assert np.allclose((np.atleast_1d(sensor.distance(h + dh)) - tau) - (d0 - tau),
                       np.atleast_1d(sensor.delta_distance(dh)))


def test_unique_maximiser_and_value():
    grid = np.linspace(1e-6, 0.5 - 1e-6, 2_000_001)
    assert grid[int(margin_m3(grid).argmax())] == pytest.approx(1 / 3, abs=1e-6)
    W = 2.2805212277347544
    assert epsilon_h(1 / 3, W) == pytest.approx(epsilon_h_cantor(W)) == pytest.approx(2 * W / 27)
    for rho in (0.25, 0.28, 0.30, 0.36, 0.40, 0.44):
        assert epsilon_h(1 / 3, W) > epsilon_h(rho, W)


def test_post_gate_certificate_run_is_not_confirmatory():
    invalid = json.loads((ROOT / "results/v3_4_0r/tables/POST_GATE_INVALIDATION.json").read_text())
    assert "results/v3_4_0r/tables/certificate_validation.json" in invalid["invalidated_artifacts"]
    assert invalid["blocking_gate"] == "ST3_WINDOW_SHIFT"


def test_certificate_scope_is_stated_narrowly():
    verdict = json.loads((ROOT / "results/v3_4_0r/tables/final_verdict.json").read_text())
    assert verdict["structural_claim_scope"] == "residual policy-transition certificate only"
    assert verdict["semantic_safety_guarantee_claimed"] is False
