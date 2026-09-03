import numpy as np
import pytest

from cantor_guard_v340.cantor_geometry import epsilon_h, epsilon_h_cantor, margin_m3
from cantor_guard_v340.sensor_distance import SensorHyperplane


def test_theorem_s_one_lipschitz(rng):
    s = SensorHyperplane(rng.normal(size=128), 0.4)
    dh = rng.normal(size=(500, 128)) * rng.uniform(0.1, 5.0, size=(500, 1))
    assert np.all(np.abs(s.delta_distance(dh)) <= np.linalg.norm(dh, axis=1) + 1e-9)
    assert np.all(s.lipschitz_slack(dh) >= -1e-9)


def test_bound_is_tight_along_the_sensor_normal(rng):
    s = SensorHyperplane(rng.normal(size=64), -0.9)
    for eps in (0.01, 1.0, 17.5):
        assert abs(s.delta_distance(eps * s.w_hat)) == pytest.approx(eps, rel=1e-12)
        assert s.lipschitz_slack(eps * s.w_hat)[0] == pytest.approx(0.0, abs=1e-9)


def test_delta_distance_is_translation_invariant(rng):
    s = SensorHyperplane(rng.normal(size=40), 3.1)
    h, dh = rng.normal(size=(8, 40)), rng.normal(size=(8, 40))
    assert np.allclose(np.atleast_1d(s.distance(h + dh)) - np.atleast_1d(s.distance(h)),
                       np.atleast_1d(s.delta_distance(dh)))


def test_m3_and_unique_maximiser():
    rho = np.linspace(1e-6, 0.5 - 1e-6, 2_000_001)
    m = margin_m3(rho)
    assert rho[int(m.argmax())] == pytest.approx(1 / 3, abs=1e-6)
    assert margin_m3(1 / 3) == pytest.approx(1 / 27)
    # strictly increasing then strictly decreasing about 1/3
    assert np.all(np.diff(m[rho < 1 / 3]) > 0)
    assert np.all(np.diff(m[rho > 1 / 3]) < 0)


def test_epsilon_equals_two_w_times_m3():
    W = 2.2805
    for rho in (0.25, 0.28, 0.30, 1 / 3, 0.36, 0.40, 0.44):
        assert epsilon_h(rho, W) == pytest.approx(2 * W * margin_m3(rho))
    assert epsilon_h(1 / 3, W) == pytest.approx(epsilon_h_cantor(W))
    assert epsilon_h_cantor(W) == pytest.approx(2 * W / 27)


def test_epsilon_is_independent_of_eta():
    """The certificate is geometry; no controller gain appears in it."""
    import inspect

    assert "eta" not in inspect.signature(epsilon_h).parameters
