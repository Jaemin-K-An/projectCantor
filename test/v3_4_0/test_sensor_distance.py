import numpy as np
import pytest

from cantor_guard_v340.sensor_distance import SensorHyperplane


def test_distance_is_normalised_logit(rng):
    w, b = rng.normal(size=32), 0.7
    s = SensorHyperplane(w, b)
    h = rng.normal(size=(10, 32))
    assert np.allclose(s.distance(h), s.logit(h) / np.linalg.norm(w))


def test_distance_is_zero_exactly_on_the_hyperplane(rng, ):
    s = SensorHyperplane(rng.normal(size=48), -1.3)
    h = rng.normal(size=(20, 48))
    assert np.abs(s.distance(s.project_to_hyperplane(h))).max() < 1e-10


def test_boundary_is_d_equals_zero_not_a_fitted_tau(rng):
    """The decision boundary is structural, so there is no tau to estimate."""
    s = SensorHyperplane(rng.normal(size=16), 2.0)
    on_plane = s.project_to_hyperplane(rng.normal(size=16))
    assert abs(s.distance(on_plane)) < 1e-10
    assert abs(s.logit(on_plane)) < 1e-9


def test_rejects_degenerate_sensor():
    with pytest.raises(ValueError):
        SensorHyperplane(np.zeros(8), 0.0)
    with pytest.raises(ValueError):
        SensorHyperplane(np.ones(8), float("nan"))


def test_coupling_is_cosine_and_scale_invariant(rng):
    w, v = rng.normal(size=64), rng.normal(size=64)
    s = SensorHyperplane(w, 0.0)
    expected = float(w @ v / (np.linalg.norm(w) * np.linalg.norm(v)))
    assert s.coupling(v) == pytest.approx(expected)
    assert s.coupling(7.5 * v) == pytest.approx(expected)


def test_dimension_mismatch_raises(rng):
    s = SensorHyperplane(rng.normal(size=10), 0.0)
    with pytest.raises(ValueError):
        s.distance(rng.normal(size=(3, 11)))
