import sys

import numpy as np

sys.path.insert(0, "llm/src")
from cantor_guard_v335c.affine_coordinate import AffineCoordinate, calibrate_window


def test_boundary_is_exactly_centered_and_metric_preserved():
    coordinate = AffineCoordinate(tau=2.0, W=3.0, orientation=-1)
    assert coordinate.transform(2.0) == 0.5
    z = np.linspace(-1.0, 5.0, 301)
    r = coordinate.transform(z)
    np.testing.assert_allclose(np.abs(np.diff(r)) / np.diff(z), 1 / 6)


def test_outside_window_is_nan_never_clipped():
    coordinate = AffineCoordinate(tau=0.0, W=1.0, orientation=1)
    assert np.isnan(coordinate.transform(-2.0))
    assert np.isnan(coordinate.transform(2.0))
    assert coordinate.transform(-1.0) == 0.0
    assert coordinate.transform(1.0) == 1.0


def test_frozen_window_rule_and_coverage():
    z = np.linspace(-2, 2, 1001)
    W = calibrate_window(z, 0.0)
    assert W == 1.05 * np.quantile(np.abs(z), 0.99)
    coordinate = AffineCoordinate(0, W, 1)
    assert coordinate.coverage(z) >= 0.99
