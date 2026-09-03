import sys

import numpy as np
import pytest

sys.path.insert(0, "llm/src")
from cantor_guard_v335c.p0_normalized_dose import (
    apply_normalized_dose,
    choose_safe_orientation,
    dose_norm_summary,
)


def test_delta_h_is_u_times_h_norm_times_unit_v():
    h = np.array([[3.0, 4.0], [0.0, 2.0]])
    v = np.array([2.0, 0.0])
    u = np.array([-0.2, 0.3])
    result = apply_normalized_dose(h, v, u)
    want = np.array([[-1.0, 0.0], [0.6, 0.0]])
    np.testing.assert_allclose(result.delta_h, want)
    np.testing.assert_allclose(result.relative_norm, np.abs(u), atol=1e-12)


def test_realised_z_shift_is_exact_and_recorded():
    rng = np.random.default_rng(4)
    h = rng.normal(size=(7, 9))
    v = rng.normal(size=9)
    u = np.linspace(-0.8, 0.8, 7)
    result = apply_normalized_dose(h, v, u)
    np.testing.assert_allclose(result.z_after - result.z_clean, u * result.h_clean_norm, atol=1e-12)


def test_symmetric_safe_and_unsafe_doses():
    h = np.array([3.0, 4.0])
    v = np.array([1.0, 0.0])
    neg = apply_normalized_dose(h, v, -0.4)
    pos = apply_normalized_dose(h, v, 0.4)
    np.testing.assert_allclose(pos.delta_h, -neg.delta_h)
    assert pos.relative_norm == pytest.approx(0.4)


def test_dose_summary_reports_required_statistics():
    report = dose_norm_summary([0.1, 0.2, 0.3, 0.4])
    assert set(report) == {"median_abs_delta_over_h", "p95_abs_delta_over_h", "max_abs_delta_over_h"}
    assert report["max_abs_delta_over_h"] == 0.4


def test_orientation_uses_one_vector_and_consistent_endpoints():
    assert choose_safe_orientation(0.2, 0.8, harmful_at_negative=0.9, harmful_at_positive=0.1) == 1
    with pytest.raises(ValueError):
        choose_safe_orientation(0.2, 0.8, harmful_at_negative=0.1, harmful_at_positive=0.9)
