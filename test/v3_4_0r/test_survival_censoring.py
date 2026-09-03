import numpy as np
import pytest

from _stats import discrete_survival

GRID = [0.1, 0.2, 0.4, 0.8, 1.6]


def test_all_censored_gives_no_median():
    out = discrete_survival([np.nan] * 30, 1.6, GRID)
    assert out["median"] is None
    assert out["median_status"] == "NOT_IDENTIFIED_IN_TESTED_RANGE"
    assert out["censoring_rate"] == pytest.approx(1.0)
    assert out["final_survival"] == pytest.approx(1.0)


def test_the_v340_trap_is_refused():
    """70% censored, observed failures all large: no population median exists."""
    out = discrete_survival([1.6] * 6 + [np.nan] * 14, 1.6, GRID)
    assert out["median_status"] == "NOT_IDENTIFIED_IN_TESTED_RANGE"
    assert out["censoring_rate"] == pytest.approx(0.7)
    # an observed-only median would have reported 1.6 here
    assert out["median"] is None


def test_median_is_reported_when_the_curve_actually_crosses():
    out = discrete_survival([0.1] * 12 + [0.2] * 4 + [np.nan] * 4, 1.6, GRID)
    assert out["median_status"] == "IDENTIFIED"
    assert out["median"] == pytest.approx(0.1)


def test_survival_is_monotone_non_increasing():
    out = discrete_survival([0.1] * 3 + [0.4] * 5 + [1.6] * 2 + [np.nan] * 10, 1.6, GRID)
    s = [row["survival"] for row in out["curve"]]
    assert all(s[i] >= s[i + 1] - 1e-12 for i in range(len(s) - 1))
    assert 0.0 <= s[-1] <= 1.0


def test_restricted_mean_is_bounded_by_the_grid():
    out = discrete_survival([0.4] * 10 + [np.nan] * 10, 1.6, GRID)
    assert 0 < out["restricted_mean_failure_free_epsilon"] <= max(GRID)


def test_event_and_censoring_counts_add_up():
    first = [0.2] * 7 + [np.nan] * 13
    out = discrete_survival(first, 1.6, GRID)
    assert out["n"] == 20 and out["n_events"] == 7
    assert out["censoring_rate"] == pytest.approx(13 / 20)


def test_empty_input_is_handled():
    assert discrete_survival([], 1.6, GRID)["median_status"] == "NO_DATA"
