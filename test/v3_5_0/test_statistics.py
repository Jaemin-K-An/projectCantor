import numpy as np

from scripts.v3_5_0._stats import discrete_survival, max_t_intervals, shared_index


def test_shared_bootstrap_is_deterministic_and_paired():
    a = shared_index(200, n_boot=100, seed=35000)
    b = shared_index(200, n_boot=100, seed=35000)
    assert np.array_equal(a, b)
    assert a.shape == (100, 200)


def test_max_t_uses_one_index_matrix():
    idx = shared_index(40, n_boot=500, seed=7)
    values = {"a": np.arange(40), "b": np.arange(40) - 1, "c": np.arange(40) + 2}
    got = max_t_intervals(values, [("a", "b"), ("a", "c")], idx)
    assert got["n_boot"] == 500 and len(got["contrasts"]) == 2


def test_heavy_censoring_does_not_fabricate_median():
    got = discrete_survival([np.nan, np.nan, np.nan, 2.0], [0, 1, 2, 3])
    assert got["censoring_rate"] == .75
    assert got["median"] is None
    assert got["median_status"] == "NOT_IDENTIFIED_IN_TESTED_RANGE"
