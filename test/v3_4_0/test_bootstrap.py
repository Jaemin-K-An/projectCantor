import numpy as np
import pathlib
import pytest
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts/v3_4_0"))
from _stats import max_t_intervals, shared_index  # noqa: E402


def test_index_shape_and_determinism():
    a = shared_index(80, n_boot=500, seed=3400)
    b = shared_index(80, n_boot=500, seed=3400)
    assert a.shape == (500, 80) and np.array_equal(a, b)
    assert not np.array_equal(a, shared_index(80, n_boot=500, seed=1))


def test_one_matrix_is_reused_across_every_contrast():
    """The V3.3.5b defect: a fresh resample per loop destroys the pairing."""
    idx = shared_index(40, n_boot=300, seed=7)
    arms = {name: np.random.default_rng(i).normal(size=40) for i, name in enumerate("abc")}
    out = max_t_intervals(arms, [("a", "b"), ("a", "c")], idx)
    # a shared critical value is only meaningful if the family moved together
    assert out["critical_value"] > 0
    assert len({round(c["se"], 12) for c in out["contrasts"]}) == 2


def test_perfectly_paired_arms_give_a_zero_width_interval():
    idx = shared_index(30, n_boot=200, seed=3)
    x = np.random.default_rng(0).normal(size=30)
    out = max_t_intervals({"a": x, "b": x}, [("a", "b")], idx)
    c = out["contrasts"][0]
    assert c["mean_difference"] == pytest.approx(0.0)
    assert c["se"] == pytest.approx(0.0, abs=1e-12)


def test_simultaneous_interval_is_wider_than_a_single_se():
    idx = shared_index(60, n_boot=2000, seed=11)
    rng = np.random.default_rng(5)
    arms = {"a": rng.normal(size=60), "b": rng.normal(size=60), "c": rng.normal(size=60)}
    out = max_t_intervals(arms, [("a", "b"), ("a", "c")], idx)
    assert out["critical_value"] > 1.96
    for c in out["contrasts"]:
        assert c["simultaneous_hi"] - c["simultaneous_lo"] > 2 * 1.96 * c["se"] - 1e-9


def test_detects_a_real_shift():
    idx = shared_index(80, n_boot=2000, seed=2)
    rng = np.random.default_rng(9)
    base = rng.normal(size=80)
    out = max_t_intervals({"a": base + 1.5, "b": base}, [("a", "b")], idx)
    assert out["contrasts"][0]["excludes_zero"]
