import numpy as np
import pytest

from _stats import max_t_intervals, shared_index


def test_one_matrix_is_reused_and_deterministic():
    a = shared_index(80, n_boot=500, seed=34000)
    assert a.shape == (500, 80)
    assert np.array_equal(a, shared_index(80, n_boot=500, seed=34000))
    assert not np.array_equal(a, shared_index(80, n_boot=500, seed=1))


def test_pairing_is_preserved_for_identical_arms():
    idx = shared_index(40, n_boot=400, seed=3)
    x = np.random.default_rng(0).normal(size=40)
    out = max_t_intervals({"a": x, "b": x}, [("a", "b")], idx)
    c = out["contrasts"][0]
    assert c["mean_difference"] == pytest.approx(0.0)
    assert c["se"] == pytest.approx(0.0, abs=1e-12)
    assert not c["excludes_zero"]


def test_simultaneous_is_wider_than_pointwise():
    idx = shared_index(60, n_boot=3000, seed=11)
    rng = np.random.default_rng(5)
    arms = {k: rng.normal(size=60) for k in "abcd"}
    out = max_t_intervals(arms, [("a", "b"), ("a", "c"), ("a", "d")], idx)
    assert out["critical_value"] > 1.96
    for c in out["contrasts"]:
        assert c["simultaneous_hi"] - c["simultaneous_lo"] >= 2 * 1.96 * c["se"] - 1e-9


def test_detects_a_real_effect():
    idx = shared_index(80, n_boot=3000, seed=2)
    base = np.random.default_rng(9).normal(size=80)
    out = max_t_intervals({"a": base + 1.2, "b": base}, [("a", "b")], idx)
    assert out["contrasts"][0]["excludes_zero"]


def test_config_declares_one_shared_matrix():
    import json, pathlib

    stats = json.loads((pathlib.Path(__file__).resolve().parents[2]
                        / "configs/v3_4_0r/statistics.json").read_text())
    assert stats["unit"] == "prompt"
    assert stats["n_boot"] >= 20000
    assert "shared" in stats["method"].lower()
