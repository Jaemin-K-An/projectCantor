import sys

import numpy as np
import pytest

sys.path.insert(0, "llm/src")
from cantor_guard_v335c.p0_behavioral_boundary import (
    fit_behavioral_boundary,
    fit_logistic,
    isotonic_crossing,
    monotonicity_checks,
    prompt_cluster_bootstrap,
)


def test_logistic_boundary_uses_realised_z():
    z = np.linspace(-3, 3, 2000)
    rng = np.random.default_rng(9)
    y = rng.random(len(z)) < 1 / (1 + np.exp(-(1.2 * z - 0.6)))
    a, b = fit_logistic(z, y)
    assert b > 0
    assert -a / b == pytest.approx(0.5, abs=0.2)


def test_isotonic_supports_either_frozen_orientation():
    z = np.repeat(np.arange(-2, 3), 10)
    y_pos = np.repeat([0, 0, 0, 1, 1], 10)
    y_neg = y_pos[::-1]
    assert isotonic_crossing(z, y_pos, expected_slope_sign=1) is not None
    assert isotonic_crossing(z, y_neg, expected_slope_sign=-1) is not None


def test_monotonicity_detects_major_reversal():
    u = np.repeat([-1, 0, 1], 20)
    y = np.concatenate([np.zeros(20), np.ones(20), np.zeros(20)])
    report = monotonicity_checks(u, y, safe_sign=1, reversal_tolerance=0.15)
    assert report["major_adjacent_reversals"] == 1
    assert not report["no_major_reversal"]


def test_cluster_bootstrap_requires_at_least_twenty_thousand():
    with pytest.raises(ValueError):
        prompt_cluster_bootstrap([0, 1], [0, 1], ["a", "b"], n_boot=19_999)


def test_sigma_p0_uses_clean_prompt_distribution_not_dose_shifted_z():
    prompts = np.repeat(np.arange(12), 5)
    clean_by_prompt = np.linspace(-0.5, 0.5, 12)
    z_clean = np.repeat(clean_by_prompt, 5)
    u = np.tile(np.linspace(-2, 2, 5), 12)
    z_after = z_clean + 10 * u
    y = (z_after > 0).astype(int)
    result = fit_behavioral_boundary(
        z_after=z_after, z_clean=z_clean, outcome=y, pid=prompts, u=u,
        coherence=np.ones(len(y)), degeneration=np.zeros(len(y)),
        safe_sign=1, confirm_direction_agrees=True, n_boot=20_000, seed=3,
    )
    assert result.sigma_p0 == pytest.approx(np.std(clean_by_prompt, ddof=1))
    assert result.sigma_p0 < np.std(z_after, ddof=1) / 10
