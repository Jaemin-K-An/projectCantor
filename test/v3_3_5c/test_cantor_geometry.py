import sys

import numpy as np
import pytest

sys.path.insert(0, "llm/src")
from cantor_guard_v335c.cantor_geometry import (
    RHO_CANTOR,
    classify,
    direct_terminal_transition,
    epsilon_cantor,
    epsilon_z,
    margin_derivative_m3,
    margin_m3,
    partition,
)


def test_depth_three_has_eight_leaves_and_seven_guards():
    leaves, guards = partition(RHO_CANTOR, 3)
    assert len(leaves) == 8
    assert len(guards) == 7
    assert [leaf.index for leaf in leaves] == list(range(8))


def test_middle_third_first_guard_contains_half():
    cell = classify(0.5, RHO_CANTOR, 3)
    assert cell.kind == "guard" and cell.level == 1
    assert cell.lo == pytest.approx(1 / 3)
    assert cell.hi == pytest.approx(2 / 3)


def test_frozen_m3_formula_and_unique_maximum():
    grid = np.linspace(1e-5, 0.49999, 100_001)
    values = margin_m3(grid)
    assert grid[np.argmax(values)] == pytest.approx(1 / 3, abs=1e-5)
    assert margin_m3(1 / 3) == pytest.approx(1 / 27)
    assert margin_derivative_m3(1 / 3) == pytest.approx(0)


def test_residual_certificate_is_2w_m3_and_cantor_value():
    for rho in (0.25, 0.30, 1 / 3, 0.40, 0.44):
        assert epsilon_z(rho, 2.5) == pytest.approx(5 * rho**2 * (1 - 2 * rho))
    assert epsilon_cantor(2.5) == pytest.approx(5 / 27)


def test_direct_crossing_is_impossible_below_certificate_in_r_space():
    rng = np.random.default_rng(10)
    for rho in (0.25, 0.28, 0.30, 1 / 3, 0.36, 0.40, 0.44):
        margin = float(margin_m3(rho))
        leaves, _ = partition(rho, 3)
        for leaf in leaves:
            for _ in range(10):
                clean = rng.uniform(leaf.lo + 1e-10, leaf.hi - 1e-10)
                for sign in (-1, 1):
                    attacked = clean + sign * 0.999 * margin
                    assert not direct_terminal_transition(clean, attacked, rho)
