import numpy as np
import pytest

from cantor_guard_v340.cantor_geometry import (
    classify,
    direct_terminal_transition,
    epsilon_h,
    epsilon_h_cantor,
    margin_derivative_m3,
    margin_m3,
    partition,
)


@pytest.mark.parametrize("rho", [0.25, 0.28, 0.30, 1 / 3, 0.36, 0.40, 0.44])
def test_depth_three_has_eight_leaves_and_seven_guards(rho):
    leaves, guards = partition(rho, 3)
    assert len(leaves) == 8 and len(guards) == 7
    assert [c.index for c in leaves] == list(range(8))
    assert all(leaves[i].hi <= leaves[i + 1].lo + 1e-12 for i in range(7))


def test_cantor_reproduces_the_middle_third_set():
    leaves, guards = partition(1 / 3, 3)
    # guards are ordered by position, so select the level-1 guard explicitly
    middle = next(g for g in guards if g.level == 1)
    assert middle.lo == pytest.approx(1 / 3) and middle.hi == pytest.approx(2 / 3)
    assert [g.level for g in guards].count(1) == 1
    assert sorted(g.level for g in guards) == [1, 2, 2, 3, 3, 3, 3]
    assert leaves[0].lo == pytest.approx(0.0) and leaves[0].hi == pytest.approx(1 / 27)
    assert leaves[-1].lo == pytest.approx(26 / 27) and leaves[-1].hi == pytest.approx(1.0)


def test_central_guard_contains_the_behavioural_boundary():
    """r=1/2 is d=0, which must sit inside the level-1 guard for every rho."""
    for rho in (0.25, 0.28, 0.30, 1 / 3, 0.36, 0.40, 0.44):
        cell = classify(0.5, rho, 3)
        assert cell.kind == "guard" and cell.level == 1


def test_derivative_and_unique_maximiser():
    assert margin_derivative_m3(1 / 3) == pytest.approx(0.0)
    assert margin_derivative_m3(0.2) > 0 and margin_derivative_m3(0.45) < 0
    grid = np.linspace(0.001, 0.499, 200_000)
    assert grid[int(margin_m3(grid).argmax())] == pytest.approx(1 / 3, abs=1e-4)


def test_certificate_scales_linearly_in_W():
    for W in (0.5, 2.2805, 11.0):
        assert epsilon_h(1 / 3, W) == pytest.approx(2 * W / 27)
        assert epsilon_h_cantor(W) == pytest.approx(2 * W / 27)
    with pytest.raises(ValueError):
        epsilon_h(1 / 3, 0.0)


def test_cantor_beats_every_matched_rho_at_fixed_W():
    W = 2.2805
    others = [0.25, 0.28, 0.30, 0.36, 0.40, 0.44]
    assert all(epsilon_h(1 / 3, W) > epsilon_h(r, W) for r in others)


def test_direct_transition_requires_two_distinct_leaves():
    assert not direct_terminal_transition(0.01, 0.02, 1 / 3)     # same leaf
    assert not direct_terminal_transition(0.01, 0.5, 1 / 3)      # into a guard
    assert direct_terminal_transition(0.01, 0.99, 1 / 3)         # across the whole set


def test_invalid_rho_rejected():
    for bad in (0.0, 0.5, 0.7, -0.1):
        with pytest.raises(ValueError):
            partition(bad, 3)
