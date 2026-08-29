"""Barrier mathematics (harness §49 items 1-8), Python side."""
import numpy as np, pytest
from cantor_guard.cantor_barrier import (cantor_gap_list, build_layout, smoothstep,
                                         dsmoothstep, worst_displacement,
                                         LAYOUT_FAMILIES)

NS = [2, 3, 4, 5, 6, 7]


def test_smoothstep_contract():
    assert smoothstep(0.0) == 0.0 and smoothstep(1.0) == 1.0
    assert dsmoothstep(0.0) == 0.0 and dsmoothstep(1.0) == 0.0
    assert np.isclose(dsmoothstep(0.5), 1.5)
    u = np.linspace(0, 1, 4001)
    assert (dsmoothstep(u) >= -1e-15).all()
    assert np.isclose(np.trapz(dsmoothstep(u), u), 1.0, rtol=1e-6)


@pytest.mark.parametrize("n", NS)
def test_gap_counts_and_widths(n):
    """N_k = 2^(k-1), w_k = 3^-k, total removed = 1-(2/3)^n."""
    gaps = cantor_gap_list(n)
    assert len(gaps) == 2 ** n - 1
    for k in range(1, n + 1):
        lv = [g for g in gaps if g.level == k]
        assert len(lv) == 2 ** (k - 1)
        for g in lv:
            assert np.isclose(g.width, 3.0 ** -k, rtol=1e-9)
    assert np.isclose(sum(g.width for g in gaps), 1 - (2 / 3) ** n, rtol=1e-9)
    for a, b in zip(gaps, gaps[1:]):
        assert a.b <= b.a + 1e-12                       # disjoint, sorted


@pytest.mark.parametrize("n", NS)
def test_theorem_A_level_action(n):
    """Theorem A: every level carries total action E0, in EVERY family."""
    E0 = 1.0 / n
    for fam in ["L3_periodic", "L4_random", "L5_shuffled",
                "L6_center_anchored", "L7_cantor"]:
        L = build_layout(fam, n, E0, seed=7)
        for k in range(1, n + 1):
            e = sum(L.est[i] for i, g in enumerate(L.gaps) if g.level == k)
            assert np.isclose(e, E0, rtol=1e-12)
        assert np.isclose(L.total_action(), 1.0, rtol=1e-12)


@pytest.mark.parametrize("n", NS)
def test_theorem_B_peak_scaling(n):
    """Theorem B: peak = 3*E0*(3/2)^k and the ratio is exactly 3/2."""
    L = build_layout("L7_cantor", n, 1.0 / n)
    for k in range(2, n + 1):
        assert np.isclose(L.peak_of_level(k) / L.peak_of_level(k - 1), 1.5, rtol=1e-14)
    for k in range(1, n + 1):
        g = next(g for g in L.gaps if g.level == k)
        assert np.isclose(L.field(g.centre)[0], L.peak_of_level(k), rtol=1e-9)


def test_ordering_controls_match_widths_and_energies():
    """Shuffles preserve every width, level and energy: ablation is ordering-only."""
    n, E0 = 6, 1.0 / 6
    base = build_layout("L7_cantor", n, E0)
    ref_w = np.sort([g.width for g in base.gaps])
    ref_l = sorted(g.level for g in base.gaps)
    for fam in ["L3_periodic", "L4_random", "L5_shuffled", "L6_center_anchored"]:
        for s in range(5):
            L = build_layout(fam, n, E0, seed=s)
            assert np.allclose(np.sort([g.width for g in L.gaps]), ref_w, rtol=1e-9)
            assert sorted(g.level for g in L.gaps) == ref_l
            assert np.isclose(L.total_action(), base.total_action(), rtol=1e-12)
            assert L.gaps[0].a >= -1e-12 and L.gaps[-1].b <= 1 + 1e-9


def test_center_anchored_keeps_boundary_barrier():
    """L6 pins the level-1 gap on r = 1/2; L5 generally does not."""
    n, E0 = 6, 1.0 / 6
    for s in range(8):
        L = build_layout("L6_center_anchored", n, E0, seed=s)
        g1 = next(g for g in L.gaps if g.level == 1)
        assert np.isclose(g1.centre, 0.5, atol=1e-9)
        assert np.isclose(g1.width, 1 / 3, rtol=1e-12)
    off = [abs(next(g for g in build_layout("L5_shuffled", n, E0, seed=s).gaps
                    if g.level == 1).centre - 0.5) for s in range(8)]
    assert max(off) > 0.05


def test_field_nonnegative_and_potential_monotone():
    r = np.linspace(0, 1, 20001)
    for fam in LAYOUT_FAMILIES:
        L = build_layout(fam, 5, 1 / 5, seed=1)
        assert (L.field(r) >= -1e-15).all()
        assert (np.diff(L.potential(r)) >= -1e-12).all()


def test_proposition_E_cantor_is_minimiser_but_barely():
    """Cantor minimises worst displacement, but only slightly vs shuffles and
    hugely vs periodic -- the pre-registered size of the effect."""
    n, E0 = 8, 1.0 / 8
    C = build_layout("L7_cantor", n, E0)
    for k in range(1, n + 1):
        dc = worst_displacement(C, k)
        dp = worst_displacement(build_layout("L3_periodic", n, E0), k)
        ds = [worst_displacement(build_layout("L5_shuffled", n, E0, seed=s), k)
              for s in range(20)]
        assert dp >= dc - 1e-9
        assert np.median(ds) >= dc - 1e-9
        assert np.median(ds) <= dc * 1.10        # the margin is SMALL
