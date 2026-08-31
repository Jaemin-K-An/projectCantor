"""V3.3.1 mandatory mathematical tests (harness section 62)."""
import sys, math
sys.path.insert(0, "llm/src")
from fractions import Fraction
import numpy as np, pytest

from cantor_guard_v331.guard_geometry import (
    guard_width, retention, hausdorff_dim, alpha_field, alpha_sensitivity,
    rho_star, kappa_of_rho, bottleneck, feasible, retained_measure,
    new_guard_measure, cumulative_coverage, new_coverage_argmax, RHO_CANTOR)
from cantor_guard_v331.rho_family import RhoBarrier, rho_gap_list
from cantor_guard_v331.refinement import (structural_drift, deployment_drift,
                                          backward_compatibility, measures)
from cantor_guard_v33.symbolic_cantor import cantor_field

GRID = np.linspace(0.001, 0.499, 4001)


# ------------------------------------------------------------- geometry
def test_guard_width_identity():
    for r in GRID:
        assert abs(guard_width(r) - (1 - 2 * r)) < 1e-15
        assert abs(2 * r + guard_width(r) - 1.0) < 1e-15


def test_domain_enforced():
    for bad in (0.0, 0.5, -0.1, 0.7):
        with pytest.raises(ValueError):
            guard_width(bad)


# ------------------------------------------------------- THEOREM G / BGR
@pytest.mark.parametrize("kappa", [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0])
def test_theorem_G_rho_star(kappa):
    r = rho_star(kappa)
    assert abs(r - 1.0 / (2.0 + kappa)) < 1e-15
    assert feasible(r, kappa)                       # boundary is feasible
    assert not feasible(r + 1e-6, kappa)            # anything larger is not


def test_kappa_one_gives_exactly_one_third():
    assert rho_star(1.0) == 1.0 / 3.0
    assert abs(kappa_of_rho(1.0 / 3.0) - 1.0) < 1e-15
    assert abs(guard_width(1 / 3) - 1 / 3) < 1e-15   # guard == child


@pytest.mark.parametrize("kappa", [0.5, 0.75, 1.0, 1.5, 2.0])
def test_theorem_BGR_unique_max(kappa):
    vals = np.array([bottleneck(r, kappa) for r in GRID])
    i = int(vals.argmax())
    assert abs(GRID[i] - rho_star(kappa)) < 2e-3
    assert abs(vals[i] - 1.0 / (2.0 + kappa)) < 2e-3
    # unique: strictly increasing then strictly decreasing
    assert np.all(np.diff(vals[:i]) > -1e-12)
    assert np.all(np.diff(vals[i:]) < 1e-12)


def test_bottleneck_max_at_one_third_for_kappa_one():
    vals = np.array([bottleneck(r, 1.0) for r in GRID])
    assert abs(GRID[int(vals.argmax())] - 1 / 3) < 2e-3
    assert abs(vals.max() - 1 / 3) < 2e-3


def test_cantor_beats_rho_028_on_balanced_bottleneck():
    """The V3.3 comparison, done on the right quantity."""
    assert bottleneck(1 / 3, 1.0) > bottleneck(0.28, 1.0)
    assert abs(bottleneck(1 / 3, 1.0) - 1 / 3) < 1e-12
    assert abs(bottleneck(0.28, 1.0) - 0.28) < 1e-12


# ------------------------------------------------------------ THEOREM P
def test_monotonicity_on_domain():
    r = GRID
    assert np.all(np.diff([retention(x) for x in r]) > 0)
    assert np.all(np.diff([hausdorff_dim(x) for x in r]) > 0)
    assert np.all(np.diff([alpha_field(x) for x in r]) < 0)
    assert np.all(np.diff([alpha_sensitivity(x) for x in r]) < 0)


@pytest.mark.parametrize("kappa", [0.5, 1.0, 1.5, 2.0])
def test_constrained_joint_optimum_is_upper_boundary(kappa):
    rs = rho_star(kappa)
    feas = [x for x in GRID if feasible(x, kappa)]
    assert abs(max(feas) - rs) < 2e-3
    # all four objectives are optimised there
    assert retention(rs) >= max(retention(x) for x in feas) - 1e-9
    assert hausdorff_dim(rs) >= max(hausdorff_dim(x) for x in feas) - 1e-9
    assert alpha_field(rs) <= min(alpha_field(x) for x in feas) + 1e-9
    assert alpha_sensitivity(rs) <= min(alpha_sensitivity(x) for x in feas) + 1e-9


def test_cantor_amplifications_are_theorem_B_and_T_consistent():
    assert abs(alpha_field(1 / 3) - 1.5) < 1e-15
    assert abs(alpha_sensitivity(1 / 3) - 4.5) < 1e-15


# ------------------------------------------------------- COUNTEREXAMPLES
def test_without_guard_constraint_cantor_is_not_optimal():
    """rho -> 1/2 beats 1/3 on every individual objective."""
    big = 0.499
    assert retention(big) > retention(1 / 3)
    assert hausdorff_dim(big) > hausdorff_dim(1 / 3)
    assert alpha_field(big) < alpha_field(1 / 3)
    assert alpha_sensitivity(big) < alpha_sensitivity(1 / 3)


def test_kappa_not_one_shifts_the_optimum_away_from_one_third():
    for kappa in (0.5, 0.75, 1.25, 1.5, 2.0):
        assert abs(rho_star(kappa) - 1 / 3) > 1e-3


def test_new_coverage_argmax_is_not_one_third():
    """F_n(rho) = (1-2rho)(2rho)^n peaks at n/(2(n+1)), which is 1/3 only at n=2."""
    for n in (1, 2, 3, 5, 10, 50):
        want = n / (2 * (n + 1))
        assert abs(new_coverage_argmax(n) - want) < 1e-15
        grid = np.linspace(0.001, 0.499, 20001)
        vals = np.array([(1 - 2 * x) * (2 * x) ** n for x in grid])
        assert abs(grid[int(vals.argmax())] - want) < 2e-3
    assert abs(new_coverage_argmax(2) - 1 / 3) < 1e-15
    assert abs(new_coverage_argmax(3) - 1 / 3) > 0.04


# ------------------------------------------------------------ MEASURES
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_cantor_measures(n):
    assert abs(retained_measure(1 / 3, n) - (2 / 3) ** n) < 1e-15
    assert abs(new_guard_measure(1 / 3, n) - (1 / 3) * (2 / 3) ** n) < 1e-15
    assert abs(cumulative_coverage(1 / 3, n) - (1 - (2 / 3) ** n)) < 1e-15


@pytest.mark.parametrize("rho", [0.2, 0.25, 1 / 3, 0.4, 0.45])
@pytest.mark.parametrize("n", [1, 3, 6])
def test_general_rho_measures(rho, n):
    assert abs(retained_measure(rho, n) - (2 * rho) ** n) < 1e-15
    assert abs(new_guard_measure(rho, n) - (1 - 2 * rho) * (2 * rho) ** n) < 1e-15


# ------------------------------------------------------- rho-family controller
@pytest.mark.parametrize("n", [1, 3, 5, 8])
def test_rho_one_third_reproduces_cantor(n):
    r = np.linspace(0, 1, 20001)
    a = RhoBarrier(1 / 3, n, 1.0).field(r)
    b = cantor_field(r, n, 1.0)
    scale = max(1.0, float(np.abs(b).max()))
    assert np.abs(a - b).max() / scale < 1e-14 * 3.0 ** n


@pytest.mark.parametrize("rho", [0.2, 0.28, 1 / 3, 0.4])
@pytest.mark.parametrize("n", [2, 4, 6])
def test_symbolic_matches_explicit_gap_list(rho, n):
    b = RhoBarrier(rho, n, 1.0)
    gaps = rho_gap_list(rho, n)
    assert len(gaps) == 2 ** n - 1
    rng = np.random.default_rng(0)
    r = rng.uniform(0, 1, 5000)
    lv = b.level(r)
    for x, k in zip(r[:400], lv[:400]):
        hit = [t for t in gaps if t[1] <= x < t[2]]
        assert (k == hit[0][0]) if hit else (k == 0)


@pytest.mark.parametrize("rho", [0.2, 0.28, 1 / 3, 0.4])
def test_level_energy_totals_E0(rho):
    """Theorem A holds for the whole family, not just Cantor."""
    b = RhoBarrier(rho, 6, 1.0)
    for k in range(1, 7):
        assert abs(2 ** (k - 1) * b.gap_energy(k) - 1.0) < 1e-12


# ------------------------------------------------------------ THEOREM R
@pytest.mark.parametrize("rho", [0.2, 0.25, 1 / 3, 0.4])
@pytest.mark.parametrize("n", [2, 4, 6])
def test_refinement_consistency_is_exact(rho, n):
    d = structural_drift(rho, n)
    assert d["n_old_points"] > 0
    assert d["max_abs_drift"] == 0.0          # exactly zero, not "small"
    assert d["new_support_fraction"] > 0      # refinement did add something


def test_deployment_drift_is_reported_not_assumed_zero():
    """Under gain renormalisation the guarantee does NOT hold; the metric must
    show that rather than hiding it."""
    d = deployment_drift(1 / 3, 4, eta_n=1.0, eta_next=0.8)
    assert d["max_rel_drift"] > 0.1
    assert backward_compatibility(1 / 3, 4, 1.0, 1.0) == pytest.approx(1.0)
    assert backward_compatibility(1 / 3, 4, 1.0, 0.8) < 1.0


def test_new_support_lies_in_previous_survivor_set():
    rho, n = 1 / 3, 5
    a, b = RhoBarrier(rho, n), RhoBarrier(rho, n + 1)
    r = np.linspace(0, 1, 100001)
    new = b.in_guard(r) & ~a.in_guard(r)
    assert new.any()
    assert not a.in_guard(r[new]).any()       # new support subset of K_n
