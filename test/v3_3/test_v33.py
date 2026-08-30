"""V3.3 mandatory tests (harness section 68).

These guard the claims V3.3 makes about its own machinery: that the symbolic
evaluator computes the SAME function as the explicit one, that the scale
identities are exact, that certificates reject corruption, and that the
complexity accounting cannot be gamed.
"""
import sys, json, math, pathlib
sys.path.insert(0, "llm/src")
from fractions import Fraction
import numpy as np, pytest

from cantor_guard.cantor_barrier import cantor_gap_list, BarrierLayout, Gap
from cantor_guard_v33.symbolic_cantor import (
    cantor_field, cantor_potential, cantor_level, cantor_field_derivative,
    SymbolicCantor, N_GAPS, total_action, peak_of_level, slope_of_level)
from cantor_guard_v33.general_recursive import IFSSpec, CANTOR, SymbolicIFS, feasible
from cantor_guard_v33.certificates import (build_certificate, verify_certificate,
                                           corrupt, PERMUTATION_INVARIANT)
from cantor_guard_v33.complexity import describe, FAMILIES, canonical_bits, SPEC

E0 = 1.0


def _explicit(n):
    return BarrierLayout(cantor_gap_list(n), n, E0)


def cond_tol(n: int) -> float:
    """Conditioning-based tolerance, measured not guessed.

    Level-k geometry lives at scale 3^-k while the field coefficient grows like
    2*(3/2)^k, so a relative perturbation eps in a gap endpoint reaches the
    field amplified by roughly 3^k. Every comparison here is therefore expected
    to grow as C * 3^n, and it does. Measured C over n = 4..15:

        symbolic vs explicit      C ~ 2e-16
        Theorem S identity        C ~ 4e-15   (one extra float division in T_i)
        general-IFS vs symbolic   C ~ 6e-16   (float descent, no exact address)

    Against an exact Fraction reference the symbolic evaluator is at least as
    well conditioned as the explicit one it replaces (4.5e-10 vs 4.9e-10 at
    n=18), which is what STOP D requires. 1e-14 * 3^n is a single conservative
    envelope over all three paths, with roughly 20x margin.
    """
    return max(1e-13, 1e-14 * 3.0 ** n)


# ------------------------------------------------------------- MATHEMATICS
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 11])
def test_symbolic_field_equals_explicit(n):
    r = np.linspace(0, 1, 20001)
    a, b = _explicit(n).field(r), cantor_field(r, n, E0)
    scale = max(1.0, float(np.abs(a).max()))
    assert np.abs(a - b).max() / scale < cond_tol(n)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 11])
def test_symbolic_potential_equals_explicit(n):
    r = np.linspace(0, 1, 20001)
    a, b = _explicit(n).potential(r), cantor_potential(r, n, E0)
    assert np.abs(a - b).max() < 1e-10


@pytest.mark.parametrize("n", [1, 3, 6, 10])
def test_potential_total_action_is_theorem_A(n):
    # V(1) = n*E0 -- the offset V3 dropped (defect D4) shows up here first.
    assert abs(float(cantor_potential([1.0], n, E0)[0]) - total_action(n, E0)) < 1e-12


def test_theorem_S_scale_identity_exact():
    n = 10
    r = np.linspace(1e-9, 1 - 1e-9, 5000)
    for T in (lambda x: x / 3.0, lambda x: 2.0 / 3.0 + x / 3.0):
        lhs = cantor_field(T(r), n, E0)
        rhs = 1.5 * cantor_field(r, n - 1, E0)
        denom = np.maximum(np.abs(rhs), 1.0)
        assert (np.abs(lhs - rhs) / denom).max() < cond_tol(n)


def test_potential_offset_identity():
    # V_n(T_i(r)) - V_n(T_i(0)) = (1/2) V_{n-1}(r)
    n = 9
    r = np.linspace(0, 1, 2000)
    for T in (lambda x: x / 3.0, lambda x: 2.0 / 3.0 + x / 3.0):
        lhs = cantor_potential(T(r), n, E0) - cantor_potential([T(0.0)], n, E0)[0]
        rhs = 0.5 * cantor_potential(r, n - 1, E0)
        assert np.abs(lhs - rhs).max() < 1e-10


@pytest.mark.parametrize("k", [1, 2, 3, 5, 7])
def test_theorem_B_and_T_from_general_formulas(k):
    assert abs(CANTOR.peak_of_level(k, E0) - peak_of_level(k, E0)) < 1e-9
    assert abs(CANTOR.slope_of_level(k, E0) - slope_of_level(k, E0)) < 1e-6


def test_measured_peak_matches_theorem_B():
    n = 6
    for k in range(1, n + 1):
        gaps = [g for g in cantor_gap_list(n) if g.level == k]
        r = np.concatenate([np.linspace(g.a, g.b, 2001) for g in gaps[:2]])
        assert abs(cantor_field(r, n, E0).max() - peak_of_level(k, E0)) < 1e-6


def test_level_detection_correct():
    n = 7
    L = _explicit(n)
    r = np.random.default_rng(0).uniform(0, 1, 3000)
    lv = cantor_level(r, n)
    for x, k in zip(r, lv):
        hit = [g for g in L.gaps if g.a <= x < g.b]
        assert (k == hit[0].level) if hit else (k == 0)


def test_generalised_formulas_reduce_to_cantor():
    for k in range(1, 8):
        assert CANTOR.n_gaps_at_level(k) == 2 ** (k - 1)
        assert abs(CANTOR.gap_width(k) - 3.0 ** -k) < 1e-15
    assert abs(CANTOR.alpha_field - 1.5) < 1e-15
    assert abs(CANTOR.alpha_sensitivity - 4.5) < 1e-15


@pytest.mark.parametrize("n", [1, 3, 6, 9])
def test_general_ifs_matches_symbolic_cantor(n):
    r = np.linspace(0, 1, 5000)
    a = SymbolicIFS(CANTOR, n, E0).field(r)
    b = cantor_field(r, n, E0)
    scale = max(1.0, float(np.abs(b).max()))
    assert np.abs(a - b).max() / scale < cond_tol(n)


def test_infeasible_ifs_rejected():
    assert not feasible(2, 0.6)          # b*rho = 1.2 > 1, images overlap
    with pytest.raises(ValueError):
        IFSSpec(2, 0.6)


def test_recursive_non_cantor_has_different_scale_law():
    s = IFSSpec(2, 0.28)
    assert abs(s.alpha_field - 1.0 / 0.56) < 1e-12
    assert abs(s.alpha_field - CANTOR.alpha_field) > 0.2   # genuinely different


# ------------------------------------------------------------ BOUNDARY CASES
def test_exact_ternary_endpoints():
    """Cantor endpoints are the numerically nasty points (the V1 erratum)."""
    n = 8
    L = _explicit(n)
    pts = [Fraction(0), Fraction(1), Fraction(1, 3), Fraction(2, 3),
           Fraction(1, 9), Fraction(2, 9), Fraction(7, 9), Fraction(8, 9),
           Fraction(1, 27), Fraction(26, 27)]
    for q in pts:
        x = float(q)
        assert abs(float(L.field(x)[0]) - float(cantor_field(x, n, E0)[0])) < 1e-9


@pytest.mark.parametrize("eps", [1e-15, 1e-12, 1e-9])
def test_adversarial_near_endpoints(eps):
    n = 8
    L = _explicit(n)
    base = [1 / 3, 2 / 3, 1 / 9, 2 / 9, 7 / 9, 8 / 9]
    r = np.array([b + s * eps for b in base for s in (-1, 1)])
    a, b_ = L.field(r), cantor_field(r, n, E0)
    scale = max(1.0, float(np.abs(a).max()))
    assert np.abs(a - b_).max() / scale < cond_tol(n)


def test_differential_random_large():
    """100k random points, harness section 50."""
    rng = np.random.default_rng(20260831)
    for n in (5, 10, 15):
        r = rng.uniform(0, 1, 100_000)
        a, b = _explicit(n).field(r), cantor_field(r, n, E0)
        scale = max(1.0, float(np.abs(a).max()))
        assert np.abs(a - b).max() / scale < cond_tol(n)


# -------------------------------------------------------------- STRUCTURAL
@pytest.mark.parametrize("n", range(1, 13))
def test_component_count(n):
    assert N_GAPS(n) == 2 ** n - 1 == len(cantor_gap_list(n))
    assert CANTOR.n_components(n) == 2 ** n - 1


def test_canonical_encoding_deterministic():
    a = describe("cantor_recursive", 7)
    b = describe("cantor_recursive", 7)
    assert a.canonical_bits == b.canonical_bits
    assert a.serialized_bytes == b.serialized_bytes


def test_recursive_description_independent_of_depth():
    bits = {describe("cantor_recursive", n).canonical_bits for n in range(1, 20)}
    assert len(bits) == 1          # depth is a field, not a payload


def test_explicit_description_grows_exponentially():
    b8 = describe("shuffled_explicit", 8).canonical_bits
    b12 = describe("shuffled_explicit", 12).canonical_bits
    assert b12 / b8 > 15           # 2^4 = 16-fold component growth


def test_seeded_shuffle_is_also_short():
    """STOP B: the honest control. A seeded shuffle describes in O(1) too, so
    description length alone is NOT evidence of a Cantor-specific advantage."""
    c = describe("cantor_recursive", 12).canonical_bits
    s = describe("shuffled_seeded", 12, seed=1).canonical_bits
    assert s < c                   # it is in fact SHORTER


def test_description_uses_same_codec_for_all_families():
    for f in FAMILIES:
        kw = {"seed": 1}
        if f == "recursive_non_cantor":
            kw["ifs"] = IFSSpec(2, 0.28)
        d = describe(f, 6, **kw)
        assert d.canonical_bits > 0 and d.n_components == 2 ** 6 - 1


def test_symbolic_storage_does_not_grow_with_depth():
    words = {SymbolicCantor(n, E0).storage_words() for n in range(1, 20)}
    assert words == {2}


# ------------------------------------------------------------ CERTIFICATION
@pytest.mark.parametrize("n", [2, 5, 10, 14])
def test_valid_cantor_certificate_passes(n):
    c = build_certificate("cantor_recursive", n)
    r = verify_certificate(c)
    assert r["ok"] and r["holds"]["P6_scale_identity"]


def test_recursive_non_cantor_passes_its_own_certificate():
    c = build_certificate("recursive_non_cantor", 8, spec=IFSSpec(2, 0.28))
    r = verify_certificate(c)
    assert r["ok"] and r["holds"]["P6_scale_identity"]


@pytest.mark.parametrize("kind", ["coordinate", "width", "level", "missing",
                                  "overlap"])
def test_corrupted_layout_rejected(kind):
    n = 8
    g = corrupt(cantor_gap_list(n), kind)
    c = build_certificate("shuffled_seeded", n, seed=3)
    assert not verify_certificate(c, gaps=g)["ok"]


def test_shuffled_does_not_falsely_pass_scale_identity():
    n = 8
    c = build_certificate("shuffled_seeded", n, seed=3)
    r = verify_certificate(c, gaps=cantor_gap_list(n))
    assert r["holds"]["P6_scale_identity"] is False


def test_inductive_scheme_visits_O_n_not_O_2n():
    for n in (6, 10, 14):
        c = build_certificate("cantor_recursive", n)
        assert verify_certificate(c)["visited"] <= 4 * n


def test_enumerative_scheme_visits_every_component():
    n = 9
    c = build_certificate("shuffled_seeded", n, seed=3)
    r = verify_certificate(c, gaps=cantor_gap_list(n))
    assert r["visited"] >= 2 ** n - 1


def test_permutation_invariant_properties_are_cheap_for_shuffled_too():
    """The fairness check. Properties depending only on the multiset must NOT
    be counted as expensive for a permuted layout."""
    n = 10
    cc = build_certificate("cantor_recursive", n)
    cs = build_certificate("shuffled_seeded", n, seed=3)
    for p in PERMUTATION_INVARIANT:
        assert cs.obligations[p] <= cc.obligations[p] + 1


def test_certificate_digest_stable():
    a = build_certificate("cantor_recursive", 9)
    b = build_certificate("cantor_recursive", 9)
    assert a.digest() == b.digest()


# ------------------------------------------------------------------ PARETO
def test_equivalence_gate_precedes_structural_dominance():
    from cantor_guard_v33.pareto import dominates
    # safety outside SESOI must block dominance no matter how cheap
    assert not dominates(r_a=0.10, r_b=0.20, c_a=1.0, c_b=1e9, sesoi=0.03)
    # inside SESOI and strictly cheaper -> dominates
    assert dominates(r_a=0.199, r_b=0.20, c_a=1.0, c_b=2.0, sesoi=0.03)
    # equal cost is not dominance
    assert not dominates(r_a=0.20, r_b=0.20, c_a=2.0, c_b=2.0, sesoi=0.03)


def _exact_field(rq: Fraction, n: int, E0v=1) -> Fraction:
    """Reference evaluator in exact rational arithmetic. Slow; used as ground
    truth so neither float implementation is treated as the standard."""
    p, pw = 0, 1
    for k in range(1, n + 1):
        pw3 = pw * 3
        a, b = Fraction(3 * p + 1, pw3), Fraction(3 * p + 2, pw3)
        if rq < a:
            p, pw = 3 * p, pw3
        elif rq >= b:
            p, pw = 3 * p + 2, pw3
        else:
            w = Fraction(1, pw3)
            u = (rq - a) / w
            return Fraction(E0v, 2 ** (k - 1)) / w * 6 * u * (1 - u)
    return Fraction(0)


@pytest.mark.parametrize("n", [6, 10, 14])
def test_symbolic_no_worse_than_explicit_against_exact_reference(n):
    """STOP D. The symbolic evaluator must not be a numerical downgrade, or the
    inherited safety equivalence would not transfer."""
    rng = np.random.default_rng(11)
    qs = [Fraction(int(rng.integers(1, 2 ** 40)), 2 ** 40) for _ in range(200)]
    ref = np.array([float(_exact_field(q, n)) for q in qs])
    rr = np.array([float(q) for q in qs])
    ex = _explicit(n).field(rr)
    sy = cantor_field(rr, n, E0)
    scale = max(1.0, float(np.abs(ref).max()))
    e_ex = float(np.abs(ex - ref).max()) / scale
    e_sy = float(np.abs(sy - ref).max()) / scale
    assert e_sy < cond_tol(n)
    assert e_sy <= e_ex * 3.0 + 1e-15      # not materially worse conditioned
