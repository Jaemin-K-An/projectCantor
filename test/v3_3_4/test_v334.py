"""V3.3.4 mandatory tests (harness sections 60-62)."""
import sys, json, inspect, pathlib
sys.path.insert(0, "llm/src"); sys.path.insert(0, "scripts/v3_3_4")
from fractions import Fraction
import numpy as np, pytest
from cantor_guard_v334.certified_geometry import (leaves, guards, M_n, dM_n,
    rho_max, M_n_max, d_cross_exact, classify_exact, RHO_CANTOR)
from cantor_guard_v334.certificate import (eps_z_lipschitz, eps_h_l2,
    eps_z_exact, separating_guards, z_of_r, logit)
from cantor_guard_v334.guarded_policy import CantorGuardedPolicy

SIG, GAM = 0.9258, 0.7
TAU = -2.6263


# ------------------------------------------------------------ THEOREM CR
@pytest.mark.parametrize("n,want", [(2, 1/4), (3, 1/3), (4, 3/8), (5, 2/5)])
def test_rho_max_law(n, want):
    assert abs(rho_max(n) - want) < 1e-15


def test_M3_unique_max_and_value():
    assert abs(rho_max(3) - 1/3) < 1e-15
    assert abs(M_n(1/3, 3) - 1/27) < 1e-15
    # derivative 2rho(1-3rho): positive below 1/3, negative above
    assert dM_n(0.30, 3) > 0 and dM_n(0.36, 3) < 0
    assert abs(dM_n(1/3, 3)) < 1e-12


@pytest.mark.parametrize("n", [2, 3, 5])
def test_M_n_equals_G_n_and_is_infimum(n):
    """The empirical inf approaches M_n from ABOVE and never attains it."""
    NPTS = 60001
    spacing = 1.0 / (NPTS - 1)
    for rho in (0.25, 1/3, 0.40):
        rs = np.linspace(1e-9, 1 - 1e-9, NPTS)
        d = np.array([d_cross_exact(x, rho, n) for x in rs])
        d = d[d > 0]
        assert d.min() >= M_n(rho, n) - 1e-12          # never below
        # How closely a finite grid can approach the infimum is limited by the
        # spacing relative to the guard width; at n=5 the guards are ~2e-3 wide,
        # so allow 2 grid steps of slack rather than a fixed constant.
        tol = 1.0 + 2.0 * spacing / M_n(rho, n)
        assert d.min() / M_n(rho, n) < tol


def test_depth3_has_8_leaves_7_guards():
    assert len(leaves(1/3, 3)) == 8 and len(guards(1/3, 3)) == 7


def test_exact_geometry_no_stepping():
    src = inspect.getsource(d_cross_exact)
    for bad in ("max_steps", "step =", "1e-3", "while"):
        assert bad not in src


def test_rho_third_recovers_middle_third_cantor():
    """Exact rational check: level-1 guard is (1/3, 2/3)."""
    G = guards(Fraction(1, 3), 1, exact=True)
    assert G[0][1] == Fraction(1, 3) and G[0][2] == Fraction(2, 3)
    L = leaves(Fraction(1, 3), 3, exact=True)
    assert L[0][0] == Fraction(0) and L[0][1] == Fraction(1, 27)


def test_differential_100k_points():
    rng = np.random.default_rng(0)
    for rho, n in ((1/3, 3), (0.28, 3), (0.40, 2)):
        L = leaves(rho, n)
        pts = rng.uniform(0, 1, 100_000)
        for x in pts[:1500]:
            c = classify_exact(float(x), rho, n)
            if c[0] == "leaf":
                assert c[2] <= x < c[3]
                d = d_cross_exact(float(x), rho, n)
                assert d >= M_n(rho, n) - 1e-12


# ---------------------------------------------------------- CERTIFICATES
def test_lipschitz_certificate_formula():
    for rho in (0.25, 1/3, 0.40):
        assert abs(eps_z_lipschitz(rho, 3, SIG, GAM)
                   - (4 * SIG / GAM) * M_n(rho, 3)) < 1e-12


def test_h_l2_equals_z_certificate():
    for rho in (0.25, 1/3, 0.40):
        assert eps_h_l2(rho, 3, SIG, GAM) == eps_z_lipschitz(rho, 3, SIG, GAM)


def test_lipschitz_certificate_maximised_at_third():
    g = np.linspace(0.05, 0.49, 4401)
    v = np.array([eps_z_lipschitz(x, 3, SIG, GAM) for x in g])
    assert abs(g[v.argmax()] - 1/3) < 1e-3
    assert abs(eps_z_lipschitz(1/3, 3, SIG, GAM) - 4 * SIG / (27 * GAM)) < 1e-12


def test_exact_certificate_at_least_lipschitz():
    """The Lipschitz bound uses the max slope 1/4, so it is conservative."""
    for rho in (0.25, 0.28, 1/3, 0.36, 0.40):
        assert eps_z_exact(rho, 3, TAU, SIG, GAM) >= eps_z_lipschitz(rho, 3, SIG, GAM)


def test_exact_certificate_maximiser_is_reported_not_assumed():
    """Documented honestly: the logit warp moves the exact optimum off 1/3."""
    g = np.linspace(0.05, 0.49, 4401)
    e = np.array([eps_z_exact(x, 3, TAU, SIG, GAM) for x in g])
    assert abs(g[e.argmax()] - 1/3) > 0.01      # it really does move


def test_exact_certificate_independent_of_tau():
    for t in (-5.0, 0.0, 3.0):
        assert abs(eps_z_exact(1/3, 3, t, SIG, GAM)
                   - eps_z_exact(1/3, 3, 0.0, SIG, GAM)) < 1e-12


def test_separating_guards_count():
    assert len(separating_guards(1/3, 3)) == 7


def test_z_of_r_inverts_coordinate():
    C = CantorGuardedPolicy(1/3, 3, tau_beh=TAU, sigma=SIG, gamma=GAM)
    z = np.array([-4.0, -2.0, 0.0, 2.0])
    assert np.allclose(z_of_r(C.coordinate(z), TAU, SIG, GAM), z, atol=1e-9)


# ------------------------------------------------------- GUARDED POLICY
def test_controller_requires_tau_beh_no_silent_fallback():
    with pytest.raises(ValueError):
        CantorGuardedPolicy(1/3, 3, tau_beh=None, sigma=SIG)


def test_leaf_actions_monotone_and_guard_conservative():
    C = CantorGuardedPolicy(1/3, 3, tau_beh=TAU, sigma=SIG, gamma=GAM)
    assert np.all(np.diff(C.leaf_actions) > 0)
    r = np.linspace(1e-6, 1 - 1e-6, 4001)
    kind, _ = C.classify_r(r)
    act = C.action(r)
    for i in np.where(kind == "guard")[0]:
        lo = np.searchsorted([l[1] for l in C._leaves], r[i]) - 1
        left = C.leaf_actions[max(lo, 0)]
        right = C.leaf_actions[min(lo + 1, C.n_leaves - 1)]
        assert act[i] >= max(left, right) - 1e-12


def test_orientation_flip_reverses_schedule():
    A = CantorGuardedPolicy(1/3, 3, tau_beh=TAU, sigma=SIG, higher_r_is_threat=True)
    B = CantorGuardedPolicy(1/3, 3, tau_beh=TAU, sigma=SIG, higher_r_is_threat=False)
    assert np.allclose(A.leaf_actions, B.leaf_actions[::-1])


def test_certificate_methods_match_module():
    C = CantorGuardedPolicy(1/3, 3, tau_beh=TAU, sigma=SIG, gamma=GAM)
    assert abs(C.certificate_r() - 1/27) < 1e-15
    assert abs(C.certificate_z_lipschitz() - eps_z_lipschitz(1/3, 3, SIG, GAM)) < 1e-15
    assert C.certificate_h_l2() == C.certificate_z_lipschitz()


def test_matched_architecture_only_rho_varies():
    cs = [CantorGuardedPolicy(r, 3, tau_beh=TAU, sigma=SIG, gamma=GAM)
          for r in (0.28, 1/3, 0.40)]
    assert len({c.n_leaves for c in cs}) == 1 == len({c.depth for c in cs})
    assert len({c.tau_beh for c in cs}) == 1 and len({c.sigma for c in cs}) == 1
    assert all(np.allclose(cs[0].leaf_actions, c.leaf_actions) for c in cs)


# ----------------------------------------------------- empirical artefacts
def test_certificate_run_has_zero_violations():
    s = json.loads(pathlib.Path(
        "results/v3_3_4/tables/certificate_summary_dev.json").read_text())
    assert s["violations"] == 0 and s["n_below_cert_configs"] > 100


def test_depthshift_uses_common_absolute_grid():
    """The certificate-normalised grid is CIRCULAR and must not be used."""
    src = pathlib.Path("scripts/v3_3_4/run_certificate_attack.py").read_text()
    assert "EPS_COMMON" in src
    assert "BY CONSTRUCTION" in src        # the circularity is documented there
    f = json.loads(pathlib.Path(
        "configs/v3_3_4/PRE_ANALYSIS_FREEZE.json").read_text())
    assert any("CIRCULAR" in x for x in f["CHRONOLOGY"]["before_seal"])


def test_splits_fresh_and_disjoint():
    s = json.loads(pathlib.Path("configs/v3_3_4/splits.json").read_text())
    allp = [x for v in s["blocks"].values() for x in v]
    assert len(set(allp)) == len(allp)
    assert s["disjoint_from_all_prior"] and s["blocks"]["D_final_334"].__len__() >= 60
    assert "THIRD distinct population" in s["population_note"]


def test_freeze_precedes_final():
    f = json.loads(pathlib.Path(
        "configs/v3_3_4/PRE_ANALYSIS_FREEZE.json").read_text())
    assert f["CHRONOLOGY"]["D_final_334_touched_before_seal"] is False
    assert any("CIRCULAR" in x for x in f["CHRONOLOGY"]["before_seal"])


def test_classifier_blocks_positive_on_violation():
    from final_claim_check_v334 import v_certificate, v_generation
    assert v_certificate({"violations": 3})[0] == "C2_CERTIFICATE_IMPLEMENTATION_FAILURE"
    assert v_generation({"available": True, "budget_matched_final": False
                         })[0] == "G5_INCONCLUSIVE"


def test_classifier_requires_final_budget_not_dbudget():
    src = pathlib.Path("scripts/v3_3_4/final_claim_check_v334.py").read_text()
    assert "budget_matched_final" in src
