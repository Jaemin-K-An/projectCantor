"""V3.3.5 mandatory tests (harness sections 57-64)."""
import sys, json, pathlib
sys.path.insert(0, "llm/src"); sys.path.insert(0, "scripts/v3_3_5")
import numpy as np, pytest
from cantor_guard_v335.affine_coordinate import AffineCoordinate, choose_W, OUTSIDE
from cantor_guard_v335.certificate import (eps_z_affine, eps_h_affine,
                                           cantor_gain_table, logistic_exact)
from cantor_guard_v335.affine_guarded_policy import AffineCantorGuardedPolicy
from cantor_guard_v335.g1_only_generation import G1Trace, PREFILL, G1, G2PLUS
from cantor_guard_v334.certified_geometry import M_n, rho_max
W, TAU = 2.0, 0.5


# ------------------------------------------------------------- THEOREM CP
def test_affine_slope_is_constant():
    C = AffineCoordinate(TAU, W)
    z = np.linspace(TAU - W, TAU + W, 501)
    r = C.r(z)
    d = np.abs(np.diff(r)) / np.abs(np.diff(z))
    assert np.allclose(d, C.slope()) and abs(C.slope() - 1 / (2 * W)) < 1e-15


def test_logistic_slope_is_NOT_constant():
    """Theorem CP's hypothesis FAILS for the logistic map, which is exactly why
    V3.3.4's exact optimum drifted off 1/3.

    The variation grows with the window: over +-3 the slope ratio is ~2.9, over
    +-6 it is ~24. The affine map's ratio is exactly 1 at any width, so the
    comparison is asserted against the affine baseline rather than a magic
    constant.
    """
    g, s = 0.7, 0.9258
    for span, floor in ((3.0, 2.0), (6.0, 10.0)):
        z = np.linspace(-span, span, 800)
        r = 1 / (1 + np.exp(np.clip(g * z / s, -60, 60)))
        d = np.abs(np.diff(r)) / np.abs(np.diff(z))
        assert d.max() / d.min() > floor
    # the affine map, on any window, has ratio exactly 1
    C = AffineCoordinate(TAU, W)
    zz = np.linspace(TAU - W, TAU + W, 800)
    da = np.abs(np.diff(C.r(zz))) / np.abs(np.diff(zz))
    assert abs(da.max() / da.min() - 1.0) < 1e-9


def test_affine_endpoints_and_centre_exact():
    C = AffineCoordinate(TAU, W, orientation=+1)
    assert abs(C.r(np.array([TAU - W]))[0] - 0.0) < 1e-15
    assert abs(C.r(np.array([TAU]))[0] - 0.5) < 1e-15
    assert abs(C.r(np.array([TAU + W]))[0] - 1.0) < 1e-15


def test_inverse_exact():
    C = AffineCoordinate(TAU, W)
    z = np.linspace(TAU - W, TAU + W, 200)
    assert np.allclose(C.z_of_r(C.r(z)), z, atol=1e-12)


def test_outside_is_nan_not_clipped():
    C = AffineCoordinate(TAU, W)
    assert np.all(~np.isfinite(C.r(np.array([TAU - 5 * W, TAU + 5 * W]))))


def test_orientation_flips_direction():
    a = AffineCoordinate(TAU, W, +1).r(np.array([TAU + 0.5 * W]))[0]
    b = AffineCoordinate(TAU, W, -1).r(np.array([TAU + 0.5 * W]))[0]
    assert abs(a + b - 1.0) < 1e-15


def test_choose_W_is_frozen_rule():
    z = np.random.default_rng(0).normal(TAU, 1.0, 5000)
    w = choose_W(z, TAU)
    assert abs(w - 1.05 * np.quantile(np.abs(z - TAU), 0.99)) < 1e-12


# ------------------------------------------------------- AFFINE CERTIFICATE
def test_eps_equals_2W_Mn():
    for rho in (0.25, 1 / 3, 0.40):
        for n in (2, 3, 5):
            assert abs(eps_z_affine(rho, n, W) - 2 * W * M_n(rho, n)) < 1e-15
    assert eps_h_affine(1 / 3, 3, W) == eps_z_affine(1 / 3, 3, W)


@pytest.mark.parametrize("n,want", [(2, 1 / 4), (3, 1 / 3), (5, 2 / 5)])
def test_exact_affine_optimum_follows_depth_law(n, want):
    g = np.linspace(0.05, 0.49, 8801)
    e = np.array([eps_z_affine(x, n, W) for x in g])
    assert abs(g[e.argmax()] - want) < 1e-3
    assert abs(rho_max(n) - want) < 1e-15


def test_cantor_value_is_2W_over_27():
    assert abs(eps_z_affine(1 / 3, 3, W) - 2 * W / 27) < 1e-15


def test_certificate_independent_of_eta():
    a = AffineCantorGuardedPolicy(1 / 3, 3, tau_g1=TAU, W=W, eta=0.1)
    b = AffineCantorGuardedPolicy(1 / 3, 3, tau_g1=TAU, W=W, eta=9.0)
    assert a.certificate_z_exact() == b.certificate_z_exact()


def test_cantor_ranks_first_at_depth3():
    t = cantor_gain_table(W)
    best = max(t, key=lambda r: r["eps_z_affine"])
    assert best["is_cantor"] and all(r["cantor_gain_pct"] >= 0 for r in t)


def test_logistic_historical_control_preserved():
    """V3.3.4's finding must remain visible, not quietly dropped."""
    g = np.linspace(0.05, 0.49, 4401)
    L = np.array([logistic_exact(x, 3, 0.9258, 0.7) for x in g])
    assert abs(g[L.argmax()] - 1 / 3) > 0.01


# ------------------------------------------------------------ AFFINE POLICY
def test_policy_requires_tau_g1():
    with pytest.raises(ValueError):
        AffineCantorGuardedPolicy(1 / 3, 3, tau_g1=None, W=W)


def test_policy_takes_h_directly_no_margin_api():
    C = AffineCantorGuardedPolicy(1 / 3, 3, tau_g1=TAU, W=W)
    assert not hasattr(C, "magnitude")          # the V3.3.4 reconstruction API
    assert hasattr(C, "intervene")
    h = np.random.default_rng(0).normal(size=(4, 6))
    v = np.zeros(6); v[0] = 1.0
    dh, mag = C.intervene(h, v)
    assert dh.shape == h.shape
    assert np.allclose(dh[:, 1:], 0.0)          # correction is along v only


def test_outside_gets_conservative_action():
    C = AffineCantorGuardedPolicy(1 / 3, 3, tau_g1=TAU, W=W)
    assert C.action(np.array([TAU + 10 * W]))[0] == C.a_max


def test_guard_action_at_least_both_neighbours():
    C = AffineCantorGuardedPolicy(1 / 3, 3, tau_g1=TAU, W=W)
    z = np.linspace(TAU - W + 1e-9, TAU + W - 1e-9, 3001)
    kind, _ = C.classify(z); act = C.action(z)
    r = C.coord.r(z)
    edges = np.array([l[1] for l in C._leaves])
    for i in np.where(kind == "guard")[0]:
        lo = int(np.searchsorted(edges, r[i])) - 1
        assert act[i] >= max(C.leaf_actions[max(lo, 0)],
                             C.leaf_actions[min(lo + 1, C.n_leaves - 1)]) - 1e-12


def test_leaf_actions_monotone_and_identical_across_rho():
    cs = [AffineCantorGuardedPolicy(r, 3, tau_g1=TAU, W=W)
          for r in (0.28, 1 / 3, 0.40)]
    assert all(np.all(np.diff(c.leaf_actions) > 0) for c in cs)
    assert all(np.allclose(cs[0].leaf_actions, c.leaf_actions) for c in cs)
    assert len({c.n_leaves for c in cs}) == 1 == len({c.W for c in cs})


def test_q_cap_enforced():
    C = AffineCantorGuardedPolicy(1 / 3, 3, tau_g1=TAU, W=W, eta=100.0, max_q=0.05)
    h = np.random.default_rng(1).normal(size=(5, 8))
    v = np.zeros(8); v[0] = 1.0
    _, mag = C.intervene(h, v)
    assert np.all(mag <= 0.05 * np.linalg.norm(h, axis=-1) + 1e-9)


# --------------------------------------------------------------- G1 PHASES
def test_g1_trace_phase_sequence():
    t = G1Trace(); t.reset()
    assert t.phase() == PREFILL; t.forward_index += 1
    assert t.phase() == G1; t.forward_index += 1
    assert t.phase() == G2PLUS; t.forward_index += 1
    assert t.phase() == G2PLUS


def test_g1_trace_resets():
    t = G1Trace(); t.forward_index = 7; t.reset()
    assert t.forward_index == 0 and t.phase() == PREFILL


def test_recorded_run_applied_only_at_g1():
    d = json.loads(pathlib.Path(
        "results/v3_3_5/tables/g1_boundary_D_beh_g1_confirm.json").read_text())
    assert d["g1_trace_verified"] is True


# ------------------------------------------------------- empirical artefacts
def test_certificate_zero_violations_real_forward():
    s = json.loads(pathlib.Path(
        "results/v3_3_5/tables/certificate_summary.json").read_text())
    assert s["violations"] == 0 and s["n_below_cert"] >= 40
    assert s["real_forward_attack"] is True
    assert s["max_dz_error"] < 1e-5          # pure directional attack: |dz| = eps
    assert s["cantor_rank_1"] is True


def test_anchor_disclosed_as_non_behavioural():
    s = json.loads(pathlib.Path(
        "results/v3_3_5/tables/certificate_summary.json").read_text())
    assert s["anchor_is_behavioural"] is False


def test_g1_boundary_unidentifiable_and_not_substituted():
    c = json.loads(pathlib.Path(
        "results/v3_3_5/tables/g1_boundary_D_beh_g1_confirm.json").read_text())
    assert c["status"] == "TAU_G1_UNIDENTIFIABLE"
    assert c["tau_g1"] is None               # withheld, not replaced


def test_dev_and_confirm_agree():
    dev = json.loads(pathlib.Path(
        "results/v3_3_5/tables/g1_boundary_D_beh_g1_dev.json").read_text())
    con = json.loads(pathlib.Path(
        "results/v3_3_5/tables/g1_boundary_D_beh_g1_confirm.json").read_text())
    assert dev["status"] == con["status"]
    assert abs(dev["isotonic"] - con["isotonic"]) < 5.0


def test_splits_fresh_and_final_untouched():
    s = json.loads(pathlib.Path("configs/v3_3_5/splits.json").read_text())
    allp = [x for v in s["blocks"].values() for x in v]
    assert len(set(allp)) == len(allp) and s["hash_overlap_with_all_prior"] == 0
    assert len(s["blocks"]["D_final_335"]) >= 80
    assert "FOURTH distinct population" in s["population_note"]
    f = json.loads(pathlib.Path(
        "configs/v3_3_5/PRE_ANALYSIS_FREEZE.json").read_text())
    assert f["CHRONOLOGY"]["D_final_335_touched"] is False


def test_classifier_cannot_reach_A_without_B1():
    from final_claim_check_v335 import v_behavior, overall, v_math, v_certificate
    vm = ("M1_CP_AND_M2_CANTOR_EXACT_MAXIMIN_PROVED", "")
    vb = v_behavior({"identified": False, "slope": 0.03, "ci_width_sigma": 24.0})
    vc = ("C1_EXACT_AFFINE_CERTIFICATE_VALIDATED", "")
    ov = overall(vm, vb, vc, ("G5_INCONCLUSIVE", ""), ("U_NOT_RUN", ""))
    assert ov[0] == "E_NO_APPLICABLE_BEHAVIORAL_CONTROLLER"
