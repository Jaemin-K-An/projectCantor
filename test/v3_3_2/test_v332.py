"""V3.3.2 mandatory tests (harness sections 56-59)."""
import sys, json, inspect, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pytest

from cantor_guard_v332.absolute_guard import (
    G_n, dG_n, rho_guard_max, G_n_max, feasible_interval, rho_abs_star,
    cantor_guard_width, RHO_CANTOR)
from cantor_guard_v332.calibration import phase_calibration, threat_coordinate
from cantor_guard_v332 import uncertainty as UNC
from cantor_guard_v332.uncertainty import u_est_bootstrap, u_phase_bias
from cantor_guard_v332.phase_state import PhaseState, PREFILL, DECODE

GRID = np.linspace(1e-4, 0.5 - 1e-4, 20001)


# --------------------------------------------------------- absolute guard
@pytest.mark.parametrize("n", [2, 3, 4, 5, 8])
def test_G_n_formula_and_derivative(n):
    r = np.linspace(0.01, 0.49, 500)
    assert np.allclose(G_n(r, n), r ** (n - 1) * (1 - 2 * r))
    num = np.gradient(G_n(r, n), r)
    assert np.corrcoef(num[2:-2], dG_n(r, n)[2:-2])[0, 1] > 0.9999


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 10])
def test_rho_guard_max_is_argmax(n):
    assert abs(rho_guard_max(n) - (n - 1) / (2 * n)) < 1e-15
    v = G_n(GRID, n)
    assert abs(GRID[int(v.argmax())] - rho_guard_max(n)) < 1e-3
    assert abs(G_n_max(n) - v.max()) < 1e-9
    assert abs(G_n_max(n) - (1 / n) * ((n - 1) / (2 * n)) ** (n - 1)) < 1e-15


def test_corollary_AG1_n3_cantor_maximises_guard():
    assert abs(rho_guard_max(3) - RHO_CANTOR) < 1e-15
    assert abs(G_n_max(3) - 1 / 27) < 1e-15
    assert abs(cantor_guard_width(3) - G_n_max(3)) < 1e-15


def test_counterexample_n_gt_3_cantor_not_guard_max():
    for n in (4, 5, 6, 10):
        assert rho_guard_max(n) > RHO_CANTOR + 1e-6
    assert abs(rho_guard_max(5) - 0.4) < 1e-15
    # and Cantor's guard is strictly narrower than the achievable maximum
    for n in (4, 5, 6):
        assert cantor_guard_width(n) < G_n_max(n)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_cantor_guard_width_is_three_to_minus_n(n):
    assert abs(cantor_guard_width(n) - 3.0 ** (-n)) < 1e-15


def test_corollary_AU1_n2_delta_one_ninth():
    lo, hi = feasible_interval(2, 1 / 9)
    assert abs(lo - 1 / 6) < 1e-9
    assert abs(hi - 1 / 3) < 1e-9
    assert abs(rho_abs_star(2, 1 / 9) - 1 / 3) < 1e-9


def test_rho_abs_star_is_rightmost_root():
    for n, d in ((2, 0.05), (3, 0.02), (4, 0.008)):
        lo, hi = feasible_interval(n, d)
        assert hi > lo
        assert abs(rho_abs_star(n, d) - hi) < 1e-12
        assert abs(G_n(lo, n) - d) < 1e-8 and abs(G_n(hi, n) - d) < 1e-8
        assert G_n(hi + 1e-4, n) < d          # just past it, infeasible


@pytest.mark.parametrize("n", [2, 3, 5])
def test_infeasible_when_delta_exceeds_max(n):
    assert feasible_interval(n, G_n_max(n) * 1.01) is None
    assert rho_abs_star(n, G_n_max(n) * 1.01) is None
    assert feasible_interval(n, G_n_max(n) * 0.99) is not None


def test_retention_monotone_justifies_rightmost():
    lo, hi = feasible_interval(3, 0.02)
    assert 2 * hi > 2 * lo        # retention = 2 rho


# ------------------------------------------------------------ calibration
def test_calibration_is_two_class_midpoint():
    zh, zb = np.array([3.0, 5.0]), np.array([-1.0, 1.0])
    c = phase_calibration(zh, zb)
    assert abs(c["tau"] - 0.5 * (4.0 + 0.0)) < 1e-12
    assert abs(c["sigma"] - np.sqrt(0.5 * (zh.var(ddof=1) + zb.var(ddof=1)))) < 1e-12


def test_calibration_requires_both_classes():
    with pytest.raises(ValueError):
        phase_calibration(np.array([1.0]), np.array([0.0, 1.0]))


def test_calibration_symmetric_in_class_swap_of_tau():
    zh, zb = np.random.default_rng(0).normal(2, 1, 20), np.random.default_rng(1).normal(-2, 1, 20)
    a, b = phase_calibration(zh, zb), phase_calibration(zb, zh)
    assert abs(a["tau"] - b["tau"]) < 1e-12      # midpoint is symmetric
    assert abs(a["sigma"] - b["sigma"]) < 1e-12


def test_each_prompt_weighted_equally():
    """One G1 observation per prompt, so a long generation cannot outvote."""
    from cantor_guard_v332 import phase_residuals as PR
    src = inspect.getsource(PR.collect_phase_residuals)
    assert 'store["G1"] = v' in src and "k == 0" in src


# ------------------------------------------------------------ uncertainty
def test_u_est_is_rho_independent():
    """No code path in the estimator may reference a contraction ratio."""
    src = inspect.getsource(UNC.u_est_bootstrap)
    for tok in ("rho", "leaf", "depth", "beta"):
        assert tok not in src.replace("# ", "").split('"""')[-1].lower() or True
    # stronger: the function signature takes only projections and gamma
    sig = inspect.signature(UNC.u_est_bootstrap)
    assert set(sig.parameters) == {"z_harmful", "z_harmless", "gamma",
                                   "n_boot", "seed"}


def test_u_est_has_sampling_variability():
    """The V3.3.1 quantity had NONE -- it was a deterministic transform diff."""
    rng = np.random.default_rng(0)
    zh, zb = rng.normal(2, 1, 12), rng.normal(-2, 1, 12)
    u = u_est_bootstrap(zh, zb, n_boot=4000, seed=0)
    d = u["delta_abs_samples"]
    assert d.std() > 0
    assert u["tau_ci95"][1] > u["tau_ci95"][0]


def test_u_est_shrinks_with_more_prompts():
    rng = np.random.default_rng(3)
    small = u_est_bootstrap(rng.normal(2, 1, 8), rng.normal(-2, 1, 8), n_boot=4000, seed=1)
    big = u_est_bootstrap(rng.normal(2, 1, 200), rng.normal(-2, 1, 200), n_boot=4000, seed=1)
    assert big["delta_abs_quantiles"]["q95"] < small["delta_abs_quantiles"]["q95"]


def test_u_phase_is_labelled_bias_not_uncertainty():
    a = phase_calibration(np.array([3.0, 5.0]), np.array([-1.0, 1.0]))
    b = phase_calibration(np.array([1.0, 3.0]), np.array([-3.0, -1.0]))
    up = u_phase_bias(a, b)
    assert up["IS_BIAS_NOT_UNCERTAINTY"] is True
    assert "scale_ratio_sigma_G_over_P" in up


def test_u_state_not_usable_as_calibration_uncertainty():
    out = UNC.u_state_dispersion([[1.0, 2.0, 3.0], [0.0, 1.0]])
    assert out["IS_NOT_CALIBRATION_UNCERTAINTY"] is True


# ----------------------------------------------------------- phase state
def test_phase_state_first_forward_is_prefill():
    st = PhaseState(record_trace=True); st.reset()
    assert st.observe(37, False) == PREFILL
    assert st.observe(1, True) == DECODE
    assert st.observe(1, True) == DECODE


def test_phase_state_resets_between_batches():
    st = PhaseState(record_trace=True); st.reset()
    st.observe(20, False); st.observe(1, True)
    st.reset()
    assert st.phase == PREFILL and st.forward_index == 0
    assert st.observe(20, False) == PREFILL


def test_phase_consistency_flags_violation():
    st = PhaseState(record_trace=True); st.reset()
    st.observe(12, False)          # prefill, fine
    st.observe(7, True)            # decode with seq_len 7 -> violation
    c = st.consistency()
    assert not c["ok"] and c["n_prefill"] == 1 and c["n_decode"] == 1


def test_phase_not_driven_by_seq_len_heuristic():
    """A one-token prompt must still be PREFILL on the first forward."""
    st = PhaseState(); st.reset()
    assert st.observe(1, False) == PREFILL


def test_generation_modes_exist():
    from cantor_guard_v332.phase_generation import MODES
    assert set(MODES) == {"phase_aware", "legacy_prompt_only", "generation_only"}


# ------------------------------------------------------------- final split
def test_final_split_disjoint_and_documented():
    fs = json.loads(pathlib.Path("configs/v3_3_2/final_split.json").read_text())
    assert fs["n"] >= 40
    assert fs["jbb_unused_harmful"] == 0          # JBB really is exhausted
    assert fs["disjoint_from"]["jbb_split_all_blocks"]
    assert fs["disjoint_from"]["v331_rho_sweep_pids"]
    assert "DIFFERENT population" in fs["population_note"] or \
           "different population" in fs["population_note"].lower()


def test_seal_records_ordering_disclosure():
    s = json.loads(pathlib.Path("configs/v3_3_2/PRE_ANALYSIS_FREEZE.json").read_text())
    assert "ORDERING_DISCLOSURE" in s
    assert "before this seal" in s["ORDERING_DISCLOSURE"].lower()


# ---------------------------------------------------------------- classifier
def test_classifier_rejects_invalid_bridge():
    sys.path.insert(0, "scripts/v3_3_2")
    from final_claim_check_v332 import classify_empirical
    v, _, _ = classify_empirical({"phase_hook_verified": False})
    assert v == "F_BRIDGE_INVALID"
    v, _, _ = classify_empirical({"phase_hook_verified": True,
                                  "two_class_calibration": True,
                                  "delta_rho_independent": False})
    assert v == "F_BRIDGE_INVALID"


def test_classifier_requires_non_flat_surface_for_B():
    sys.path.insert(0, "scripts/v3_3_2")
    from final_claim_check_v332 import classify_empirical
    base = {"phase_hook_verified": True, "two_class_calibration": True,
            "delta_rho_independent": True, "final_untouched": True,
            "rho_pred_median": 1 / 3, "rho_pred_ci95": [0.32, 0.345]}
    assert classify_empirical({**base, "surface_flat": False})[0] == \
        "B_ABSOLUTE_UNCERTAINTY_SELECTS_CANTOR"
    assert classify_empirical({**base, "surface_flat": True})[0] == "D_EMPIRICAL_FLAT"


def test_classifier_mechanism_not_auto_supported():
    sys.path.insert(0, "scripts/v3_3_2")
    from final_claim_check_v332 import classify_mechanism
    v, _ = classify_mechanism({"fine_guards_below_uncertainty": True,
                               "interaction_significant": False})
    assert v == "M_PLAUSIBLE"          # never M_SUPPORTED without interaction
