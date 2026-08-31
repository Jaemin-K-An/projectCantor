"""V3.3.3 mandatory tests (harness section 42)."""
import sys, json, pathlib
sys.path.insert(0, "llm/src"); sys.path.insert(0, "scripts/v3_3_3")
import numpy as np, pytest
from cantor_guard_v332.absolute_guard import (G_n, dG_n, rho_guard_max, G_n_max,
                                              feasible_interval, rho_abs_star)
from cantor_guard_v333.behavioral_boundary import (fit_logistic, isotonic_crossing,
                                                   identifiability, tau_beh_bootstrap)
from cantor_guard_v333.adversarial_crossing import (cell_of, leaf_bounds,
                                                    d_cross_exact,
                                                    d_cross_to_other_leaf)

# ---------------------------------------------------------------- mathematics
@pytest.mark.parametrize("n,want", [(2, 1/4), (3, 1/3), (4, 3/8), (5, 2/5)])
def test_rho_max_values(n, want):
    assert abs(rho_guard_max(n) - want) < 1e-15


def test_G3_at_cantor_is_one_27():
    assert abs(G_n(1/3, 3) - 1/27) < 1e-15
    assert abs(G_n_max(3) - 1/27) < 1e-15


def test_cantor_not_maximiser_for_all_n():
    """Explicitly: 1/3 is special at n=3 only."""
    assert abs(rho_guard_max(3) - 1/3) < 1e-15
    for n in (2, 4, 5, 6, 10):
        assert abs(rho_guard_max(n) - 1/3) > 1e-3


@pytest.mark.parametrize("n", [2, 3, 5])
def test_derivative_matches_numeric(n):
    r = np.linspace(.02, .48, 400)
    num = np.gradient(G_n(r, n), r)
    assert np.corrcoef(num[3:-3], dG_n(r, n)[3:-3])[0, 1] > 0.9999


@pytest.mark.parametrize("n", [2, 3, 5])
def test_feasible_roots_and_infeasibility(n):
    d = G_n_max(n) * 0.6
    lo, hi = feasible_interval(n, d)
    assert abs(G_n(lo, n) - d) < 1e-8 and abs(G_n(hi, n) - d) < 1e-8
    assert rho_abs_star(n, d) == hi
    assert feasible_interval(n, G_n_max(n) * 1.02) is None


# ---------------------------------------------------------------- uncertainty
def test_quantiles_are_ordered():
    df = __import__("pandas").read_csv(
        "results/v3_3_3/tables/quantile_sensitivity.csv")
    for (u, n), g in df.groupby(["uncertainty", "n"]):
        q = g.set_index("quantile").delta
        assert q["q50"] <= q["q75"] <= q["q90"] <= q["q95"]


def test_both_uncertainty_definitions_reported():
    df = __import__("pandas").read_csv(
        "results/v3_3_3/tables/quantile_sensitivity.csv")
    assert set(df.uncertainty.unique()) == {"U_EST_mid", "U_EST_beh"}
    assert set(df["quantile"].unique()) == {"q50", "q75", "q90", "q95"}


# ------------------------------------------------------------------ behaviour
def test_logistic_recovers_known_boundary():
    rng = np.random.default_rng(0)
    z = np.linspace(-4, 6, 900)
    y = (rng.random(900) < 1 / (1 + np.exp(-1.8 * (z - 1.2)))).astype(float)
    a, b = fit_logistic(z, y)
    assert abs(-a / b - 1.2) < 0.35 and abs(b - 1.8) < 0.6


def test_isotonic_agrees_in_sign_with_logistic():
    rng = np.random.default_rng(1)
    z = np.linspace(-4, 6, 900)
    y = (rng.random(900) < 1 / (1 + np.exp(-1.8 * (z - 1.2)))).astype(float)
    a, b = fit_logistic(z, y)
    iso = isotonic_crossing(z, y)
    assert iso is not None and abs(iso - (-a / b)) < 1.5


@pytest.mark.parametrize("mode", ["all_refuse", "all_comply", "flat"])
def test_unidentifiable_cases_rejected(mode):
    rng = np.random.default_rng(2)
    z = np.linspace(-4, 6, 600)
    y = {"all_refuse": np.ones(600), "all_comply": np.zeros(600),
         "flat": (rng.random(600) < .5).astype(float)}[mode]
    a, b = fit_logistic(z, y)
    tau = -a / b if abs(b) > 1e-12 else np.nan
    g = identifiability(z, y, a, b, tau, dose_bins=np.repeat(np.arange(10), 60))
    assert not g["all_pass"]


def test_transition_observed_gate_is_strict():
    """A fit whose observed proportions never cross 0.5 must be rejected even
    if tau falls numerically inside the z range."""
    z = np.linspace(-6, 6, 600)
    y = (np.random.default_rng(3).random(600) < 0.85).astype(float)
    a, b = fit_logistic(z, y)
    g = identifiability(z, y, a, b, -a / b if abs(b) > 1e-12 else np.nan,
                        dose_bins=np.repeat(np.arange(10), 60))
    assert g["transition_observed"] is False and not g["all_pass"]


def test_tau_beh_bootstrap_is_prompt_clustered():
    rng = np.random.default_rng(4)
    pid = np.repeat(np.arange(40), 10)
    z = np.tile(np.linspace(-3, 5, 10), 40)
    y = (rng.random(400) < 1 / (1 + np.exp(-(z - 1)))).astype(float)
    R = tau_beh_bootstrap(z, y, pid, n_boot=800, seed=0)
    assert R["n_prompts"] == 40 and R["n_obs"] == 400
    assert R["tau_ci95"][0] < R["tau_hat"] < R["tau_ci95"][1]


def test_real_fit_is_identified_and_reports_gap():
    b = json.loads(pathlib.Path(
        "results/v3_3_3/tables/behavioral_boundary.json").read_text())
    assert b["status"] == "IDENTIFIED"
    assert b["identifiability"]["transition_observed"] is True
    assert abs(b["gap_tau_mid_minus_tau_beh_sigma"]) > 1.0   # midpoint IS biased


# ------------------------------------------------------------------ System B
def test_cell_geometry_hand_checked():
    # rho=1/3, n=1: leaf [0,1/3], guard [1/3,2/3], leaf [2/3,1]
    assert cell_of(0.1, 1/3, 1)[0] == "leaf"
    assert cell_of(0.5, 1/3, 1)[0] == "guard"
    assert abs(leaf_bounds(0.1, 1/3, 1)[1] - 1/3) < 1e-12
    assert d_cross_exact(0.5, 1/3, 1) == 0.0


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("rho", [0.25, 1/3, 0.40])
def test_min_crossing_never_below_guard_width(n, rho):
    """The theorem is an INEQUALITY: d_cross >= G_n, never violated."""
    rs = np.linspace(1e-6, 1 - 1e-6, 4001)
    d = np.array([d_cross_to_other_leaf(float(x), rho, n) for x in rs])
    it = np.array([cell_of(float(x), rho, n)[0] == "leaf" for x in rs])
    v = d[it & np.isfinite(d)]
    assert len(v) and v.min() >= G_n(rho, n) * (1 - 1e-6)


def test_adversarial_never_exceeds_random_expected_distance():
    """The adversarial (minimum) distance must be <= a random probe's."""
    rng = np.random.default_rng(5)
    rho, n = 1/3, 3
    pts = rng.uniform(0, 1, 400)
    for r in pts[:120]:
        if cell_of(float(r), rho, n)[0] != "leaf":
            continue
        adv = d_cross_to_other_leaf(float(r), rho, n)
        lo, hi, _ = leaf_bounds(float(r), rho, n)
        assert adv <= max(r - lo, hi - r) + G_n(rho, n) + 1e-9


def test_system_b_summary_has_zero_violations():
    s = json.loads(pathlib.Path(
        "results/v3_3_3/tables/systemB_adversarial_summary.json").read_text())
    assert s["adversarial"] is True and s["bound_violations"] == 0


# ------------------------------------------------------------------ System A
def test_budget_matching_enforced_and_failures_excluded():
    g = json.loads(pathlib.Path(
        "results/v3_3_3/tables/systemA_gate.json").read_text())
    assert g["budget_matched"] is True
    m = json.loads(pathlib.Path(
        "results/v3_3_3/tables/systemA_meta.json").read_text())
    for k, v in m["gains"].items():
        assert v["matched"] == (abs(v["rel"]) <= 0.03)
    assert "0.46" in g["excluded_for_budget"]


def test_qcap_and_identical_pairing():
    import pandas as pd
    df = pd.read_csv("results/v3_3_3/raw/systemA_qwen2.5-0.5b-instruct_n3.csv")
    P = json.loads(pathlib.Path("configs/v3_3_3/protocol.json").read_text())
    assert df.C_max.max() <= P["q_cap"] * 1.001
    # every rho saw exactly the same prompts / attacks / severities
    key = df.groupby("rho").apply(
        lambda g: tuple(sorted(set(zip(g.pid, g.attack, g.eps)))))
    assert len(set(key)) == 1


def test_endpoint_not_degenerate():
    import pandas as pd
    a = pd.read_csv("results/v3_3_3/tables/systemA_auc_per_prompt.csv")
    assert 0.0 < a.auc.mean() < 1.0 and a.auc.std() > 0


def test_no_dfinal_used_for_fitting():
    import pandas as pd
    v = pd.read_csv("results/v3_3_3/tables/leakage_audit.csv")
    assert int(v.n_violations.sum()) == 0


# --------------------------------------------------------------- freeze/verdict
def test_freeze_precedes_final_and_discloses_chronology():
    s = json.loads(pathlib.Path(
        "configs/v3_3_3/PRE_ANALYSIS_FREEZE.json").read_text())
    c = s["CHRONOLOGY_DISCLOSURE"]
    assert c["D_final_touched_before_this_seal"] is False
    assert c["v332_system_B_ran_before_its_own_seal"] is True
    assert len(c["exploratory_before_seal"]) >= 4
    assert s["POST_SEAL_AMENDMENT"]["decision_logic_changed"] is False


def test_splits_disjoint():
    S = json.loads(pathlib.Path("configs/v3_3_3/splits.json").read_text())
    assert not (set(S["D_beh"]) & set(S["D_final"]))
    assert S["n_final"] >= 40


def test_classifier_is_deterministic_and_covers_all_arms():
    from final_claim_check_v333 import v_math, v_behavior, v_system_b, v_system_a
    G = json.loads(pathlib.Path("results/v3_3_3/tables/gates.json").read_text())
    a1 = (v_math(G["math"]), v_behavior(G["behavior"]),
          v_system_b(G["system_b"]), v_system_a(G["system_a"]))
    a2 = (v_math(G["math"]), v_behavior(G["behavior"]),
          v_system_b(G["system_b"]), v_system_a(G["system_a"]))
    assert a1 == a2


def test_classifier_cannot_force_positive():
    from final_claim_check_v333 import v_system_a
    assert v_system_a({"budget_matched": False})[0] == "F_INCONCLUSIVE"
    assert v_system_a({"budget_matched": True, "endpoint_attainable": True,
                       "cantor_vs_alt_ci": [-0.05, 0.05], "sesoi": 0.02
                       })[0] == "F_INCONCLUSIVE"
    assert v_system_a({"budget_matched": True, "endpoint_attainable": True,
                       "cantor_vs_alt_ci": [0.01, 0.03], "sesoi": 0.02
                       })[0] == "A_CANTOR_GENERATION_POSITIVE"
