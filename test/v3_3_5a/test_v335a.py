"""V3.3.5a mandatory tests (harness sections 44-45)."""
import sys, json, pathlib
sys.path.insert(0, "llm/src"); sys.path.insert(0, "scripts/v3_3_5a")
import numpy as np, torch, pytest
from cantor_guard_v335a.p0_residual import (last_valid_index, P0Trace,
                                            PREFILL, DECODE)
from cantor_guard_v335.certificate import eps_z_affine
from cantor_guard_v334.certified_geometry import M_n, rho_max


# ------------------------------------------------------------- PADDING SAFETY
def test_last_valid_index_left_padding():
    m = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    assert last_valid_index(m).tolist() == [4, 4, 4]


def test_last_valid_index_right_padding():
    m = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 0, 0, 0, 0]])
    assert last_valid_index(m).tolist() == [2, 3, 0]


def test_naive_minus_one_wrong_under_right_padding():
    """h[:, -1, :] is only safe because the tokenizer pads LEFT; the mask-based
    index does not depend on that coupling."""
    m = torch.tensor([[1, 1, 1, 0, 0]])
    assert last_valid_index(m).item() == 2 != m.shape[1] - 1


def test_historical_extraction_is_left_padded():
    import inspect
    from cantor_guard import models
    assert 'padding_side = "left"' in inspect.getsource(models.load_model)


# ------------------------------------------------------------------ P0 PHASE
def test_p0_trace_phases():
    t = P0Trace(); t.reset()
    assert t.phase() == PREFILL
    t.forward_index += 1; assert t.phase() == DECODE
    t.forward_index += 1; assert t.phase() == DECODE


def test_p0_trace_resets():
    t = P0Trace(); t.forward_index = 5; t.z_clean = np.zeros(2); t.reset()
    assert t.forward_index == 0 and t.z_clean is None


def test_p0_intervention_confined_to_prefill():
    d = json.loads(pathlib.Path(
        "results/v3_3_5a/tables/p0_direction.json").read_text())
    assert d["first_token"]["prefill_only"] is True


def test_p0_is_upstream_of_first_token():
    """The defining property of P0: it can change token 1. G1 cannot."""
    d = json.loads(pathlib.Path(
        "results/v3_3_5a/tables/p0_direction.json").read_text())
    ft = d["first_token"]
    assert ft["max_abs_dlogit"] > 1.0 and ft["kl"] > 0 and ft["top1_flip"] > 0


# ----------------------------------------------------- MATH UNCHANGED
@pytest.mark.parametrize("n,want", [(2, 1/4), (3, 1/3), (5, 2/5)])
def test_certificate_optimum_unchanged(n, want):
    g = np.linspace(0.05, 0.49, 8801)
    e = np.array([eps_z_affine(x, n, 2.0) for x in g])
    assert abs(g[e.argmax()] - want) < 1e-3 and abs(rho_max(n) - want) < 1e-15


def test_cantor_value_unchanged():
    W = 2.0
    assert abs(eps_z_affine(1/3, 3, W) - 2*W/27) < 1e-15
    assert abs(M_n(1/3, 3) - 1/27) < 1e-15


def test_no_new_coordinate_transform_introduced():
    src = pathlib.Path("llm/src/cantor_guard_v335a/p0_residual.py").read_text()
    # "logits" legitimately appears (first-token logits); look for TRANSFORM use
    for bad in ("sigmoid(", "np.log(", "torch.log(", "tanh(", "expit"):
        assert bad not in src


# --------------------------------------------------- STANDARDIZED GATE
def test_standardized_gate_is_scale_invariant():
    """beta_std = b*sigma must not change if the projection is rescaled."""
    from cantor_guard_v333.behavioral_boundary import fit_logistic
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2.0, 4000)
    y = (rng.random(4000) < 1/(1+np.exp(-(0.8*z)))).astype(float)
    a1, b1 = fit_logistic(z, y); s1 = z.std(ddof=1)
    a2, b2 = fit_logistic(z*10, y); s2 = (z*10).std(ddof=1)
    assert abs(b1*s1 - b2*s2) < 0.05 * abs(b1*s1)
    assert abs(b1 - b2) > 0.5 * abs(b1)      # the RAW slope is not invariant


def test_g1_reworded_not_zero_effect():
    """V3.3.5's G1 slope CI excluded zero: effect existed, leverage was tiny."""
    c = json.loads(pathlib.Path(
        "results/v3_3_5/tables/g1_boundary_D_beh_g1_confirm.json").read_text())
    assert c["bootstrap"]["slope_ci95"][0] > 0


# ------------------------------------------------------ empirical artefacts
def test_p0_boundary_status_reproduced():
    dev = json.loads(pathlib.Path(
        "results/v3_3_5a/tables/p0_boundary_D_beh_P0_dev.json").read_text())
    con = json.loads(pathlib.Path(
        "results/v3_3_5a/tables/p0_boundary_D_beh_P0_confirm.json").read_text())
    assert dev["status"] == con["status"] == "B2_P0_CAUSAL_BUT_BOUNDARY_IMPRECISE"
    assert not dev["gate"]["all_pass"] and not con["gate"]["all_pass"]


def test_b2_and_b3_not_collapsed():
    from final_claim_check_v335a import v_boundary
    assert v_boundary({"status": "B2_P0_CAUSAL_BUT_BOUNDARY_IMPRECISE",
                       "beta_std": .02, "dP_2sigma": .01,
                       "ci_width_sigma": 40})[0] != v_boundary(
        {"status": "B3_P0_NOT_CAUSAL"})[0]


def test_phase_comparison_uses_standardized_slopes():
    p = json.loads(pathlib.Path(
        "results/v3_3_5a/tables/phase_causality.json").read_text())
    assert len(p["rows"]) == 3
    for r in p["rows"]:
        assert abs(r["beta_std"] - r["b_raw"] * r["sigma_phase"]) < 1e-9


def test_global_dominates_single_states():
    p = json.loads(pathlib.Path(
        "results/v3_3_5a/tables/phase_causality.json").read_text())
    d = {r["phase"]: r["beta_std"] for r in p["rows"]}
    assert d["GLOBAL (all forwards)"] > 5 * max(d["P0 (pre-token-1)"],
                                                d["G1 (first decode)"])


def test_final_set_not_spent():
    f = json.loads(pathlib.Path(
        "configs/v3_3_5a/PRE_ANALYSIS_FREEZE.json").read_text())
    assert f["CHRONOLOGY"]["D_final_P0_touched"] is False
    s = json.loads(pathlib.Path("configs/v3_3_5a/splits.json").read_text())
    assert len(s["blocks"]["D_final_P0"]) >= 80
    assert s["disjoint_from_all_prior"]


def test_splits_disjoint():
    s = json.loads(pathlib.Path("configs/v3_3_5a/splits.json").read_text())
    allp = [x for v in s["blocks"].values() for x in v]
    assert len(set(allp)) == len(allp)


def test_classifier_cannot_reach_A_without_B1():
    from final_claim_check_v335a import overall
    ov = overall(("M1_CANTOR_AFFINE_MAXIMIN_VALID", ""),
                 ("D2_P0_DIRECTION_NOT_CAUSAL", ""),
                 ("B2_P0_CAUSAL_BUT_BOUNDARY_IMPRECISE", ""),
                 ("PHASE_LEVERAGE_IS_DISTRIBUTED", ""),
                 ("G5_NOT_RUN_GATE_FAILURE", ""))
    assert ov[0] == "E_SINGLE_STATE_CAUSAL_CONTROLLER_NOT_SUPPORTED"
