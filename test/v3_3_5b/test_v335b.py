"""V3.3.5b mandatory tests (harness sections 59-60)."""
import sys, json, pathlib
sys.path.insert(0, "llm/src"); sys.path.insert(0, "scripts/v3_3_5b")
import numpy as np, pandas as pd, pytest
from cantor_guard_v335b.temporal_budget import (SCHEDULES, schedule_weights,
                                                q_from_budget, b2, b1, active_states)
from cantor_guard_v335b.temporal_generation import TemporalTrace
from cantor_guard_v334.certified_geometry import M_n, rho_max
from cantor_guard_v335.certificate import eps_z_affine


# ------------------------------------------------------------- BUDGET MATH
@pytest.mark.parametrize("B", [0.005, 0.02, 0.1, 0.4, 0.8])
def test_b2_identical_across_schedules(B):
    vals = {round(b2(q_from_budget(s, B)), 12)
            for s in SCHEDULES if SCHEDULES[s]}
    assert vals == {round(B, 12)}


def test_single_state_q_equals_B2():
    B = 0.3
    for s in ("S1_P0_ONLY", "S2_G1_ONLY"):
        q = q_from_budget(s, B)
        assert len(q) == 1 and abs(list(q.values())[0] - B) < 1e-15


@pytest.mark.parametrize("s", ["S3_EARLY_2", "S4_EARLY_4", "S5_EARLY_8", "S6_ALL_K"])
def test_uniform_q_is_B2_over_sqrt_T(s):
    B, T = 0.4, len(active_states(s))
    q = q_from_budget(s, B)
    assert all(abs(x - B/np.sqrt(T)) < 1e-15 for x in q.values())
    assert abs(b2(q) - B) < 1e-15


def test_unit_l2_weights():
    for s in SCHEDULES:
        w = list(schedule_weights(s).values())
        if w:
            assert abs(float(np.linalg.norm(w)) - 1.0) < 1e-12


def test_b1_differs_and_is_secondary():
    B = 0.2
    vals = {round(b1(q_from_budget(s, B)), 9) for s in SCHEDULES if SCHEDULES[s]}
    assert len(vals) > 1          # L1 is NOT matched; that is why B2 is primary


def test_schedule_state_sets():
    assert active_states("S1_P0_ONLY") == [0]
    assert active_states("S2_G1_ONLY") == [1]
    assert active_states("S3_EARLY_2") == [0, 1]
    assert 0 not in active_states("S7_LATE_4")     # LATE excludes P0


# -------------------------------------------------------------- TRACE/HOOK
def test_trace_resets():
    t = TemporalTrace(); t.forward_index = 5; t.z[3] = np.zeros(2); t.reset()
    assert t.forward_index == 0 and not t.z and not t.q_realised


# ------------------------------------------------- realised budget in the data
@pytest.mark.parametrize("split", ["D_temporal_dev", "D_temporal_confirm"])
def test_realised_budget_within_tolerance(split):
    d = pd.read_csv(f"results/v3_3_5b/raw/temporal_{split}.csv")
    a = d[d.B2_target > 0]
    assert (abs(a.B2_realised/a.B2_target - 1) <= 0.03).all()


@pytest.mark.parametrize("split", ["D_temporal_dev", "D_temporal_confirm"])
def test_same_prompts_every_schedule(split):
    d = pd.read_csv(f"results/v3_3_5b/raw/temporal_{split}.csv")
    a = d[d.B2_target > 0]
    sets = a.groupby("schedule").pid.apply(lambda s: tuple(sorted(set(s))))
    assert len(set(sets)) == 1


def test_no_degeneration_in_analysis():
    d = pd.read_csv("results/v3_3_5b/raw/temporal_D_temporal_confirm.csv")
    assert (d.coherence >= 0.8).all()


def test_pre_declared_grid_produced_no_variation():
    """The documented reason the cap had to be revisited."""
    r = json.loads(pathlib.Path("results/v3_3_5b/tables/regime_gap.json").read_text())
    assert r["cap"] == 0.05
    assert r["first_effective_q_nondegenerate"] > r["cap"]
    assert min(r["historical_global_q_estimate"].values()) > r["cap"]


def test_effects_are_not_degenerate():
    r = json.loads(pathlib.Path("results/v3_3_5b/tables/regime_gap.json").read_text())
    assert all(x["coherence"] >= 0.99 for x in r["escalation"])


# ------------------------------------------------------------------ VERDICT
def test_dev_and_confirm_agree_on_ordering():
    out = {}
    for s in ("D_temporal_dev", "D_temporal_confirm"):
        d = pd.read_csv(f"results/v3_3_5b/raw/temporal_{s}.csv")
        base = d[d.B2_target == 0].refusal.mean()
        t = d[d.B2_target == 0.8].groupby("schedule").refusal.mean()
        out[s] = (base - t).idxmax()
    assert out["D_temporal_dev"] in ("S1_P0_ONLY", "S3_EARLY_2")
    assert out["D_temporal_confirm"] in ("S1_P0_ONLY", "S3_EARLY_2")


def test_distributed_does_not_beat_single_state():
    c = pd.read_csv("results/v3_3_5b/tables/temporal_contrasts.csv")
    assert not c.distributed_wins.any()
    neg = c[(c.contrast.str.contains("S1_P0_ONLY")) & (c.simult_hi < 0)]
    assert len(neg) > 0          # distribution is significantly WORSE


def test_verdict_is_td3_and_stage_b_not_opened():
    v = json.loads(pathlib.Path("results/v3_3_5b/tables/verdict_v335b.json").read_text())
    assert v["TEMPORAL"] == "TD3_SINGLE_STATE_BETTER"
    assert v["TRAJECTORY"] == "TR3_NOT_RUN_TEMPORAL_GATE_FAILED"
    assert v["OVERALL"] == "D_GLOBAL_ADVANTAGE_WAS_ACCUMULATION"


def test_classifier_would_proceed_only_on_TD1():
    from final_claim_check_v335b import overall
    ok = overall(("M1_CANTOR_AFFINE_MAXIMIN_VALID", ""),
                 ("TD1_DISTRIBUTED_SUPPORTED", ""), ("", ""), ("", ""))
    assert ok[0] == "PROCEED_TO_STAGE_B"
    for td in ("TD2_ACCUMULATION_EXPLAINS_GLOBAL", "TD3_SINGLE_STATE_BETTER"):
        assert overall(("M1_CANTOR_AFFINE_MAXIMIN_VALID", ""), (td, ""),
                       ("", ""), ("", ""))[0] == "D_GLOBAL_ADVANTAGE_WAS_ACCUMULATION"


def test_final_traj_untouched():
    f = json.loads(pathlib.Path("configs/v3_3_5b/FREEZE_STAGE_A.json").read_text())
    assert f["CHRONOLOGY"]["D_final_traj_touched"] is False
    s = json.loads(pathlib.Path("configs/v3_3_5b/splits.json").read_text())
    assert len(s["blocks"]["D_final_traj"]) >= 80 and s["disjoint_from_all_prior"]


# ---------------------------------------------------------- MATH UNCHANGED
@pytest.mark.parametrize("n,want", [(2, 1/4), (3, 1/3), (5, 2/5)])
def test_cantor_math_frozen(n, want):
    g = np.linspace(0.05, 0.49, 8801)
    e = np.array([eps_z_affine(x, n, 2.0) for x in g])
    assert abs(g[e.argmax()] - want) < 1e-3 and abs(rho_max(n) - want) < 1e-15
    assert abs(M_n(1/3, 3) - 1/27) < 1e-15
