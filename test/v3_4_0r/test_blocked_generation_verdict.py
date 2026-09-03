"""The regression that reproduces the V3.4.0 defect and proves it is fixed."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v3_4_0r"))

from patch_claim_classifier import (  # noqa: E402
    cantor_verdict,
    controller_verdict,
    generation_verdict,
    legacy_v340_generation_verdict,
    overall_verdict,
)

# The exact V3.4.0 situation: every SESOI interval was tight, and no arm was
# budget-valid.
V340_CASE = dict(budget_all_matched=False, comparison_blocked=True,
                 all_within_sesoi=True, all_favour_cantor=False,
                 any_favours_other=False, have_contrasts=True)


def test_old_classifier_reproduces_the_defect():
    assert legacy_v340_generation_verdict(**V340_CASE) == "GEN2_RHO_FAMILY_PRACTICALLY_EQUIVALENT"


def test_new_classifier_blocks_it():
    assert generation_verdict(**V340_CASE) == "CANTOR4_BLOCKED_BUDGET"
    assert cantor_verdict(**V340_CASE) == "CANTOR4_BLOCKED_BUDGET"


def test_the_stored_v340_artifacts_really_were_in_that_state():
    gen = json.loads((ROOT / "results/v3_4_0/tables/generation_analysis.json").read_text())
    budget = json.loads((ROOT / "results/v3_4_0/tables/final_budget_audit.json").read_text())
    verdict = json.loads((ROOT / "results/v3_4_0/tables/final_verdict.json").read_text())
    assert gen["confirmatory_comparison_blocked"] is True
    assert budget["all_matched"] is False
    assert verdict["GENERATION"] == "GEN2_RHO_FAMILY_PRACTICALLY_EQUIVALENT"


def test_budget_block_overrides_every_sesoi_outcome():
    for extra in (dict(all_within_sesoi=True, all_favour_cantor=False, any_favours_other=False),
                  dict(all_within_sesoi=False, all_favour_cantor=True, any_favours_other=False),
                  dict(all_within_sesoi=False, all_favour_cantor=False, any_favours_other=True)):
        case = dict(budget_all_matched=False, comparison_blocked=False,
                    have_contrasts=True, **extra)
        assert generation_verdict(**case) == "CANTOR4_BLOCKED_BUDGET"
        assert cantor_verdict(**case) == "CANTOR4_BLOCKED_BUDGET"


def test_valid_budget_still_lets_real_verdicts_through():
    ok = dict(budget_all_matched=True, comparison_blocked=False, have_contrasts=True)
    assert cantor_verdict(**ok, all_within_sesoi=True, all_favour_cantor=False,
                          any_favours_other=False) == "RHO2_PRACTICALLY_EQUIVALENT"
    assert cantor_verdict(**ok, all_within_sesoi=False, all_favour_cantor=True,
                          any_favours_other=False) == "RHO1_CANTOR_SPECIFIC_GAIN"
    assert cantor_verdict(**ok, all_within_sesoi=False, all_favour_cantor=False,
                          any_favours_other=True) == "RHO3_OTHER_RHO_BETTER"


def test_inertness_needs_the_baseline_not_rho_similarity():
    # interval straddling zero but wider than SESOI is inconclusive, not inert
    assert controller_verdict(interval_lo=-0.20, interval_hi=0.18, efficacy_sesoi=0.03) \
        == "CTRL4_INCONCLUSIVE"
    assert controller_verdict(interval_lo=-0.01, interval_hi=0.02, efficacy_sesoi=0.03) \
        == "CTRL2_PRACTICALLY_INERT"
    assert controller_verdict(interval_lo=0.05, interval_hi=0.19, efficacy_sesoi=0.03) \
        == "CTRL1_EFFECTIVE"
    assert controller_verdict(interval_lo=None, interval_hi=None, efficacy_sesoi=0.03) \
        == "CTRL4_INCONCLUSIVE"


def test_overall_prefers_budget_block_over_everything():
    common = dict(sensor_transport="ST1_PASS", certificate="CERT1_VALID",
                  baseline="BASE1_CANTOR_BEATS_LINEAR", rho="RHO1_CANTOR_SPECIFIC_GAIN")
    assert overall_verdict(**common, budget="BUD2_MISMATCH",
                           controller="CTRL1_EFFECTIVE", utility="U1_PASS") \
        == "F_BUDGET_CONFIRMATION_BLOCKED"
    assert overall_verdict(**common, budget="BUD1_MATCHED",
                           controller="CTRL2_PRACTICALLY_INERT", utility="U1_PASS") \
        == "D_CANTOR_STRUCTURAL_ONLY"


def test_window_shift_dominates_downstream_results():
    assert overall_verdict(
        sensor_transport="ST3_WINDOW_SHIFT", certificate="CERT1_VALID",
        budget="BUD1_MATCHED", controller="CTRL1_EFFECTIVE",
        baseline="BASE1_CANTOR_BEATS_LINEAR", rho="RHO1_CANTOR_SPECIFIC_GAIN",
        utility="U1_PASS",
    ) == "E_EXTERNAL_SENSOR_TRANSPORT_FAILURE"
