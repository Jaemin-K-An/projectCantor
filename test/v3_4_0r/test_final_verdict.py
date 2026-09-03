"""The verdict machine must be a pure function of the stored gates."""
import importlib
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v3_4_0r"))
from patch_claim_classifier import cantor_verdict, controller_verdict, overall_verdict  # noqa: E402


def test_budget_mismatch_dominates_every_other_gate():
    assert overall_verdict(certificate="CERT1_VALID", budget="BUD2_MISMATCH",
                           controller="CTRL1_CONTROLLER_EFFECTIVE",
                           cantor="CANTOR1_SPECIFIC_GAIN", utility="U1_PASS") \
        == "F_EQUAL_BUDGET_CONFIRMATION_BLOCKED"


def test_inert_controller_gives_verdict_e():
    assert overall_verdict(certificate="CERT1_VALID", budget="BUD1_MATCHED",
                           controller="CTRL2_CONTROLLER_PRACTICALLY_INERT",
                           cantor="CANTOR2_RHO_FAMILY_EQUIVALENT", utility="U1_PASS") \
        == "E_CONTROLLER_PRACTICALLY_INERT"


def test_effective_controller_without_cantor_specificity_is_verdict_d():
    assert overall_verdict(certificate="CERT1_VALID", budget="BUD1_MATCHED",
                           controller="CTRL1_CONTROLLER_EFFECTIVE",
                           cantor="CANTOR3_OTHER_RHO_BETTER", utility="U1_PASS") \
        == "D_SENSOR_ACTUATOR_CONTROLLER_EFFECTIVE_BUT_CANTOR_NOT_SPECIAL"


def test_cantor_gain_requires_everything():
    assert overall_verdict(certificate="CERT1_VALID", budget="BUD1_MATCHED",
                           controller="CTRL1_CONTROLLER_EFFECTIVE",
                           cantor="CANTOR1_SPECIFIC_GAIN", utility="U1_PASS") \
        == "B_CANTOR_STRUCTURAL_AND_BEHAVIORAL_ADVANTAGE"
    assert overall_verdict(certificate="CERT1_VALID", budget="BUD1_MATCHED",
                           controller="CTRL1_CONTROLLER_EFFECTIVE",
                           cantor="CANTOR1_SPECIFIC_GAIN", utility="U2_FAIL") \
        == "G_INCONCLUSIVE"


def test_certificate_failure_blocks_everything():
    assert overall_verdict(certificate="CERT2_IMPLEMENTATION_FAILURE", budget="BUD1_MATCHED",
                           controller="CTRL1_CONTROLLER_EFFECTIVE",
                           cantor="CANTOR1_SPECIFIC_GAIN", utility="U1_PASS") \
        == "G_INCONCLUSIVE"


def test_controller_verdict_needs_the_interval_inside_sesoi_for_inertness():
    assert controller_verdict(interval_lo=-0.02, interval_hi=0.01, efficacy_sesoi=0.03) \
        == "CTRL2_CONTROLLER_PRACTICALLY_INERT"
    assert controller_verdict(interval_lo=-0.09, interval_hi=0.09, efficacy_sesoi=0.03) \
        == "CTRL3_INCONCLUSIVE"


def test_recorded_verdict_is_internally_consistent():
    path = ROOT / "results/v3_4_0r/tables/final_verdict.json"
    if not path.exists():
        pytest.skip("final verdict not produced yet")
    v = json.loads(path.read_text())
    assert v["OVERALL"] == overall_verdict(certificate=v["CERTIFICATE"], budget=v["BUDGET"],
                                           controller=v["CONTROLLER_EFFECT"],
                                           cantor=v["CANTOR_BEHAVIOR"], utility=v["UTILITY"])
    if v["BUDGET"] == "BUD2_MISMATCH":
        assert v["CANTOR_BEHAVIOR"] == "CANTOR4_BLOCKED_BUDGET"
        assert v["GENERATION"] == "GEN6_EQUAL_BUDGET_COMPARISON_BLOCKED"
    if v["SEMANTIC"] != "SEM1_VALID":
        assert v["semantic_claim_allowed"] is False


def test_structural_claim_is_phrased_as_structural_only():
    path = ROOT / "results/v3_4_0r/tables/final_verdict.json"
    if not path.exists():
        pytest.skip("final verdict not produced yet")
    v = json.loads(path.read_text())
    assert "STRUCTURAL POLICY-SEPARATION OPTIMUM" in v["structural_claim"]
    assert "not an empirical LLM safety optimum" in v["structural_claim"]
    assert v["llm_is_not_claimed_to_be_fractal"] is True
