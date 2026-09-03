"""The verdict machine is a pure function of the stored preregistered gates."""
import json
import pathlib

from patch_claim_classifier import baseline_verdict, controller_verdict, overall_verdict

ROOT = pathlib.Path(__file__).resolve().parents[2]


def common(**overrides):
    values = dict(
        sensor_transport="ST1_PASS", certificate="CERT1_VALID", budget="BUD1_MATCHED",
        controller="CTRL1_EFFECTIVE", baseline="BASE1_CANTOR_BEATS_LINEAR",
        rho="RHO1_CANTOR_SPECIFIC_GAIN", utility="U1_PASS",
    )
    values.update(overrides)
    return values


def test_window_shift_has_first_precedence():
    assert overall_verdict(**common(sensor_transport="ST3_WINDOW_SHIFT")) \
        == "E_EXTERNAL_SENSOR_TRANSPORT_FAILURE"


def test_budget_mismatch_blocks_equal_budget_confirmation():
    assert overall_verdict(**common(budget="BUD2_MISMATCH")) \
        == "F_BUDGET_CONFIRMATION_BLOCKED"


def test_supported_and_cantor_specific_paths():
    assert overall_verdict(**common()) == "B_CANTOR_ADDS_BEHAVIORAL_VALUE"
    assert overall_verdict(**common(baseline="BASE4_INCONCLUSIVE", rho="RHO4_INCONCLUSIVE")) \
        == "A_EXTERNAL_SENSOR_ACTUATOR_CONTROLLER_SUPPORTED"
    assert overall_verdict(**common(baseline="BASE2_CANTOR_LINEAR_EQUIVALENT")) \
        == "C_CONTROLLER_WORKS_BUT_CANTOR_NOT_SPECIAL"


def test_structural_only_requires_direct_inertness_interval():
    assert overall_verdict(**common(controller="CTRL2_PRACTICALLY_INERT")) \
        == "D_CANTOR_STRUCTURAL_ONLY"
    assert controller_verdict(interval_lo=-0.02, interval_hi=0.01, efficacy_sesoi=0.03) \
        == "CTRL2_PRACTICALLY_INERT"
    assert controller_verdict(interval_lo=-0.09, interval_hi=0.09, efficacy_sesoi=0.03) \
        == "CTRL4_INCONCLUSIVE"


def test_linear_baseline_labels():
    assert baseline_verdict(interval_lo=0.04, interval_hi=0.08, sesoi=0.03) \
        == "BASE1_CANTOR_BEATS_LINEAR"
    assert baseline_verdict(interval_lo=-0.02, interval_hi=0.02, sesoi=0.03) \
        == "BASE2_CANTOR_LINEAR_EQUIVALENT"


def test_recorded_stop_verdict_is_consistent_and_narrow():
    v = json.loads((ROOT / "results/v3_4_0r/tables/final_verdict.json").read_text())
    assert v["SENSOR_TRANSPORT"] == "ST3_WINDOW_SHIFT"
    assert v["OVERALL"] == "E_EXTERNAL_SENSOR_TRANSPORT_FAILURE"
    assert v["invalid_q025_final_used"] is False
    assert v["semantic_safety_guarantee_claimed"] is False
