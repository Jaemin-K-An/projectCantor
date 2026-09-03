"""Phase 1 -- the corrected verdict machine.

The V3.4.0 classifier could emit a practical-equivalence label while the budget
audit said no arm was valid. Here budget validity is a HARD PRECONDITION that
overrides every SESOI result, and controller efficacy is evaluated against an
attacked no-controller baseline rather than inferred from rho similarity.
"""
from __future__ import annotations


def generation_verdict(*, budget_all_matched: bool, comparison_blocked: bool,
                       all_within_sesoi: bool, all_favour_cantor: bool,
                       any_favours_other: bool, have_contrasts: bool) -> str:
    """Budget validity dominates. No override exists."""
    if comparison_blocked or not budget_all_matched:
        return "GEN6_EQUAL_BUDGET_COMPARISON_BLOCKED"
    if not have_contrasts:
        return "GEN5_INCONCLUSIVE"
    if all_favour_cantor:
        return "GEN1_CANTOR_GAIN"
    if any_favours_other:
        return "GEN3_OTHER_RHO_BETTER"
    if all_within_sesoi:
        return "GEN2_RHO_FAMILY_PRACTICALLY_EQUIVALENT"
    return "GEN5_INCONCLUSIVE"


def cantor_verdict(*, budget_all_matched: bool, comparison_blocked: bool,
                   all_within_sesoi: bool, all_favour_cantor: bool,
                   any_favours_other: bool, have_contrasts: bool) -> str:
    if comparison_blocked or not budget_all_matched:
        return "CANTOR4_BLOCKED_BUDGET"
    if not have_contrasts:
        return "CANTOR5_INCONCLUSIVE"
    if all_favour_cantor:
        return "CANTOR1_SPECIFIC_GAIN"
    if any_favours_other:
        return "CANTOR3_OTHER_RHO_BETTER"
    if all_within_sesoi:
        return "CANTOR2_RHO_FAMILY_EQUIVALENT"
    return "CANTOR5_INCONCLUSIVE"


def controller_verdict(*, interval_lo: float | None, interval_hi: float | None,
                       efficacy_sesoi: float) -> str:
    """Efficacy is decided against attacked NO-CONTROLLER, never by rho similarity.

    CTRL2 (inert) requires the interval to lie wholly inside the SESOI band --
    an interval that merely contains zero is inconclusive, not inert.
    """
    if interval_lo is None or interval_hi is None:
        return "CTRL3_INCONCLUSIVE"
    if interval_lo > efficacy_sesoi:
        return "CTRL1_CONTROLLER_EFFECTIVE"
    if interval_lo >= -efficacy_sesoi and interval_hi <= efficacy_sesoi:
        return "CTRL2_CONTROLLER_PRACTICALLY_INERT"
    return "CTRL3_INCONCLUSIVE"


def legacy_v340_generation_verdict(*, all_within_sesoi: bool, all_favour_cantor: bool,
                                   any_favours_other: bool, have_contrasts: bool,
                                   **_ignored) -> str:
    """The V3.4.0 logic, kept ONLY so the regression test can prove it was wrong.

    Note it never reads the budget at all -- that is the defect.
    """
    if not have_contrasts:
        return "GEN5_INCONCLUSIVE"
    if all_favour_cantor:
        return "GEN4_REFUSAL_ONLY_RESULT"
    if any_favours_other:
        return "GEN3_OTHER_RHO_BETTER"
    if all_within_sesoi:
        return "GEN2_RHO_FAMILY_PRACTICALLY_EQUIVALENT"
    return "GEN5_INCONCLUSIVE"


def overall_verdict(*, certificate: str, budget: str, controller: str,
                    cantor: str, utility: str) -> str:
    if budget == "BUD2_MISMATCH":
        return "F_EQUAL_BUDGET_CONFIRMATION_BLOCKED"
    if certificate != "CERT1_VALID":
        return "G_INCONCLUSIVE"
    if controller == "CTRL2_CONTROLLER_PRACTICALLY_INERT":
        return "E_CONTROLLER_PRACTICALLY_INERT"
    if controller != "CTRL1_CONTROLLER_EFFECTIVE" or utility != "U1_PASS":
        return "G_INCONCLUSIVE"
    if cantor == "CANTOR1_SPECIFIC_GAIN":
        return "B_CANTOR_STRUCTURAL_AND_BEHAVIORAL_ADVANTAGE"
    if cantor == "CANTOR2_RHO_FAMILY_EQUIVALENT":
        return "C_CANTOR_STRUCTURAL_ONLY_RHO_EQUIVALENT"
    if cantor in ("CANTOR3_OTHER_RHO_BETTER", "CANTOR5_INCONCLUSIVE"):
        return "D_SENSOR_ACTUATOR_CONTROLLER_EFFECTIVE_BUT_CANTOR_NOT_SPECIAL"
    return "G_INCONCLUSIVE"
