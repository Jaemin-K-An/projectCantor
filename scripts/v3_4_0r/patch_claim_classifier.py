"""Pure V3.4.0R verdict functions with hard gate precedence."""
from __future__ import annotations


def generation_verdict(*, budget_all_matched: bool, comparison_blocked: bool,
                       all_within_sesoi: bool, all_favour_cantor: bool,
                       any_favours_other: bool, have_contrasts: bool) -> str:
    if comparison_blocked or not budget_all_matched:
        return "CANTOR4_BLOCKED_BUDGET"
    if not have_contrasts:
        return "RHO4_INCONCLUSIVE"
    if all_favour_cantor:
        return "RHO1_CANTOR_SPECIFIC_GAIN"
    if any_favours_other:
        return "RHO3_OTHER_RHO_BETTER"
    if all_within_sesoi:
        return "RHO2_PRACTICALLY_EQUIVALENT"
    return "RHO4_INCONCLUSIVE"


def rho_verdict(**kwargs) -> str:
    return generation_verdict(**kwargs)


def cantor_verdict(**kwargs) -> str:
    """Compatibility alias retained for older local tests/scripts."""
    return generation_verdict(**kwargs)


def controller_verdict(*, interval_lo: float | None, interval_hi: float | None,
                       efficacy_sesoi: float) -> str:
    if interval_lo is None or interval_hi is None:
        return "CTRL4_INCONCLUSIVE"
    if interval_lo > efficacy_sesoi:
        return "CTRL1_EFFECTIVE"
    if interval_hi < -efficacy_sesoi:
        return "CTRL3_HARMFUL"
    if interval_lo >= -efficacy_sesoi and interval_hi <= efficacy_sesoi:
        return "CTRL2_PRACTICALLY_INERT"
    return "CTRL4_INCONCLUSIVE"


def baseline_verdict(*, interval_lo: float | None, interval_hi: float | None,
                     sesoi: float) -> str:
    if interval_lo is None or interval_hi is None:
        return "BASE4_INCONCLUSIVE"
    if interval_lo > sesoi:
        return "BASE1_CANTOR_BEATS_LINEAR"
    if interval_hi < -sesoi:
        return "BASE3_LINEAR_BEATS_CANTOR"
    if interval_lo >= -sesoi and interval_hi <= sesoi:
        return "BASE2_CANTOR_LINEAR_EQUIVALENT"
    return "BASE4_INCONCLUSIVE"


def overall_verdict(*, sensor_transport: str, certificate: str, budget: str,
                    controller: str, baseline: str, rho: str, utility: str) -> str:
    """Apply the preregistered decision tree; upstream gates dominate."""
    if sensor_transport in {"ST2_FAIL", "ST3_WINDOW_SHIFT"}:
        return "E_EXTERNAL_SENSOR_TRANSPORT_FAILURE"
    if budget == "BUD2_MISMATCH":
        return "F_BUDGET_CONFIRMATION_BLOCKED"
    supported = (
        sensor_transport == "ST1_PASS"
        and certificate == "CERT1_VALID"
        and budget == "BUD1_MATCHED"
        and controller == "CTRL1_EFFECTIVE"
        and utility == "U1_PASS"
    )
    if supported and baseline == "BASE1_CANTOR_BEATS_LINEAR" and rho == "RHO1_CANTOR_SPECIFIC_GAIN":
        return "B_CANTOR_ADDS_BEHAVIORAL_VALUE"
    if supported and (baseline == "BASE2_CANTOR_LINEAR_EQUIVALENT"
                      or rho == "RHO2_PRACTICALLY_EQUIVALENT"):
        return "C_CONTROLLER_WORKS_BUT_CANTOR_NOT_SPECIAL"
    if supported:
        return "A_EXTERNAL_SENSOR_ACTUATOR_CONTROLLER_SUPPORTED"
    if certificate == "CERT1_VALID" and controller == "CTRL2_PRACTICALLY_INERT":
        return "D_CANTOR_STRUCTURAL_ONLY"
    return "G_INCONCLUSIVE"


def legacy_v340_generation_verdict(*, all_within_sesoi: bool, all_favour_cantor: bool,
                                   any_favours_other: bool, have_contrasts: bool,
                                   **_ignored) -> str:
    """Historical defective logic, retained only for its regression test."""
    if not have_contrasts:
        return "GEN5_INCONCLUSIVE"
    if all_favour_cantor:
        return "GEN4_REFUSAL_ONLY_RESULT"
    if any_favours_other:
        return "GEN3_OTHER_RHO_BETTER"
    if all_within_sesoi:
        return "GEN2_RHO_FAMILY_PRACTICALLY_EQUIVALENT"
    return "GEN5_INCONCLUSIVE"
