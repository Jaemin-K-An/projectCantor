"""V3.3.5b PHASE 1/2 -- matched trajectory intervention budget.

THE CONFOUND THIS REMOVES. V3.3.3's "global" dose added lambda*v on EVERY
forward and broadcast it over EVERY sequence position -- 48 forwards for a
48-token generation -- while P0-only and G1-only touched exactly one state.
The ~20x leverage ratio I reported in V3.3.5a sits between sqrt(48) = 6.9 and
48, so repeated intervention energy alone can plausibly account for it. Until
the budget is matched, "temporally distributed control" is not established.

NORMALIZED PER-STEP INTERVENTION. At state t apply

    Delta h_t = s * q_t * ||h_t||_2 * v_ref,      ||v_ref||_2 = 1

so ||Delta h_t|| / ||h_t|| = q_t exactly, independent of how the residual norm
drifts across the trajectory.

TRAJECTORY BUDGET (primary):   B2 = sqrt( sum_t q_t^2 )
                  (secondary): B1 = sum_t |q_t|

MATCHING. A single-state schedule uses q_1 = B2, giving sqrt(q_1^2) = B2. A
uniform T-state schedule uses q_t = B2/sqrt(T), giving
sqrt(T * B2^2 / T) = B2. Every schedule therefore spends the SAME trajectory
L2 energy, and only its temporal placement differs.
"""
from __future__ import annotations
import numpy as np

__all__ = ["schedule_weights", "q_from_budget", "b2", "b1", "SCHEDULES",
           "active_states", "K_HORIZON"]

K_HORIZON = 8          # decode states G1..G8, plus P0 -> 9 controllable states

SCHEDULES = {
    "S0_NONE":     [],
    "S1_P0_ONLY":  [0],
    "S2_G1_ONLY":  [1],
    "S3_EARLY_2":  [0, 1],
    "S4_EARLY_4":  [0, 1, 2, 3],
    "S5_EARLY_8":  [0, 1, 2, 3, 4, 5, 6, 7],
    "S6_ALL_K":    list(range(K_HORIZON + 1)),
    "S7_LATE_4":   [1, 2, 3, 4],            # secondary
}


def active_states(name: str) -> list[int]:
    return list(SCHEDULES[name])


def schedule_weights(name: str) -> dict[int, float]:
    """Uniform unit-L2 weights over the active set: w_t = 1/sqrt(T).

    Uniform is deliberate. Sensitivity-derived weights would let the temporal
    schedule be fitted to the very refusal effect being measured.
    """
    act = active_states(name)
    if not act:
        return {}
    w = 1.0 / np.sqrt(len(act))
    return {t: float(w) for t in act}


def q_from_budget(name: str, B2: float) -> dict[int, float]:
    """Per-state relative magnitudes realising exactly this B2."""
    return {t: float(B2 * w) for t, w in schedule_weights(name).items()}


def b2(q: dict[int, float] | list) -> float:
    v = np.asarray(list(q.values()) if isinstance(q, dict) else q, float)
    return float(np.sqrt((v ** 2).sum())) if v.size else 0.0


def b1(q: dict[int, float] | list) -> float:
    v = np.asarray(list(q.values()) if isinstance(q, dict) else q, float)
    return float(np.abs(v).sum()) if v.size else 0.0
