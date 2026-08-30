"""V3.3 PHASE 17 -- Pareto rule. Safety equivalence gates everything.

The rule is deliberately asymmetric: no amount of structural cheapness can buy
a safety deficit larger than the SESOI. That is what keeps a "structure-positive"
result from quietly becoming a weaker safety claim (harness sections 25, 26).
"""
from __future__ import annotations

SESOI = 0.03


def dominates(*, r_a: float, r_b: float, c_a: float, c_b: float,
              sesoi: float = SESOI) -> bool:
    """A dominates B iff A is safety-equivalent (within `sesoi`) AND strictly
    cheaper structurally."""
    if r_a < r_b - sesoi:
        return False                 # safety gate: fails outright
    return c_a < c_b


def pareto_front(points: dict[str, tuple[float, float]],
                 sesoi: float = SESOI) -> list[str]:
    """`points`: name -> (robustness, structural_cost). Returns non-dominated."""
    out = []
    for a, (ra, ca) in points.items():
        if not any(dominates(r_a=rb, r_b=ra, c_a=cb, c_b=ca, sesoi=sesoi)
                   for b, (rb, cb) in points.items() if b != a):
            out.append(a)
    return sorted(out)
