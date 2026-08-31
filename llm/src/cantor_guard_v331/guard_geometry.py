"""V3.3.1 -- binary symmetric recursive guard geometry.

A parent cell is split into three parts,

    [ child (rho) | guard (g) | child (rho) ]     2*rho + g = 1

so g(rho) = 1 - 2*rho on the domain 0 < rho < 1/2. The middle-third Cantor set
is rho = g = 1/3, but NOTHING here assumes that -- the whole point of V3.3.1 is
to find which criterion singles it out, and to be explicit that the criterion is
an assumption rather than a law.

  rho  REFINEMENT CAPACITY. State-space retained for deeper recursive control.
  g    GUARD CAPACITY. Region that absorbs boundary ambiguity: a coordinate
       landing here is treated conservatively instead of being handed to the
       opposite child's policy.

They compete directly (dg/drho = -2), and the guard requirement is left FREE as

    g >= kappa * rho,        kappa > 0

because fixing kappa = 1 is already assuming the Cantor answer.
"""
from __future__ import annotations
import math

__all__ = ["guard_width", "retention", "hausdorff_dim", "alpha_field",
           "alpha_sensitivity", "rho_star", "kappa_of_rho", "bottleneck",
           "feasible", "retained_measure", "new_guard_measure",
           "cumulative_coverage", "new_coverage_argmax", "RHO_CANTOR"]

RHO_CANTOR = 1.0 / 3.0
DOMAIN = (0.0, 0.5)


def _check(rho: float) -> None:
    if not (DOMAIN[0] < rho < DOMAIN[1]):
        raise ValueError(f"rho must lie in (0, 1/2); got {rho}")


def guard_width(rho: float) -> float:
    """g(rho) = 1 - 2*rho.  g' = -2 < 0."""
    _check(rho)
    return 1.0 - 2.0 * rho


def retention(rho: float) -> float:
    """Measure retained per level: 2*rho.  Strictly increasing."""
    _check(rho)
    return 2.0 * rho


def hausdorff_dim(rho: float) -> float:
    """d_H = log 2 / log(1/rho).  Strictly increasing on (0, 1/2)."""
    _check(rho)
    return math.log(2.0) / math.log(1.0 / rho)


def alpha_field(rho: float) -> float:
    """Cross-scale field amplification 1/(b*rho) with b = 2. Strictly decreasing."""
    _check(rho)
    return 1.0 / (2.0 * rho)


def alpha_sensitivity(rho: float) -> float:
    """Sensitivity amplification 1/(b*rho^2) with b = 2. Strictly decreasing."""
    _check(rho)
    return 1.0 / (2.0 * rho * rho)


def feasible(rho: float, kappa: float) -> bool:
    """Guard requirement g >= kappa*rho."""
    return guard_width(rho) >= kappa * rho - 1e-15


def rho_star(kappa: float) -> float:
    """THEOREM G. Largest feasible ratio: rho*(kappa) = 1/(2+kappa).

    1 - 2rho >= kappa*rho  <=>  1 >= (2+kappa)rho  <=>  rho <= 1/(2+kappa).
    kappa = 1 gives exactly 1/3, the middle-third Cantor ratio.
    """
    if kappa <= 0:
        raise ValueError("kappa > 0")
    return 1.0 / (2.0 + kappa)


def kappa_of_rho(rho: float) -> float:
    """The kappa for which this rho is the constrained optimum: g/rho."""
    _check(rho)
    return guard_width(rho) / rho


def bottleneck(rho: float, kappa: float = 1.0) -> float:
    """THEOREM BGR. B_kappa(rho) = min(rho, g/kappa).

    Performance is limited by whichever capacity is scarcer, with the guard
    expressed in refinement-equivalent units. Unique maximum where the two are
    equal, i.e. kappa*rho = 1 - 2rho, i.e. rho = 1/(2+kappa).
    """
    _check(rho)
    return min(rho, guard_width(rho) / kappa)


# ---------------------------------------------------------------- measures
def retained_measure(rho: float, n: int) -> float:
    """mu(K_n) = (2 rho)^n.  Cantor: (2/3)^n."""
    _check(rho)
    return (2.0 * rho) ** n


def new_guard_measure(rho: float, n: int) -> float:
    """mu(G_{n+1}) = g * (2 rho)^n -- support added by refining n -> n+1.

    At level k there are 2^(k-1) gaps of width rho^(k-1) * g, so the level-(n+1)
    total is 2^n * rho^n * g. Cantor: (1/3)(2/3)^n.
    """
    _check(rho)
    return guard_width(rho) * (2.0 * rho) ** n


def cumulative_coverage(rho: float, n: int) -> float:
    """mu(S_n) = 1 - (2 rho)^n."""
    return 1.0 - retained_measure(rho, n)


def new_coverage_argmax(n: int) -> float:
    """Maximiser of F_n(rho) = (1-2rho)(2rho)^n.

    With x = 2rho, F = (1-x)x^n and F'(x) = x^(n-1)[n - (n+1)x], so x* =
    n/(n+1) and rho* = n/(2(n+1)).

    NOTE, and it is reported rather than buried: this is 1/3 only at n = 2.
    Cantor does NOT maximise the new-coverage increment in general -- it tends
    to 1/2 as n grows. Cantor's optimality is a BALANCE result, not a
    single-metric one.
    """
    if n < 1:
        raise ValueError("n >= 1")
    return n / (2.0 * (n + 1.0))
