"""V3.3.2 PHASE 2/3 -- absolute-uncertainty guard geometry.

V3.3.1's bridge used delta = beta * rho^n, which makes the uncertainty a
FUNCTION OF THE CONTROLLER. That is fine as a scale-invariant design model
(U1), but it cannot be used to ask what an independently measured LLM
uncertainty implies, because the answer would be built into the question.

Here the uncertainty is an ABSOLUTE width delta > 0 in threat-coordinate units,
estimated from the model with no reference to rho (U2), and compared against the
finest guard the depth-n controller actually provides:

    G_n(rho) = rho^(n-1) * (1 - 2 rho)

THEOREM AG.  G_n'(rho) = rho^(n-2) * [(n-1) - 2 n rho], so on (0, 1/2) the
finest guard width is maximised uniquely at

    rho_guard_max(n) = (n-1) / (2n),      G_n,max = (1/n) [(n-1)/(2n)]^(n-1)

  COROLLARY AG.1  n = 3 gives rho_guard_max = 1/3 exactly: the middle third
  uniquely maximises the finest guard width at depth 3, with G_3,max = 1/27.

  COUNTEREXAMPLE  rho_guard_max > 1/3 exactly when n > 3 (3(n-1) > 2n <=> n > 3).
  At n = 5 it is 0.4. Cantor does NOT maximise G_n for n > 3, and this module
  reports that rather than burying it.

THEOREM AU.  For fixed delta > 0 the feasible set {rho : G_n(rho) >= delta} is
empty when delta > G_n,max, and otherwise an interval [rho_lo, rho_hi] bounded
by the two roots of G_n(rho) = delta. Since retention 2*rho increases in rho,
the retention-maximal feasible ratio is the RIGHTMOST root,

    rho_abs*(n, delta) = max { rho in (0, 1/2) : G_n(rho) >= delta }

  COROLLARY AU.1  n = 2, delta = 1/9:  rho(1-2rho) = 1/9  <=>  18 rho^2 - 9 rho
  + 1 = 0  <=>  rho in {1/6, 1/3}. The rightmost feasible ratio is exactly 1/3.

Note what AU.1 does and does not say. It says that IF the measured uncertainty
is near 1/9, THEN the middle third is the retention-maximal depth-2 solution --
derived from the uncertainty, with no kappa = 1 assumption anywhere. Whether the
measured uncertainty IS near 1/9 is an empirical question this module does not
answer.
"""
from __future__ import annotations
import math
import numpy as np

__all__ = ["G_n", "dG_n", "rho_guard_max", "G_n_max", "feasible_interval",
           "rho_abs_star", "cantor_guard_width", "RHO_CANTOR"]

RHO_CANTOR = 1.0 / 3.0
_LO, _HI = 1e-9, 0.5 - 1e-9


def G_n(rho, n: int):
    """Finest (level-n) guard width."""
    r = np.asarray(rho, dtype=float)
    return r ** (n - 1) * (1.0 - 2.0 * r)


def dG_n(rho, n: int):
    """G_n'(rho) = rho^(n-2) [(n-1) - 2 n rho]."""
    r = np.asarray(rho, dtype=float)
    return r ** (n - 2) * ((n - 1) - 2.0 * n * r)


def rho_guard_max(n: int) -> float:
    """THEOREM AG. Unique maximiser of G_n on (0, 1/2)."""
    if n < 2:
        raise ValueError("n >= 2 (at n=1 the guard is 1-2rho, maximised as rho->0)")
    return (n - 1) / (2.0 * n)


def G_n_max(n: int) -> float:
    r = rho_guard_max(n)
    return float(r ** (n - 1) * (1.0 - 2.0 * r))


def cantor_guard_width(n: int) -> float:
    """G_n(1/3) = 3^-n."""
    return 3.0 ** (-n)


def feasible_interval(n: int, delta: float, tol: float = 1e-14):
    """THEOREM AU. [rho_lo, rho_hi] with G_n >= delta, or None if infeasible.

    G_n rises then falls with a single interior maximum, so a bisection on each
    side of the peak brackets the two roots exactly.
    """
    if delta <= 0:
        return (_LO, _HI)
    peak = rho_guard_max(n)
    if G_n(peak, n) < delta:
        return None                       # delta exceeds the widest guard

    def bisect(lo, hi):
        # `lo` is the infeasible end, `hi` the feasible one. The right-hand root
        # is bracketed with lo > hi, so the width test must be on |hi - lo|;
        # using hi - lo breaks instantly there and returns the first midpoint.
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if (G_n(mid, n) >= delta) == (G_n(lo, n) >= delta):
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) < tol:
                break
        return 0.5 * (lo + hi)

    return (bisect(_LO, peak), bisect(_HI, peak))


def rho_abs_star(n: int, delta: float):
    """Retention-maximal feasible ratio = the RIGHTMOST root (retention is 2rho,
    strictly increasing, so the largest feasible rho wins)."""
    iv = feasible_interval(n, delta)
    return None if iv is None else iv[1]
