"""V3.3.4 PHASE 1/3/4 -- EXACT recursive geometry and the cross-leaf theorem.

V3.3.3 computed crossing distances by stepping outward in increments of
(leaf width)*1e-3 with a max_steps cap. That is a numerical search, so calling
it "exact" was wrong. Here every leaf and guard endpoint is computed in closed
form from its binary address, and the cross-leaf distance is a comparison of
sorted interval endpoints -- no stepping, no tolerance, no iteration cap.

THEOREM CR (cross-leaf robustness). For the symmetric binary partition
[rho | 1-2rho | rho] at depth n, let L_n^o be the union of terminal-leaf
INTERIORS and

    d_leaf(r) = inf{ |delta| : r + delta lies in a DIFFERENT terminal leaf }.

Then

    M_n(rho) = inf_{r in L_n^o} d_leaf(r) = rho^(n-1) (1 - 2 rho) = G_n(rho).

Proof sketch. Any path from one terminal leaf to another must cross at least
one separating guard, and the narrowest separating guard at depth n is a
level-n guard of width rho^(n-1)(1-2rho): level-k guards have width
rho^(k-1)(1-2rho), which is decreasing in k for rho < 1/2. A point in a leaf
interior adjacent to a level-n guard can be taken arbitrarily close to that
guard's edge, so the distance can be made arbitrarily close to the guard width
but -- because the interior is OPEN -- never attains it. Hence the quantity is
an INFIMUM, not a minimum, and equals G_n(rho).

COROLLARY CR.1. M_3(rho) = rho^2(1-2rho), M_3'(rho) = 2 rho (1 - 3 rho), which
is positive on (0,1/3), zero at 1/3 and negative on (1/3,1/2). So rho = 1/3 is
the UNIQUE maximiser of the worst-case direct terminal-policy transition margin
at depth three, with M_3(1/3) = 1/27.
"""
from __future__ import annotations
from fractions import Fraction
import numpy as np

__all__ = ["leaves", "guards", "M_n", "dM_n", "rho_max", "M_n_max",
           "d_cross_exact", "classify_exact", "RHO_CANTOR"]

RHO_CANTOR = 1.0 / 3.0


def _rec(lo, span, k, n, rho, g, out_leaf, out_guard, addr):
    a = lo + span * rho
    b = a + span * g
    out_guard.append((k, a, b))
    if k == n:
        out_leaf.append((lo, a, addr << 1))
        out_leaf.append((b, lo + span, (addr << 1) | 1))
        return
    _rec(lo, span * rho, k + 1, n, rho, g, out_leaf, out_guard, addr << 1)
    _rec(b, span * rho, k + 1, n, rho, g, out_leaf, out_guard, (addr << 1) | 1)


def _build(rho, n, exact=False):
    one = Fraction(1) if exact else 1.0
    r = Fraction(rho).limit_denominator(10**9) if exact else float(rho)
    g = one - 2 * r
    L, G = [], []
    _rec(0 * one, one, 1, n, r, g, L, G, 0)
    L.sort(key=lambda t: t[0])
    return L, G


def leaves(rho: float, n: int, exact: bool = False):
    """Closed-form terminal leaf intervals [lo, hi] with binary address."""
    return _build(rho, n, exact)[0]


def guards(rho: float, n: int, exact: bool = False):
    """All guard intervals (level, a, b)."""
    return _build(rho, n, exact)[1]


def M_n(rho, n: int):
    """THEOREM CR: the worst-case cross-leaf margin (an infimum)."""
    r = np.asarray(rho, float)
    return r ** (n - 1) * (1.0 - 2.0 * r)


def dM_n(rho, n: int):
    r = np.asarray(rho, float)
    return r ** (n - 2) * ((n - 1) - 2.0 * n * r)


def rho_max(n: int) -> float:
    """argmax M_n = (n-1)/(2n).  n=2 -> 1/4, n=3 -> 1/3, n=5 -> 2/5."""
    if n < 2:
        raise ValueError("n >= 2")
    return (n - 1) / (2.0 * n)


def M_n_max(n: int) -> float:
    r = rho_max(n)
    return float(r ** (n - 1) * (1 - 2 * r))


def classify_exact(r, rho: float, n: int, _cache={}):
    """('leaf', addr, lo, hi) or ('guard', level, a, b). Closed form."""
    key = (round(float(rho), 12), n)
    if key not in _cache:
        _cache[key] = _build(rho, n)
    L, G = _cache[key]
    x = float(r)
    for lo, hi, addr in L:
        if lo <= x < hi:
            return ("leaf", addr, lo, hi)
    for k, a, b in G:
        if a <= x < b:
            return ("guard", k, a, b)
    lo, hi, addr = L[-1]
    return ("leaf", addr, lo, hi) if abs(x - hi) < 1e-15 else ("guard", 0, 0.0, 0.0)


def d_cross_exact(r, rho: float, n: int, _cache={}):
    """EXACT distance to the nearest DIFFERENT terminal leaf. No stepping.

    From a point in leaf j the nearest other leaf is an adjacent one, so the
    answer is min over the two neighbours of the gap between r and that
    neighbour's near endpoint. Points not in a leaf return 0.0.
    """
    key = (round(float(rho), 12), n)
    if key not in _cache:
        _cache[key] = _build(rho, n)
    L, _ = _cache[key]
    x = float(r)
    for i, (lo, hi, addr) in enumerate(L):
        if lo <= x < hi:
            cand = []
            if i > 0:
                cand.append(x - L[i - 1][1])          # to previous leaf's hi
            if i + 1 < len(L):
                cand.append(L[i + 1][0] - x)          # to next leaf's lo
            return float(min(cand)) if cand else float("inf")
    return 0.0
