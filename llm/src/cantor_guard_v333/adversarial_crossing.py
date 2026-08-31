"""V3.3.3 gap 3 -- EXACT minimum crossing distance, not random perturbation.

V3.3.2 perturbed by random +-delta and counted how often the leaf assignment
changed. That measures AVERAGE robustness. The guard theorem is a WORST-CASE
statement, so it cannot be validated that way: a random probe may simply never
try the direction that breaks the guarantee.

Here we compute, for each real coordinate r, the exact minimum perturbation

    d_cross(r; rho, n) = min { |t| : leaf(r + t) != leaf(r) }

by walking the recursive cell decomposition to the two nearest boundaries of
the cell that r occupies. For a point inside a depth-n leaf this is the
distance to the nearer end of that leaf; for a point already inside a guard it
is 0 by convention (the guard has already caught it), and such points are
excluded from the worst-case statistic.

THE THEOREM'S PREDICTION. If every leaf has width rho^n and adjacent leaves are
separated by a guard of width at least G_n(rho) = rho^(n-1)(1-2rho), then a
perturbation smaller than G_n cannot move a point from one leaf to a DIFFERENT
leaf -- it can at most push it into the guard. So

    min over leaf-interior points of d_cross_to_a_DIFFERENT_LEAF  >=  G_n(rho)

and this is what gets tested, on the certified interior only.
"""
from __future__ import annotations
import numpy as np

__all__ = ["cell_of", "leaf_bounds", "d_cross_exact", "d_cross_to_other_leaf"]


def cell_of(r: float, rho: float, n: int):
    """Walk the recursion. Returns (kind, level, lo, hi, address).

    kind = 'guard' with the guard's level, or 'leaf' at depth n.
    lo/hi are the exact bounds of the occupied interval.
    """
    g = 1.0 - 2.0 * rho
    lo, span, addr = 0.0, 1.0, 0
    for k in range(1, n + 1):
        a = lo + span * rho
        b = a + span * g
        if r < a:
            span *= rho; addr = addr << 1
        elif r >= b:
            lo, span, addr = b, span * rho, (addr << 1) | 1
        else:
            return "guard", k, a, b, -1
    return "leaf", n, lo, lo + span, addr


def leaf_bounds(r: float, rho: float, n: int):
    kind, _, lo, hi, addr = cell_of(r, rho, n)
    return (lo, hi, addr) if kind == "leaf" else None


def d_cross_exact(r: float, rho: float, n: int) -> float:
    """Minimum |t| that changes the cell assignment at all (leaf -> guard counts)."""
    kind, _, lo, hi, _ = cell_of(r, rho, n)
    if kind == "guard":
        return 0.0
    return float(min(r - lo, hi - r))


def d_cross_to_other_leaf(r: float, rho: float, n: int,
                          max_steps: int = 4096) -> float:
    """Minimum |t| that lands r in a DIFFERENT leaf, crossing the guard.

    Exact for this geometry: from a leaf, moving outward you must traverse the
    remaining leaf width plus the full adjacent guard before another leaf
    begins. Both directions are examined and the smaller returned. Points
    already in a guard return 0 and are excluded by the caller.
    """
    b = leaf_bounds(r, rho, n)
    if b is None:
        return 0.0
    lo, hi, addr = b
    out = []
    for direction, edge in ((+1.0, hi), (-1.0, lo)):
        # step outward in geometrically shrinking probes until a different leaf
        t = abs(edge - r)
        step = (hi - lo) * 1e-3 + 1e-15
        for _ in range(max_steps):
            t += step
            rr = r + direction * t
            if not (0.0 <= rr <= 1.0):
                t = np.inf; break
            k2 = cell_of(rr, rho, n)
            if k2[0] == "leaf" and k2[4] != addr:
                break
        out.append(t)
    return float(min(out))
