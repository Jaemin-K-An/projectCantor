"""V3.3.1 PHASE 13 -- the hierarchical guard controller (SYSTEM B).

SYSTEM A is the smooth multiscale barrier of V2/V3/V3.3. SYSTEM B, defined
here, is the discrete abstraction the guard-resolution theorems are about:

    descend the cell;  if r lands in the central guard -> act conservatively
                       if r lands in a child          -> recurse
                       at depth n                     -> leaf policy

The two are kept SEPARATE (harness section 14). Theorems G/BGR/P are geometry
statements about SYSTEM B; the bridge to SYSTEM A is that the Cantor barrier's
removed middle third at each level IS the guard zone of SYSTEM B, and the
surviving intervals are the refinement zones.
"""
from __future__ import annotations
import numpy as np

__all__ = ["GuardController", "GUARD", "LEAF"]

GUARD, LEAF = 1, 0


class GuardController:
    """rho-family hierarchical guard policy on [0,1]."""

    def __init__(self, rho: float, n: int):
        if not (0.0 < rho < 0.5):
            raise ValueError("rho in (0, 1/2)")
        self.rho, self.n = float(rho), int(n)
        self.g = 1.0 - 2.0 * self.rho

    def classify(self, r):
        """Return (kind, level, leaf_address).

        kind = GUARD  -> conservative/abstain action at that level
        kind = LEAF   -> survived to depth n; leaf_address is the L/R path,
                         i.e. WHICH child policy would be applied
        """
        rr = np.atleast_1d(np.asarray(r, float))
        kind = np.zeros(len(rr), dtype=np.int8)
        level = np.zeros(len(rr), dtype=np.int16)
        addr = np.zeros(len(rr), dtype=np.int64)
        for i, x0 in enumerate(rr):
            x = float(x0)
            lo, span, a_ = 0.0, 1.0, 0
            if not (0.0 <= x <= 1.0):
                kind[i], level[i], addr[i] = GUARD, 0, -1
                continue
            for k in range(1, self.n + 1):
                a = lo + span * self.rho
                b = a + span * self.g
                if x < a:
                    span *= self.rho
                    a_ = (a_ << 1)
                elif x >= b:
                    lo, span = b, span * self.rho
                    a_ = (a_ << 1) | 1
                else:
                    kind[i], level[i], addr[i] = GUARD, k, -1
                    break
            else:
                kind[i], level[i], addr[i] = LEAF, 0, a_
        return kind, level, addr

    def guard_measure(self) -> float:
        """Total measure occupied by guard zones down to depth n."""
        return sum((2 ** (k - 1)) * (self.rho ** (k - 1) * self.g)
                   for k in range(1, self.n + 1))

    def leaf_measure(self) -> float:
        return (2.0 * self.rho) ** self.n

    def leaf_width(self) -> float:
        """Width of a single depth-n leaf = the achieved resolution."""
        return self.rho ** self.n
