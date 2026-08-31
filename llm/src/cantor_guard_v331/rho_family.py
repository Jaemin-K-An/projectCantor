"""V3.3.1 -- the binary symmetric rho-family of recursive barrier controllers.

Same smoothstep barrier as V2/V3, but the contraction ratio is a free
parameter. rho = 1/3 recovers the Cantor controller exactly; every other rho is
a legitimate member of the same family, which is what makes the comparison a
test of the RATIO rather than a test of "recursion vs not".

Geometry per level k (b = 2 branches):
    gap count   N_k = 2^(k-1)
    gap width   w_k = rho^(k-1) * g,     g = 1 - 2*rho
    gap energy  e_k = E0 / 2^(k-1)       (level total = E0, Theorem A)
    field       V'(r) = (e_k / w_k) * 6u(1-u)

Evaluation is symbolic: O(n) per query, O(1) storage, no materialised list.
"""
from __future__ import annotations
import numpy as np
from .guard_geometry import guard_width, RHO_CANTOR

__all__ = ["RhoBarrier", "rho_gap_list"]


def _dsmooth(u):
    u = np.asarray(u, float)
    return np.where((u <= 0.0) | (u >= 1.0), 0.0, 6.0 * u * (1.0 - u))


def _d2smooth(u):
    u = np.asarray(u, float)
    return np.where((u <= 0.0) | (u >= 1.0), 0.0, 6.0 - 12.0 * u)


class RhoBarrier:
    """Recursive barrier with contraction ratio `rho`, depth `n`, budget `E0`.

    The descent mirrors `symbolic_cantor`: locate r among [left child | guard |
    right child] of the current cell; if it lands in the guard, stop; if in a
    child, rescale and recurse. At rho = 1/3 this is the Cantor controller.
    """

    def __init__(self, rho: float, n: int, E0: float = 1.0):
        if not (0.0 < rho < 0.5):
            raise ValueError("rho in (0, 1/2)")
        self.rho, self.n, self.E0 = float(rho), int(n), float(E0)
        self.g = guard_width(self.rho)
        self.is_cantor = abs(self.rho - RHO_CANTOR) < 1e-12
        self.family = f"rho_{self.rho:.6f}"

    # -- geometry ---------------------------------------------------------
    def gap_width(self, k: int) -> float:
        return self.rho ** (k - 1) * self.g

    def gap_energy(self, k: int) -> float:
        return self.E0 / 2.0 ** (k - 1)

    def n_components(self) -> int:
        return 2 ** self.n - 1

    def peak_of_level(self, k: int) -> float:
        return 1.5 * self.gap_energy(k) / self.gap_width(k)

    def slope_of_level(self, k: int) -> float:
        return 6.0 * self.gap_energy(k) / self.gap_width(k) ** 2

    def total_action(self) -> float:
        return self.n * self.E0

    # -- symbolic descent -------------------------------------------------
    def _descend(self, r: float):
        """Return (level, u) if r is in a guard gap, else (0, 0.0)."""
        if not (0.0 <= r <= 1.0):
            return 0, 0.0
        lo, span = 0.0, 1.0
        for k in range(1, self.n + 1):
            a = lo + span * self.rho              # guard starts here
            b = a + span * self.g                 # guard ends here
            if r < a:
                span *= self.rho
            elif r >= b:
                lo, span = b, span * self.rho
            else:
                return k, (r - a) / (b - a)
        return 0, 0.0

    def level(self, r):
        rr = np.atleast_1d(np.asarray(r, float))
        return np.array([self._descend(float(x))[0] for x in rr], dtype=int)

    def field(self, r):
        rr = np.atleast_1d(np.asarray(r, float))
        out = np.zeros_like(rr)
        for i, x in enumerate(rr):
            k, u = self._descend(float(x))
            if k:
                out[i] = (self.gap_energy(k) / self.gap_width(k)) * float(_dsmooth(u))
        return out

    def field_derivative(self, r):
        rr = np.atleast_1d(np.asarray(r, float))
        out = np.zeros_like(rr)
        for i, x in enumerate(rr):
            k, u = self._descend(float(x))
            if k:
                out[i] = (self.gap_energy(k) / self.gap_width(k) ** 2) * float(_d2smooth(u))
        return out

    def in_guard(self, r):
        """True where the coordinate lies in ANY guard gap up to depth n."""
        return self.level(r) > 0

    # -- controller interface used by the LLM hook ------------------------
    def magnitude(self, m):
        """Same signature as Controller31.magnitude: margin -> |u| >= 0.

        The threat coordinate is the V2/V3 map r = sigmoid(-gamma*m).
        """
        m = np.asarray(m, float)
        r = 1.0 / (1.0 + np.exp(np.clip(self.gamma * m, -60, 60)))
        return self.eta * self.field(r.ravel()).reshape(r.shape)

    gamma = 0.7
    eta = 1.0
    harm_gate = True
    max_q = None
    sup_deriv = 0.0


def rho_gap_list(rho: float, n: int):
    """Explicit gap list, for differential testing against the symbolic path."""
    g = guard_width(rho)
    out = []

    def rec(lo, span, k):
        if k > n:
            return
        a = lo + span * rho
        out.append((k, a, a + span * g))
        rec(lo, span * rho, k + 1)
        rec(a + span * g, span * rho, k + 1)

    rec(0.0, 1.0, 1)
    out.sort(key=lambda t: t[1])
    return out
