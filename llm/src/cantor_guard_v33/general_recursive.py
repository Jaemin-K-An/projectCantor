"""V3.3 PHASE 6/8 -- general recursive barriers, with Cantor as one point.

Cantor is not the only self-similar layout, and V3.3 must not confuse "recursion
helps" with "Cantor helps" (harness section 35: CASE A vs CASE B). So the
theory is derived for a general iterated function system and Cantor is
recovered as the special case (b, rho) = (2, 1/3).

SETUP. Take b affine contractions of ratio rho on [0,1], images disjoint and
evenly spaced. Feasibility requires b*rho < 1. The complement of the images is
(b-1) equal gaps of width

    g = (1 - b*rho) / (b - 1)

Recursing gives, at level k:

    N_k = (b-1) * b^(k-1)            gap count
    w_k = rho^(k-1) * g              gap width
    e_k = E0 / N_k                   per-gap energy (level total = E0)

and with the same smoothstep profile as the Cantor controller:

    peak_k  = 1.5 * e_k / w_k        (max of 6u(1-u) is 1.5)
    slope_k = 6.0 * e_k / w_k^2      (max of |6-12u| is 6)

The two scale factors that matter are

    alpha_field = 1 / (b * rho)      V'_n(T_i(r)) = alpha_field * V'_{n-1}(r)
    alpha_pot   = 1 / b              potential contraction

CANTOR CHECK (b=2, rho=1/3): g = 1/3, N_k = 2^(k-1), w_k = 3^-k,
alpha_field = 3/2, peak_k = 3*E0*(3/2)^k (Theorem B),
slope_k = 12*E0*(9/2)^k (Theorem T). All three recovered exactly.

Note 1/(b*rho^2) is the sensitivity scale: Cantor gives 9/2.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .symbolic_cantor import smoothstep, dsmoothstep, d2smoothstep

__all__ = ["IFSSpec", "CANTOR", "SymbolicIFS", "feasible"]


def feasible(b: int, rho: float) -> bool:
    """Images must fit inside [0,1] without overlapping."""
    return b >= 2 and 0.0 < rho and b * rho < 1.0


@dataclass(frozen=True)
class IFSSpec:
    """A recursive barrier family. `label` is used in reports."""
    b: int
    rho: float
    label: str = ""

    def __post_init__(self):
        if not feasible(self.b, self.rho):
            raise ValueError(f"infeasible IFS: b={self.b} rho={self.rho} "
                             f"(need b*rho < 1)")

    # ---- geometry -------------------------------------------------------
    @property
    def gap_width_1(self) -> float:
        return (1.0 - self.b * self.rho) / (self.b - 1)

    @property
    def stride(self) -> float:
        return self.rho + self.gap_width_1

    def n_gaps_at_level(self, k: int) -> int:
        return (self.b - 1) * self.b ** (k - 1)

    def gap_width(self, k: int) -> float:
        return self.rho ** (k - 1) * self.gap_width_1

    def gap_energy(self, k: int, E0: float) -> float:
        return E0 / self.n_gaps_at_level(k)

    def n_components(self, n: int) -> int:
        return sum(self.n_gaps_at_level(k) for k in range(1, n + 1))

    # ---- scale laws -----------------------------------------------------
    @property
    def alpha_field(self) -> float:
        return 1.0 / (self.b * self.rho)

    @property
    def alpha_potential(self) -> float:
        return 1.0 / self.b

    @property
    def alpha_sensitivity(self) -> float:
        return 1.0 / (self.b * self.rho ** 2)

    def peak_of_level(self, k: int, E0: float) -> float:
        return 1.5 * self.gap_energy(k, E0) / self.gap_width(k)

    def slope_of_level(self, k: int, E0: float) -> float:
        return 6.0 * self.gap_energy(k, E0) / self.gap_width(k) ** 2

    def total_action(self, n: int, E0: float) -> float:
        return n * E0          # level-wise total is E0 by construction


CANTOR = IFSSpec(2, 1.0 / 3.0, "cantor")


class SymbolicIFS:
    """Symbolic evaluator for any `IFSSpec` -- O(n) query, O(1) storage.

    Same descent as `symbolic_cantor`, generalised: locate r among the b images
    and (b-1) gaps of the current cell; if it lands in a gap, stop; if it lands
    in an image, rescale and recurse.
    """

    def __init__(self, spec: IFSSpec, n: int, E0: float):
        self.spec, self.n, self.E0 = spec, int(n), float(E0)
        self.family = f"ifs_b{spec.b}_rho{spec.rho:.4f}"

    def _descend(self, r: float):
        s, rho = self.spec.stride, self.spec.rho
        y = r
        for k in range(1, self.n + 1):
            i = int(np.floor(y / s + 1e-12))
            i = min(max(i, 0), self.spec.b - 1)
            start = i * s
            if y <= start + rho:                      # inside image i
                y = (y - start) / rho
                continue
            if i >= self.spec.b - 1:                  # past the last image
                return 0, 0.0
            a = start + rho
            return k, (y - a) / self.spec.gap_width_1
        return 0, 0.0

    def field(self, r):
        rr = np.atleast_1d(np.asarray(r, dtype=float))
        out = np.zeros_like(rr)
        for i, x in enumerate(rr):
            k, u = self._descend(float(x))
            if k:
                coef = (self.spec.gap_energy(k, self.E0)
                        / self.spec.gap_width(k))
                out[i] = coef * float(dsmoothstep(u))
        return out

    def field_derivative(self, r):
        rr = np.atleast_1d(np.asarray(r, dtype=float))
        out = np.zeros_like(rr)
        for i, x in enumerate(rr):
            k, u = self._descend(float(x))
            if k:
                coef = (self.spec.gap_energy(k, self.E0)
                        / self.spec.gap_width(k) ** 2)
                out[i] = coef * float(d2smoothstep(u))
        return out

    def level(self, r):
        rr = np.atleast_1d(np.asarray(r, dtype=float))
        return np.array([self._descend(float(x))[0] for x in rr], dtype=int)

    def storage_words(self) -> int:
        return 4               # b, rho, n, E0

    def n_implied_components(self) -> int:
        return self.spec.n_components(self.n)
