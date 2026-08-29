"""Python port of `src/v2/CantorBarrier.jl`.

The two implementations are checked against each other in
`llm/tests/test_barrier_matches_julia.py`: the Julia side writes a reference
table of (r, V(r), V'(r)) and the Python side must reproduce it to 1e-12.
Keeping two independent implementations in step is the only thing that makes a
cross-language claim ("the same controller was used in the synthetic study and
in the LLM study") checkable rather than asserted.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
import numpy as np

__all__ = ["Gap", "BarrierLayout", "smoothstep", "dsmoothstep",
           "cantor_gap_list", "layout_from_order", "build_layout",
           "LAYOUT_FAMILIES", "worst_displacement"]


def smoothstep(u):
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def dsmoothstep(u):
    u = np.asarray(u, dtype=float)
    out = 6.0 * u * (1.0 - u)
    return np.where((u <= 0.0) | (u >= 1.0), 0.0, out)


@dataclass(frozen=True)
class Gap:
    level: int
    a: float
    b: float
    @property
    def width(self) -> float: return self.b - self.a
    @property
    def centre(self) -> float: return 0.5 * (self.a + self.b)


def cantor_gap_list(n: int) -> list[Gap]:
    """Every removed middle third to level `n`, sorted left to right.

    Endpoints are built with exact `Fraction` arithmetic and rounded once; the
    right endpoint is set to `a + 3**-k` so each stored width is the correctly
    rounded `3^-k` rather than a difference of nearby floats. Mirrors the Julia
    construction (and the reason for it: V1's conditioning erratum).
    """
    if n < 1:
        raise ValueError("n >= 1")
    out: list[Gap] = []

    def rec(a: Fraction, b: Fraction, k: int):
        if k > n:
            return
        w = (b - a) / 3
        lo = float(a + w)
        out.append(Gap(k, lo, lo + 3.0 ** (-k)))
        rec(a, a + w, k + 1)
        rec(a + 2 * w, b, k + 1)

    rec(Fraction(0), Fraction(1), 1)
    out.sort(key=lambda g: g.a)
    return out


def layout_from_order(gaps: list[Gap], order, n: int) -> list[Gap]:
    """Re-place gaps left to right in `order`, separated by `3^-n` survivors."""
    surv = 3.0 ** (-n)
    out, x = [], 0.0
    for i in order:
        g = gaps[i]
        x += surv
        out.append(Gap(g.level, x, x + g.width))
        x += g.width
    return out


@dataclass
class BarrierLayout:
    """A scale-compensated barrier controller.

    `E0` is the energy budget PER LEVEL, so the total L1 control action is
    `n*E0` (Theorem A + Corollary A.1). Pass `E0 = B_total / n` to compare
    different `n` at a fixed budget.
    """
    gaps: list[Gap]
    n: int
    E0: float
    label: str = ""
    family: str = ""
    las: np.ndarray = field(init=False, repr=False)
    lbs: np.ndarray = field(init=False, repr=False)
    wid: np.ndarray = field(init=False, repr=False)
    est: np.ndarray = field(init=False, repr=False)
    coef: np.ndarray = field(init=False, repr=False)
    cum: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        g = sorted(self.gaps, key=lambda x: x.a)
        self.gaps = g
        self.las = np.array([x.a for x in g], dtype=float)
        self.lbs = np.array([x.b for x in g], dtype=float)
        self.wid = self.lbs - self.las if len(g) else np.zeros(0)
        self.est = np.array([self.E0 / 2.0 ** (x.level - 1) for x in g], dtype=float)
        self.coef = self.est / self.wid if len(g) else np.zeros(0)
        self.cum = np.concatenate([[0.0], np.cumsum(self.est)]) if len(g) else np.zeros(1)

    # -- field and potential, vectorised over r ------------------------------
    def field(self, r):
        """V'(r) >= 0. The applied control is `-eta*V'(r)`, always toward safety."""
        r = np.atleast_1d(np.asarray(r, dtype=float))
        out = np.zeros_like(r)
        if len(self.gaps) == 0:
            return out
        i = np.searchsorted(self.las, r, side="right") - 1
        ok = (i >= 0)
        ii = np.where(ok, i, 0)
        inside = ok & (r < self.lbs[ii])
        u = np.zeros_like(r)
        u[inside] = (r[inside] - self.las[ii[inside]]) / self.wid[ii[inside]]
        out[inside] = self.coef[ii[inside]] * dsmoothstep(u[inside])
        return out

    def potential(self, r):
        r = np.atleast_1d(np.asarray(r, dtype=float))
        if len(self.gaps) == 0:
            return np.zeros_like(r)
        i = np.searchsorted(self.las, r, side="right") - 1
        ok = (i >= 0)
        ii = np.where(ok, i, 0)
        out = np.where(ok, self.cum[ii + 1], 0.0)
        inside = ok & (r < self.lbs[ii])
        u = (r[inside] - self.las[ii[inside]]) / self.wid[ii[inside]]
        out[inside] = self.cum[ii[inside]] + self.est[ii[inside]] * smoothstep(u)
        return out

    def peak_of_level(self, k: int) -> float:
        """Theorem B: ||V'_k||_inf = 3*E0*(3/2)^k."""
        return 3.0 * self.E0 * 1.5 ** k

    def total_action(self) -> float:
        """Theorem A + Corollary A.1: n*E0 = sum of all per-gap energies."""
        return float(self.est.sum())


LAYOUT_FAMILIES = ["L0_none", "L1_constant", "L2_central", "L3_periodic",
                   "L4_random", "L5_shuffled", "L6_center_anchored", "L7_cantor"]


def build_layout(family: str, n: int, E0: float, seed: int = 0) -> BarrierLayout:
    """Dispatch identical to the Julia `build_layout` (families renamed L*)."""
    rng = np.random.default_rng(seed)
    if family == "L0_none":
        return BarrierLayout([], n, E0, "none", family)
    if family == "L1_constant":
        # same total action, spread over the whole coordinate
        return BarrierLayout([Gap(1, 0.0, 1.0)], n, n * E0, "constant", family)
    if family == "L2_central":
        return BarrierLayout([Gap(1, 1/3, 2/3)], n, n * E0, "central", family)
    gaps = cantor_gap_list(n)
    m = len(gaps)
    if family == "L7_cantor":
        return BarrierLayout(gaps, n, E0, f"cantor_n{n}", family)
    if family == "L3_periodic":
        order = sorted(range(m), key=lambda i: (gaps[i].level, gaps[i].a))
        return BarrierLayout(layout_from_order(gaps, order, n), n, E0,
                             f"periodic_n{n}", family)
    if family == "L5_shuffled":
        return BarrierLayout(layout_from_order(gaps, rng.permutation(m), n), n, E0,
                             f"shuffled_n{n}", family)
    if family == "L4_random":
        idx = rng.permutation(m)
        tot = sum(gaps[i].width for i in idx)
        cuts = np.sort(rng.random(m) * (1.0 - tot))
        out, x = [], 0.0
        for j, i in enumerate(idx):
            lo = cuts[j] + x
            out.append(Gap(gaps[i].level, lo, lo + gaps[i].width))
            x += gaps[i].width
        return BarrierLayout(out, n, E0, f"random_n{n}", family)
    if family == "L6_center_anchored":
        # keep the level-1 gap centred on r = 1/2; randomise everything else.
        # Per level k>=2, exactly half the gaps go each side, which makes the
        # two side lengths exactly what Cantor gives them (see the Julia
        # docstring), so the anchor lands on 1/2 with no slack.
        i1 = next(i for i in range(m) if gaps[i].level == 1)
        left, right = [], []
        for k in range(2, n + 1):
            idx = [i for i in range(m) if gaps[i].level == k]
            rng.shuffle(idx)
            h = len(idx) // 2
            left += idx[:h]; right += idx[h:]
        rng.shuffle(left); rng.shuffle(right)
        order = left + [i1] + right
        return BarrierLayout(layout_from_order(gaps, order, n), n, E0,
                             f"canchor_n{n}", family)
    raise ValueError(f"unknown layout family: {family}")


def worst_displacement(L: BarrierLayout, kstar: int) -> float:
    """Proposition E: worst-case distance to the first gap of level >= kstar."""
    pk = np.sort(np.array([g.centre for g in L.gaps if g.level >= kstar]))
    if pk.size == 0:
        return 1.0
    return float(max(pk[0], np.max(np.diff(pk)) if pk.size > 1 else 0.0, 1.0 - pk[-1]))
