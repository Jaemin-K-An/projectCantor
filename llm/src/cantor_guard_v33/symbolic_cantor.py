"""V3.3 PHASE 4 -- symbolic Cantor controller: O(n) query, O(1) storage.

The explicit controller materialises every removed interval: 2^n - 1 gaps at
order n, so storage is Theta(2^n) and a point query is a binary search over
that list. But the Cantor layout is not an arbitrary list of intervals -- it is
the attractor of two affine maps, and Theorem S says the field transforms
exactly under them:

    T_0(r) = r/3,  T_2(r) = 2/3 + r/3
    V'_n(T_i(r)) = (3/2) V'_{n-1}(r)

So a query can descend the ternary address of r instead of searching a list.
That is what this module does. Nothing here changes the controller's VALUE --
it is the same function, computed a different way, and
`validate_symbolic_against_explicit.py` checks that to machine precision.

CONVENTION (matches cantor_barrier.BarrierLayout exactly):
  gap at level k:  width w_k = 3^-k,  per-gap energy e_k = E0 / 2^(k-1)
  field           V'(r) = (e_k / w_k) * 6u(1-u),  u = (r - a)/w_k
  potential       V(r)  = integral_0^r V'  (so V(1) = n*E0, Theorem A)

The potential therefore carries an OFFSET: everything to the LEFT of r counts.
V3 got this wrong once (docs/v3_1 D4); the offset is derived and tested here.
"""
from __future__ import annotations
from fractions import Fraction
import numpy as np

__all__ = ["smoothstep", "dsmoothstep", "cantor_level", "cantor_field",
           "cantor_potential", "cantor_field_derivative", "SymbolicCantor",
           "N_GAPS", "total_action", "peak_of_level", "slope_of_level"]

# Scale factors, from Theorem S with (b, rho) = (2, 1/3):
ALPHA_FIELD = 1.5          # V'_n(T_i(r))     = (3/2) V'_{n-1}(r)
ALPHA_POT = 0.5            # V_n(T_i(r)) - V_n(T_i(0)) = (1/2) V_{n-1}(r)
ALPHA_DERIV = 4.5          # V''_n(T_i(r))    = (9/2) V''_{n-1}(r)


def smoothstep(u):
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def dsmoothstep(u):
    u = np.asarray(u, dtype=float)
    out = 6.0 * u * (1.0 - u)
    return np.where((u <= 0.0) | (u >= 1.0), 0.0, out)


def d2smoothstep(u):
    u = np.asarray(u, dtype=float)
    out = 6.0 - 12.0 * u
    return np.where((u <= 0.0) | (u >= 1.0), 0.0, out)


def N_GAPS(n: int) -> int:
    """Materialised component count: 2^n - 1."""
    return (1 << n) - 1


def total_action(n: int, E0: float) -> float:
    """Theorem A + Corollary A.1."""
    return n * E0


def peak_of_level(k: int, E0: float) -> float:
    """Theorem B: 3*E0*(3/2)^k."""
    return 3.0 * E0 * 1.5 ** k


def slope_of_level(k: int, E0: float) -> float:
    """Theorem T: 12*E0*(9/2)^k."""
    return 12.0 * E0 * 4.5 ** k


def _descend(r: float, n: int):
    """Ternary descent. Returns (level, u) if r lies in a removed gap of order
    <= n, else (0, 0.0) meaning r survives in K_n and the field is zero.

    At most n iterations, O(1) extra storage -- no gap list is built.

    CONDITIONING. The cell is tracked as an EXACT ternary address, an integer
    numerator p over 3^(k-1), never as a pair of floats. Repeatedly dividing a
    float by 3 loses a little each level, and the field coefficient grows like
    2*(3/2)^k, so that loss is amplified: the float version drifted to a
    relative 3e-8 against the explicit layout by n=15. With the integer address
    the endpoint is formed by one correctly-rounded division, exactly as
    `cantor_gap_list` forms it from a Fraction, so the two agree to rounding.
    """
    if not (0.0 <= r <= 1.0):
        return 0, 0.0
    p, pw = 0, 1                     # cell is [p/pw, (p+1)/pw], pw = 3^(k-1)
    for k in range(1, n + 1):
        pw3 = pw * 3
        a_num = 3 * p + 1            # gap is [(3p+1)/3^k, (3p+2)/3^k]
        a = a_num / pw3              # one correctly-rounded division
        w = 3.0 ** (-k)
        b = a + w                    # matches cantor_gap_list's construction
        if r < a:
            p, pw = 3 * p, pw3
        elif r >= b:
            p, pw = 3 * p + 2, pw3
        else:
            return k, (r - a) / w
    return 0, 0.0


def cantor_level(r, n: int):
    """Level of the gap containing r (0 if r is in the surviving set)."""
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    return np.array([_descend(float(x), n)[0] for x in rr], dtype=int)


def cantor_field(r, n: int, E0: float):
    """V'_n(r) >= 0, computed symbolically in O(n) per point.

    Equivalent to BarrierLayout(cantor_gap_list(n), n, E0).field(r).
    """
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    out = np.zeros_like(rr)
    for i, x in enumerate(rr):
        k, u = _descend(float(x), n)
        if k:
            coef = (E0 / 2.0 ** (k - 1)) / 3.0 ** (-k)
            out[i] = coef * float(dsmoothstep(u))
    return out


def cantor_field_derivative(r, n: int, E0: float):
    """V''_n(r); its sup over level k is Theorem T's 12*E0*(9/2)^k."""
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    out = np.zeros_like(rr)
    for i, x in enumerate(rr):
        k, u = _descend(float(x), n)
        if k:
            coef = (E0 / 2.0 ** (k - 1)) / 3.0 ** (-2 * k)
            out[i] = coef * float(d2smoothstep(u))
    return out


def cantor_potential(r, n: int, E0: float):
    """V_n(r) = integral_0^r V'_n, computed symbolically in O(n) per point.

    DERIVATION (this is the offset V3 dropped). Split [0,1] by the level-1 gap:

      r <= 1/3      V_n(r) = (1/2) V_{n-1}(3r)
                    -- the left copy is an affine image carrying half the energy

      1/3 < r < 2/3 V_n(r) = M + E0 * smoothstep(3r - 1)
                    -- M is ALL the energy to the left, i.e. the whole left copy

      r >= 2/3      V_n(r) = M + E0 + (1/2) V_{n-1}(3r - 2)

    with M = (1/2) * V_{n-1}(1) = (1/2)(n-1)E0. Check: V_n(1) = M + E0 + M
    = (n-1)E0 + E0 = n*E0, which is Theorem A.
    """
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    out = np.zeros_like(rr)
    for i, x in enumerate(rr):
        v, scale, off, depth, y = 0.0, 1.0, 0.0, n, float(x)
        while depth >= 1:
            if y <= 1.0 / 3.0:
                scale *= ALPHA_POT
                y *= 3.0
                depth -= 1
            elif y >= 2.0 / 3.0:
                # left copy of this sub-level is fully passed, plus its own gap
                off += scale * (0.5 * (depth - 1) * E0 + E0)
                scale *= ALPHA_POT
                y = 3.0 * y - 2.0
                depth -= 1
            else:
                off += scale * (0.5 * (depth - 1) * E0
                                + E0 * float(smoothstep(3.0 * y - 1.0)))
                v = 0.0
                break
        out[i] = off + scale * v
    return out


class SymbolicCantor:
    """Controller interface matching BarrierLayout, without materialising gaps.

    `storage_words()` reports what the representation actually holds, which is
    the point of the whole module: it does not grow with n.
    """

    def __init__(self, n: int, E0: float):
        if n < 1:
            raise ValueError("n >= 1")
        self.n, self.E0 = int(n), float(E0)
        self.family = "cantor_symbolic"

    def field(self, r):
        return cantor_field(r, self.n, self.E0)

    def potential(self, r):
        return cantor_potential(r, self.n, self.E0)

    def level(self, r):
        return cantor_level(r, self.n)

    def total_action(self) -> float:
        return total_action(self.n, self.E0)

    def peak_of_level(self, k: int) -> float:
        return peak_of_level(k, self.E0)

    def storage_words(self) -> int:
        """n and E0. Independent of depth -- contrast N_GAPS(n)*3."""
        return 2

    def n_implied_components(self) -> int:
        return N_GAPS(self.n)
