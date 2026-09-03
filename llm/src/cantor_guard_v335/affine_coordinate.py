"""V3.3.5 PHASE 1/3 -- the affine safety coordinate, forced by Theorem CP.

WHY AFFINE, AND WHY THIS IS NOT RIGGING THE ANSWER. The design requirement is
stated first and independently of any rho: the Euclidean margin of the Cantor
partition must be preserved in the residual projection by ONE position-
independent constant. Theorem CP shows that requirement alone forces an affine
map. The logistic coordinate used through V3.3.4 does not satisfy it, and that
is exactly why V3.3.4's EXACT z-space optimum drifted to rho ~ 0.296 while the
r-space optimum stayed at 1/3.

THEOREM CP. Let phi: I -> [0,1] be continuous and strictly monotone on an
interval I. If there is a single constant c > 0 with

    |phi(z2) - phi(z1)| = c |z2 - z1|    for all z1, z2 in I,

then phi is affine: phi(z) = a + s c z for some fixed s in {-1, +1}.

Proof. Strict monotonicity fixes the sign of phi(z2) - phi(z1) relative to
z2 - z1, so phi(z2) - phi(z1) = s c (z2 - z1) with s constant on I. Fixing any
z0 and letting z vary gives phi(z) = phi(z0) + s c (z - z0), which is affine.
The logistic map fails the hypothesis because |dr/dz| = (gamma/sigma) r(1-r)
depends on position. QED

OPERATING WINDOW. The affine map is defined on I = [tau - W, tau + W]:

    r_aff(z) = 1/2 + s (z - tau) / (2W)

so r(tau) = 1/2 exactly, the endpoints map exactly to 0 and 1, and

    |dr/dz| = 1 / (2W)    exactly, everywhere in I.

OUTSIDE THE WINDOW there is NO CLIPPING. Clamping z to the window edge would
map an entire half-line onto a single terminal leaf, destroying the Euclidean
correspondence the certificate rests on. Out-of-window states are a distinct
OUTSIDE state handled by a conservative fallback.
"""
from __future__ import annotations
import numpy as np

__all__ = ["AffineCoordinate", "OUTSIDE", "choose_W"]

OUTSIDE = "OUTSIDE_WINDOW"
W_QUANTILE, W_PADDING = 0.99, 1.05


def choose_W(z_clean, tau: float, quantile: float = W_QUANTILE,
             padding: float = W_PADDING) -> float:
    """FROZEN rule: W = padding * Q_quantile(|z_clean - tau|).

    W is not a performance knob -- the certificate 2W*M_n scales linearly with
    it, so tuning W would trivially inflate every controller's radius at once
    without changing any ranking. It is fixed from calibration coverage before
    any controller comparison and is identical for every rho.
    """
    d = np.abs(np.asarray(z_clean, float) - float(tau))
    return float(padding * np.quantile(d, quantile))


class AffineCoordinate:
    """r_aff(z) on [tau-W, tau+W]; OUTSIDE beyond it."""

    def __init__(self, tau: float, W: float, orientation: int = +1):
        if W <= 0:
            raise ValueError("W > 0")
        if orientation not in (-1, +1):
            raise ValueError("orientation must be -1 or +1")
        self.tau, self.W, self.s = float(tau), float(W), int(orientation)

    def inside(self, z):
        z = np.asarray(z, float)
        return np.abs(z - self.tau) <= self.W

    def r(self, z):
        """Affine coordinate; NaN outside the window (never clipped)."""
        z = np.atleast_1d(np.asarray(z, float))
        out = np.full(z.shape, np.nan)
        m = self.inside(z)
        out[m] = 0.5 + self.s * (z[m] - self.tau) / (2.0 * self.W)
        return out

    def z_of_r(self, r):
        """Exact inverse on the window."""
        r = np.asarray(r, float)
        return self.tau + self.s * (r - 0.5) * (2.0 * self.W)

    def slope(self) -> float:
        """|dr/dz| = 1/(2W), constant -- the whole point of Theorem CP."""
        return 1.0 / (2.0 * self.W)

    def dz_for_dr(self, dr: float) -> float:
        """Exact conversion, not a bound: |dz| = 2W |dr|."""
        return 2.0 * self.W * float(dr)
