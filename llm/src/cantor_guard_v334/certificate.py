"""V3.3.4 PHASE 2 -- residual robustness certificates.

SCOPE, stated before anything else. These certify ONE thing: that a residual
perturbation below the stated radius cannot move the safety coordinate directly
from one terminal policy region to a DIFFERENT terminal policy region, for this
layer, this refusal direction, this behavioural centre and this 1-D controller.
They are NOT certificates of LLM safety, jailbreak immunity, or harmless
generation.

THEOREM L (Lipschitz / projection certificate). With
    r(z) = sigmoid(-gamma (z - tau) / sigma)
we have dr/dz = -(gamma/sigma) r(1-r) and r(1-r) <= 1/4, so
    |Delta r| <= (gamma / (4 sigma)) |Delta z|.
A direct terminal transition needs |Delta r| >= M_n(rho), hence it is impossible
whenever (gamma/(4 sigma))|Delta z| < M_n(rho), i.e.

    |Delta z| < eps_z_cert = (4 sigma / gamma) * M_n(rho).

THEOREM H (full residual L2 certificate). Delta z = <Delta h, v> with ||v|| = 1,
so |Delta z| <= ||Delta h||_2 by Cauchy-Schwarz. Therefore

    ||Delta h||_2 < eps_z_cert   ==>   no direct terminal transition,

and eps_h_cert = eps_z_cert is a valid SUFFICIENT radius for arbitrary residual
perturbations.

EXACT certificate. The Lipschitz bound is conservative: it uses the maximum
slope 1/4, attained only at r = 1/2. Because the transform is strictly monotone
it can be inverted exactly, z(r) = tau - (sigma/gamma) logit(r), so the exact
projection displacement needed to jump a guard [a,b] is
    (sigma/gamma) |logit(b) - logit(a)|
and the exact certificate is the minimum of that over SEPARATING guards.

Whether rho = 1/3 also maximises the EXACT certificate is NOT assumed. The
logit warp is nonlinear and the guards sit at different coordinates, so the
maximiser can move; it is swept numerically and reported either way.
"""
from __future__ import annotations
import numpy as np
from .certified_geometry import M_n, leaves, guards, rho_max

__all__ = ["eps_z_lipschitz", "eps_h_l2", "eps_z_exact", "logit",
           "z_of_r", "separating_guards"]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))


def z_of_r(r, tau: float, sigma: float, gamma: float):
    """Inverse of r = sigmoid(-gamma (z-tau)/sigma)."""
    return tau - (sigma / gamma) * logit(r)


def eps_z_lipschitz(rho: float, n: int, sigma: float, gamma: float) -> float:
    """THEOREM L. (4 sigma / gamma) * M_n(rho)."""
    return float((4.0 * sigma / gamma) * M_n(rho, n))


def eps_h_l2(rho: float, n: int, sigma: float, gamma: float) -> float:
    """THEOREM H. Same radius; sufficient for arbitrary ||Delta h||_2."""
    return eps_z_lipschitz(rho, n, sigma, gamma)


def separating_guards(rho: float, n: int):
    """Guards that actually separate two ADJACENT terminal leaves.

    Only these matter for a direct leaf-to-leaf transition; a guard nested
    inside a region with no leaf on one side cannot be the binding constraint.
    """
    L = leaves(rho, n)
    out = []
    for i in range(len(L) - 1):
        a, b = L[i][1], L[i + 1][0]          # gap between consecutive leaves
        if b > a:
            out.append((a, b))
    return out


def eps_z_exact(rho: float, n: int, tau: float, sigma: float,
                gamma: float) -> float:
    """Exact projection displacement needed to cross the easiest guard.

    Note tau cancels: only the WIDTH in z-space matters, so the exact
    certificate is independent of where the behavioural centre sits. That is a
    property worth stating, not an accident.
    """
    best = np.inf
    for a, b in separating_guards(rho, n):
        dz = (sigma / gamma) * abs(float(logit(b)) - float(logit(a)))
        best = min(best, dz)
    return float(best)
