"""V3.3.1 PHASE 7 -- Theorem R, refinement consistency.

THEOREM R. Under the fixed level-energy convention (each level carries E0
regardless of depth), the order-(n+1) controller agrees EXACTLY with the
order-n controller on everything the order-n controller already resolved:

    u_n = sum_{k<=n} u_k,    supp(u_k) = G_k,    G_j disjoint for j != k
    => supp(u_{n+1} - u_n) = G_{n+1} subset K_n
    => u_{n+1}(r) = u_n(r)  for all r in S_n = union_{k<=n} G_k

Refining the controller adds intervention only where it previously had none.
There is no regression on already-resolved regions -- "regression-free
structural refinement".

COROLLARY R.1 (measures, general rho):
    mu(K_n)     = (2 rho)^n
    mu(S_n)     = 1 - (2 rho)^n
    mu(G_{n+1}) = g * (2 rho)^n          Cantor: (1/3)(2/3)^n

CAVEAT, and it is the reason two versions are measured separately below.
Theorem R holds for the STRUCTURAL convention (same E0, no gain
renormalisation). Under the DEPLOYMENT convention the gain eta is refitted so
that the realised budget matches a target, and eta_n != eta_{n+1}, so the old
support is rescaled and exact agreement is NOT expected. Reporting one and
calling it the other would be wrong.
"""
from __future__ import annotations
import numpy as np
from .rho_family import RhoBarrier

__all__ = ["structural_drift", "deployment_drift", "backward_compatibility",
           "measures"]


def measures(rho: float, n: int) -> dict:
    g = 1.0 - 2.0 * rho
    return {"mu_K_n": (2 * rho) ** n,
            "mu_S_n": 1.0 - (2 * rho) ** n,
            "mu_G_next": g * (2 * rho) ** n}


def structural_drift(rho: float, n: int, r=None, E0: float = 1.0) -> dict:
    """D_old under the STRUCTURAL convention. Theorem R predicts exactly 0."""
    a, b = RhoBarrier(rho, n, E0), RhoBarrier(rho, n + 1, E0)
    if r is None:
        r = np.linspace(0.0, 1.0, 200_001)
    fa, fb = a.field(r), b.field(r)
    old = a.in_guard(r)                       # r in S_n
    if not old.any():
        return {"n_old_points": 0, "max_abs_drift": 0.0, "max_rel_drift": 0.0}
    d = np.abs(fb[old] - fa[old])
    scale = np.maximum(np.abs(fa[old]), 1e-300)
    return {"n_old_points": int(old.sum()),
            "max_abs_drift": float(d.max()),
            "max_rel_drift": float((d / scale).max()),
            "new_support_fraction": float((b.in_guard(r) & ~old).mean())}


def deployment_drift(rho: float, n: int, eta_n: float, eta_next: float,
                     r=None, E0: float = 1.0) -> dict:
    """D_old under the DEPLOYMENT convention, where the gain is refitted.

    Exact agreement is NOT expected here; the quantity is reported so that the
    structural theorem is not mistaken for a deployment guarantee.
    """
    a, b = RhoBarrier(rho, n, E0), RhoBarrier(rho, n + 1, E0)
    if r is None:
        r = np.linspace(0.0, 1.0, 200_001)
    fa, fb = eta_n * a.field(r), eta_next * b.field(r)
    old = a.in_guard(r)
    if not old.any():
        return {"n_old_points": 0, "max_rel_drift": 0.0, "mean_rel_drift": 0.0}
    d = np.abs(fb[old] - fa[old])
    scale = np.maximum(np.abs(fa[old]), 1e-300)
    return {"n_old_points": int(old.sum()),
            "max_rel_drift": float((d / scale).max()),
            "mean_rel_drift": float((d / scale).mean())}


def backward_compatibility(rho: float, n: int, eta_n: float = 1.0,
                           eta_next: float = 1.0, eps: float = 1e-12) -> float:
    """BC(n, n+1) = 1 - E[ |u_{n+1} - u_n| / max(|u_n|, eps) ] on the old support.

    1.0 means the refinement changed nothing the previous order had resolved.
    Bounded below at 0 so a blown-up ratio cannot produce a large negative.
    """
    r = np.linspace(0.0, 1.0, 200_001)
    a, b = RhoBarrier(rho, n), RhoBarrier(rho, n + 1)
    fa, fb = eta_n * a.field(r), eta_next * b.field(r)
    old = a.in_guard(r)
    if not old.any():
        return 1.0
    ratio = np.abs(fb[old] - fa[old]) / np.maximum(np.abs(fa[old]), eps)
    return float(max(0.0, 1.0 - ratio.mean()))
