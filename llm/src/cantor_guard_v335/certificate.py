"""V3.3.5 PHASE 2 -- the EXACT affine residual certificate.

V3.3.4 had two certificates and they disagreed:
  Lipschitz  (4 sigma/gamma) M_n(rho)  -- a constant multiple of M_n, so its
             argmax is 1/3, but it is only an upper bound on |dr/dz|
  exact      min over guards of (sigma/gamma)|logit b - logit a| -- argmax 0.296,
             because the logistic slope varies with position
Under the affine coordinate the two coincide, because |dr/dz| = 1/(2W) is exact
rather than bounded.

THEOREM AZ. Inside the operating window, |Delta r| = |Delta z| / (2W) exactly.
A direct terminal-to-terminal transition requires |Delta r| >= M_n(rho), so it
requires

    |Delta z| >= 2W M_n(rho)  =:  eps_z^A(rho, n)                      (exact)

COROLLARY AZ3. At n = 3, eps_z^A = 2W rho^2(1-2rho). 2W does not depend on rho,
so argmax eps_z^A(., 3) = argmax M_3 = 1/3, and eps_C = 2W/27.

THEOREM AH. ||v_ref|| = 1 gives |Delta z| = |<Delta h, v_ref>| <= ||Delta h||_2,
so ||Delta h||_2 < eps_z^A implies no direct transition. eps_h^A = eps_z^A is a
SUFFICIENT L2 residual direct-policy-transition certificate.

SCOPE. Guarantees: this layer, this refusal direction, the G1 residual state,
inside the affine window, for the defined Cantor partition, against DIRECT
terminal-to-terminal switching. Guarantees nothing about arbitrary semantic
jailbreaks, other layers, or harmless text.
"""
from __future__ import annotations
import numpy as np
from cantor_guard_v334.certified_geometry import M_n, rho_max, M_n_max

__all__ = ["eps_z_affine", "eps_h_affine", "cantor_gain_table", "logistic_exact"]


def eps_z_affine(rho, n: int, W: float):
    """THEOREM AZ. EXACT z-space radius: 2W * M_n(rho). Independent of eta."""
    return 2.0 * float(W) * M_n(rho, n)


def eps_h_affine(rho, n: int, W: float):
    """THEOREM AH. Same radius, sufficient for arbitrary ||Delta h||_2."""
    return eps_z_affine(rho, n, W)


def cantor_gain_table(W: float, n: int = 3,
                      rhos=(0.25, 0.28, 0.30, 1/3, 0.36, 0.40, 0.44)):
    """Analytic table. Mathematical, not empirical."""
    eC = eps_z_affine(1/3, n, W)
    return [{"rho": float(r), "M_n": float(M_n(r, n)),
             "eps_z_affine": float(eps_z_affine(r, n, W)),
             "cantor_gain_pct": float(100.0 * (eC / eps_z_affine(r, n, W) - 1.0)),
             "is_cantor": abs(r - 1/3) < 1e-12} for r in rhos]


def logistic_exact(rho: float, n: int, sigma: float, gamma: float) -> float:
    """HISTORICAL CONTROL: V3.3.4's exact logistic certificate, kept so the
    coordinate-distortion effect stays visible and its optimum (~0.296) is not
    quietly dropped."""
    from cantor_guard_v334.certificate import eps_z_exact
    return eps_z_exact(rho, n, 0.0, sigma, gamma)
