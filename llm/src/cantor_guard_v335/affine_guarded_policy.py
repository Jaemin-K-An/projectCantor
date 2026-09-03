"""V3.3.5 PHASE 9 -- AffineCantorGuardedPolicy.

API FIX for V3.3.4 defect (E). That controller exposed magnitude(m) and rebuilt
z = m*sigma + tau_beh. The hook computed m with the PREFILL calibration, so the
reconstructed z was not the actual projection during prefill. Here the
controller takes the residual h (or the actual z) and computes z = <h, v>
ITSELF. There is no margin argument and no cross-phase reconstruction.
"""
from __future__ import annotations
import numpy as np
from .affine_coordinate import AffineCoordinate, OUTSIDE
from .certificate import eps_z_affine, eps_h_affine
from cantor_guard_v334.certified_geometry import classify_exact, leaves, M_n

__all__ = ["AffineCantorGuardedPolicy"]


class AffineCantorGuardedPolicy:
    """Depth-n Cantor guard policy on the affine behavioural coordinate.

    Three state classes, and OUTSIDE is a real class, never a clip:
        leaf j   -> monotone threat-ordered action a_j
        guard    -> conservative fallback, >= both adjacent leaf actions
        OUTSIDE  -> conservative fallback (window does not apply)
    """

    def __init__(self, rho: float, depth: int, *, tau_g1: float, W: float,
                 orientation: int = +1, a_min: float = 0.0, a_max: float = 1.0,
                 eta: float = 1.0, max_q: float | None = 0.05):
        if tau_g1 is None:
            raise ValueError("tau_g1 required; no fallback to tau_mid or global tau_beh")
        if not (0.0 < rho < 0.5):
            raise ValueError("rho in (0, 1/2)")
        self.rho, self.depth = float(rho), int(depth)
        self.coord = AffineCoordinate(tau_g1, W, orientation)
        self.tau_g1, self.W, self.s = float(tau_g1), float(W), int(orientation)
        self.a_min, self.a_max, self.eta, self.max_q = a_min, a_max, float(eta), max_q
        self.family = f"affine_cantor_rho{self.rho:.6f}_n{self.depth}"
        self._leaves = leaves(self.rho, self.depth)
        self.n_leaves = len(self._leaves)
        frac = np.arange(self.n_leaves, dtype=float) / max(self.n_leaves - 1, 1)
        self.leaf_actions = self.a_min + frac * (self.a_max - self.a_min)
        self.outside_action = float(self.a_max)   # most conservative
        self._edges = np.array([l[1] for l in self._leaves])

    # ---- classification -------------------------------------------------
    def z_of_h(self, h, v):
        return np.asarray(h, float) @ np.asarray(v, float)

    def classify(self, z):
        """('leaf', rank) | ('guard', level) | (OUTSIDE, -1) per element."""
        zz = np.atleast_1d(np.asarray(z, float))
        r = self.coord.r(zz)
        kind = np.empty(len(zz), dtype=object)
        idx = np.zeros(len(zz), dtype=int)
        for i, (rv, zv) in enumerate(zip(r, zz)):
            if not np.isfinite(rv):
                kind[i], idx[i] = OUTSIDE, -1
                continue
            c = classify_exact(float(rv), self.rho, self.depth)
            if c[0] == "leaf":
                kind[i] = "leaf"
                idx[i] = int(np.searchsorted(self._edges, rv))
            else:
                kind[i], idx[i] = "guard", c[1]
        return kind, idx

    def action(self, z):
        zz = np.atleast_1d(np.asarray(z, float))
        kind, idx = self.classify(zz)
        r = self.coord.r(zz)
        out = np.zeros(len(zz))
        for i in range(len(zz)):
            if kind[i] == OUTSIDE:
                out[i] = self.outside_action
            elif kind[i] == "leaf":
                out[i] = self.leaf_actions[min(idx[i], self.n_leaves - 1)]
            else:
                lo = int(np.searchsorted(self._edges, r[i])) - 1
                left = self.leaf_actions[max(lo, 0)]
                right = self.leaf_actions[min(lo + 1, self.n_leaves - 1)]
                out[i] = max(left, right)
        return out

    def intervene(self, h, v):
        """Return Delta h to add at this residual state. Takes h DIRECTLY."""
        h = np.asarray(h, float)
        v = np.asarray(v, float)
        z = h @ v
        a = self.action(np.atleast_1d(z))
        mag = self.eta * a
        if self.max_q is not None:
            nrm = np.linalg.norm(h, axis=-1)
            mag = np.minimum(mag, self.max_q * np.atleast_1d(nrm))
        return mag[..., None] * v, mag

    # ---- certificates ---------------------------------------------------
    def certificate_r(self) -> float:
        return float(M_n(self.rho, self.depth))

    def certificate_z_exact(self) -> float:
        return float(eps_z_affine(self.rho, self.depth, self.W))

    def certificate_h_l2(self) -> float:
        return float(eps_h_affine(self.rho, self.depth, self.W))
