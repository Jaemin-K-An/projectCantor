"""V3.3.4 PHASE 8 -- CantorGuardedPolicy: Cantor as an actual inference-time
safety controller, centred on the BEHAVIOURAL boundary.

V3.3.3 DEFECT this fixes. Its System A controller used tau_mid (+0.9887) while
the measured behavioural boundary was tau_beh (-2.6263) -- a 3.90 sigma
correction that was measured and then never applied. Here tau_beh is REQUIRED:
constructing the controller without it raises, so there is no silent fallback.

CONTROLLER H (hard, certified). Discrete leaf/guard policy matching the
theorem's semantics:
    r in terminal leaf j  ->  leaf action a_j (monotone in threat)
    r in any guard        ->  conservative fallback
This is the object the certificate is about. CONTROLLER S (the smooth
RhoBarrier of V2-V3.3) is a separate lineage and must never be used to evidence
a claim about H.
"""
from __future__ import annotations
import numpy as np
from .certified_geometry import classify_exact, leaves, M_n, RHO_CANTOR
from .certificate import eps_z_lipschitz, eps_h_l2, eps_z_exact

__all__ = ["CantorGuardedPolicy"]


class CantorGuardedPolicy:
    """Depth-n binary recursive guard policy on a behaviourally centred
    safety coordinate.

    `higher_r_is_threat` fixes the orientation. It must be determined on DEV
    from a causal audit and then frozen -- never inferred from test results.
    """

    def __init__(self, rho: float, depth: int, *, tau_beh: float, sigma: float,
                 gamma: float = 0.7, a_min: float = 0.0, a_max: float = 1.0,
                 higher_r_is_threat: bool = True, max_q: float | None = 0.05,
                 eta: float = 1.0, require_behavioural: bool = True):
        if require_behavioural and tau_beh is None:
            raise ValueError("tau_beh is required; there is no fallback to tau_mid")
        if not (0.0 < rho < 0.5):
            raise ValueError("rho in (0, 1/2)")
        self.rho, self.depth = float(rho), int(depth)
        self.tau_beh, self.sigma, self.gamma = float(tau_beh), float(sigma), float(gamma)
        self.a_min, self.a_max = float(a_min), float(a_max)
        self.higher_r_is_threat = bool(higher_r_is_threat)
        self.max_q, self.eta = max_q, float(eta)
        self.harm_gate = True
        self.family = f"cantorguard_rho{self.rho:.6f}_n{self.depth}"
        self._leaves = leaves(self.rho, self.depth)
        self.n_leaves = len(self._leaves)
        # monotone leaf schedule in coordinate order; orientation flips it
        j = np.arange(self.n_leaves, dtype=float)
        frac = j / max(self.n_leaves - 1, 1)
        if not self.higher_r_is_threat:
            frac = 1.0 - frac
        self.leaf_actions = self.a_min + frac * (self.a_max - self.a_min)
        self._addr_to_rank = {addr: i for i, (_, _, addr) in
                              enumerate(sorted(self._leaves, key=lambda t: t[0]))}

    # ---- coordinate -----------------------------------------------------
    def coordinate(self, z):
        z = np.asarray(z, float)
        return 1.0 / (1.0 + np.exp(np.clip(self.gamma * (z - self.tau_beh)
                                           / self.sigma, -60, 60)))

    def classify_r(self, r):
        """('leaf', rank) or ('guard', level), per coordinate."""
        rr = np.atleast_1d(np.asarray(r, float))
        kind = np.empty(len(rr), dtype=object)
        idx = np.zeros(len(rr), dtype=int)
        for i, x in enumerate(rr):
            c = classify_exact(float(x), self.rho, self.depth)
            if c[0] == "leaf":
                kind[i], idx[i] = "leaf", self._addr_to_rank[c[1]]
            else:
                kind[i], idx[i] = "guard", c[1]
        return kind, idx

    def action(self, r):
        """Leaf action, or the conservative guard fallback.

        The guard action is at least as conservative as EITHER neighbour, which
        is what makes the guard a genuine buffer: an attack that lands in the
        guard cannot obtain a weaker correction than it would in either
        adjacent leaf.
        """
        rr = np.atleast_1d(np.asarray(r, float))
        kind, idx = self.classify_r(rr)
        out = np.zeros(len(rr))
        for i in range(len(rr)):
            if kind[i] == "leaf":
                out[i] = self.leaf_actions[idx[i]]
            else:
                lo = np.searchsorted([l[1] for l in self._leaves], rr[i]) - 1
                left = self.leaf_actions[max(lo, 0)]
                right = self.leaf_actions[min(lo + 1, self.n_leaves - 1)]
                out[i] = max(left, right)     # conservative = stronger correction
        return out

    def magnitude(self, m):
        """Controller interface: margin -> non-negative correction magnitude.

        The hook supplies m = (z - tau)/sigma already centred on tau_beh, so
        z is recovered before the coordinate map is applied.
        """
        m = np.asarray(m, float)
        z = m * self.sigma + self.tau_beh
        return self.eta * self.action(self.coordinate(z).ravel()).reshape(m.shape)

    # ---- certificates ---------------------------------------------------
    def certificate_r(self) -> float:
        return float(M_n(self.rho, self.depth))

    def certificate_z_lipschitz(self) -> float:
        return eps_z_lipschitz(self.rho, self.depth, self.sigma, self.gamma)

    def certificate_h_l2(self) -> float:
        return eps_h_l2(self.rho, self.depth, self.sigma, self.gamma)

    def certificate_z_exact(self) -> float:
        return eps_z_exact(self.rho, self.depth, self.tau_beh, self.sigma,
                           self.gamma)
