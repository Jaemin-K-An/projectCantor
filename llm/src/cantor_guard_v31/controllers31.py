"""V3.1 LLM controllers, including a genuinely state-independent constant.

V3 defect D1: V3's "constant" was a single smoothstep barrier over [0,1]. Here
`TrueConstantController` returns the same magnitude for every margin, so
"state-independent" means what it says. `sup_deriv` records Theorem T's
||u'||_inf for each controller, which is the quantity that predicts fragility
under calibration error.
"""
from __future__ import annotations
import numpy as np
from cantor_guard.cantor_barrier import build_layout, Gap, BarrierLayout
from cantor_guard.threat_coordinate import threat_from_margin

V31_LLM_FAMILIES = ["T0_none", "T1_true_constant", "T2_global_smooth",
                    "T3_wide_central", "T4_periodic", "T5_shuffled",
                    "T6_center_anchored", "T7_cantor", "T8_minimax"]
V31_RANDOMISED = {"T5_shuffled", "T6_center_anchored"}


def _bin_layout(weights, n, E0, label, family):
    w = np.clip(np.asarray(weights, float), 1e-9, None); w = w / w.sum()
    m = len(w); edges = np.linspace(0, 1, m + 1)
    gaps = [Gap(1, float(edges[i]), float(edges[i + 1])) for i in range(m)]
    L = BarrierLayout(gaps, n, 1.0, label, family)
    L.est = w * (n * E0); L.coef = L.est / L.wid
    L.cum = np.concatenate([[0.0], np.cumsum(L.est)])
    return L


class Controller31:
    """Maps a margin array to a non-negative intervention magnitude."""

    def __init__(self, family, *, n=5, B_total=1.0, gamma=1.0, eta=1.0,
                 seed=0, weights=None, harm_gate=True, max_q=None):
        self.family, self.n, self.gamma, self.eta = family, n, gamma, eta
        self.harm_gate, self.max_q, self.B_total = harm_gate, max_q, B_total
        E0 = B_total / n
        self.layout = None
        if family == "T0_none":
            pass
        elif family == "T1_true_constant":
            self.const = n * E0                      # analytic action = B_total
        elif family == "T2_global_smooth":
            self.layout = _bin_layout(np.ones(1), n, E0, "global_smooth", family)
        elif family == "T3_wide_central":
            self.layout = BarrierLayout([Gap(1, 1/6, 5/6)], n, n * E0, "wide_central", family)
        elif family == "T8_minimax":
            self.layout = _bin_layout(np.ones(8) if weights is None else weights,
                                      n, E0, "minimax", family)
        else:
            key = {"T4_periodic": "L3_periodic", "T5_shuffled": "L5_shuffled",
                   "T6_center_anchored": "L6_center_anchored",
                   "T7_cantor": "L7_cantor"}[family]
            self.layout = build_layout(key, n, E0, seed=seed)

    def magnitude(self, m: np.ndarray) -> np.ndarray:
        """|u| >= 0 as a function of the safety margin."""
        m = np.asarray(m, float)
        if self.family == "T0_none":
            return np.zeros_like(m)
        if self.family == "T1_true_constant":
            c = np.full(m.shape, self.eta * self.const, dtype=float)
        else:
            c = self.eta * self.layout.field(threat_from_margin(m, self.gamma))
        return c

    @property
    def sup_deriv(self) -> float:
        """Theorem T's ||u'||_inf. Exactly 0 for the true constant."""
        if self.family in ("T0_none", "T1_true_constant"):
            return 0.0
        return float(6 * np.max(self.layout.est / self.layout.wid ** 2))

    @property
    def analytic_action(self) -> float:
        if self.family == "T0_none":
            return 0.0
        if self.family == "T1_true_constant":
            return self.const
        return float(self.layout.est.sum())

    def describe(self) -> dict:
        return {"family": self.family, "n": self.n, "gamma": self.gamma,
                "eta": self.eta, "B_total": self.B_total,
                "analytic_action": self.analytic_action,
                "sup_deriv": self.sup_deriv,
                "max_q": -1.0 if self.max_q is None else self.max_q}
