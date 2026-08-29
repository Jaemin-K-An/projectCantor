"""Margin -> threat coordinate, and the CantorGuard control law.

    m       = (<h, v_ref> - tau)/sigma      signed distance to the refusal
                                            decision boundary (larger = safer)
    r       = sigmoid(-gamma * m)           threat coordinate in (0,1)
    c(r)    = eta * V'_L(r)  >= 0           corrective magnitude
    h'      = h + c(r) * v_ref              intervention (v_ref is unit norm,
                                            so ||h' - h|| == c(r))

r -> 0 deep safe, r = 1/2 exactly on the boundary (m = 0), r -> 1 unsafe, so
the barrier's level-1 gap -- which straddles r = 1/2 in the Cantor and
center-anchored layouts -- sits exactly on the refusal decision boundary.

`gamma` is chosen on CALIBRATION/DEV only and frozen before the test.
"""
from __future__ import annotations
import numpy as np


def threat_from_margin(m: np.ndarray, gamma: float) -> np.ndarray:
    """r = sigmoid(-gamma*m). Monotone DECREASING in m (tested)."""
    return 1.0 / (1.0 + np.exp(np.clip(gamma * np.asarray(m, dtype=float), -60, 60)))


def margin_from_threat(r: np.ndarray, gamma: float) -> np.ndarray:
    r = np.clip(np.asarray(r, dtype=float), 1e-12, 1 - 1e-12)
    return -np.log(r / (1 - r)) / gamma


class CantorGuardController:
    """A state-dependent residual controller.

    Parameters
    ----------
    layout : BarrierLayout        the barrier (any of L0..L7)
    eta    : float                overall gain
    gamma  : float                margin -> threat sharpness
    harm_gate : bool              if True the controller is inert unless the
                                  prompt is flagged harmful. Without this, a
                                  controller that always pushes toward refusal
                                  is a trivial over-refusal machine (harness
                                  §20); the SAME detector is shared by every
                                  baseline so the comparison stays fair.
    max_norm : float | None       hard cap on ||Delta h||, applied to every
                                  controller identically.
    """

    def __init__(self, layout, *, eta: float = 1.0, gamma: float = 1.0,
                 harm_gate: bool = True, max_norm: float | None = None):
        self.layout = layout
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.harm_gate = harm_gate
        self.max_norm = max_norm

    def magnitude(self, m: np.ndarray) -> np.ndarray:
        """c(r(m)) >= 0, the norm of the residual update."""
        r = threat_from_margin(m, self.gamma)
        c = self.eta * self.layout.field(r)
        if self.max_norm is not None:
            c = np.minimum(c, self.max_norm)
        return c

    def describe(self) -> dict:
        return {"family": self.layout.family, "label": self.layout.label,
                "n": self.layout.n, "E0": self.layout.E0, "eta": self.eta,
                "gamma": self.gamma, "harm_gate": self.harm_gate,
                "max_norm": self.max_norm if self.max_norm is not None else -1.0,
                "total_action": self.layout.total_action()}
