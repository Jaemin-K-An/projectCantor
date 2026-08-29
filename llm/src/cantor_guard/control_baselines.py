"""The controller line-up (harness §30). Every entry uses the SAME v_ref, the
SAME harm detector and the SAME layer set; only the control law differs.

  L0 none                 no intervention
  L1 constant             constant refusal steering  (Turner 2023 / Rimsky 2024)
  L2 central              one barrier on the decision boundary (single scale)
  L3 periodic             matched gaps, coarse->fine lattice
  L4 random               matched widths, random packing
  L5 shuffled             matched widths+levels, shuffled order
  L6 center_anchored      as L5 but the level-1 barrier stays on the boundary
  L7 cantor               CantorGuard
  L8 learned_spline       monotone spline potential, same total action, fitted
                          on DEV with the same search budget as everything else

L1 is the literature baseline; L6 is the control that H4 must beat.
"""
from __future__ import annotations
import numpy as np
from .cantor_barrier import BarrierLayout, Gap, build_layout
from .threat_coordinate import CantorGuardController

CONTROLLER_FAMILIES = ["L0_none", "L1_constant", "L2_central", "L3_periodic",
                       "L4_random", "L5_shuffled", "L6_center_anchored",
                       "L7_cantor", "L8_learned_spline"]
RANDOMISED = {"L4_random", "L5_shuffled", "L6_center_anchored"}


class _SplineLayout(BarrierLayout):
    """L8: a monotone piecewise potential built from `m` equal-width segments
    whose energies are free parameters, renormalised to the same total action.

    Implemented as `m` adjacent gaps covering [0,1], so it reuses the exact
    same evaluation path (and therefore the same action bookkeeping) as every
    other layout -- the only difference is where the energy sits.
    """
    pass


def learned_spline_layout(weights: np.ndarray, n: int, E0: float,
                          label: str = "spline") -> BarrierLayout:
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 1e-6, None)
    w = w / w.sum()                                    # energy shares
    m = len(w)
    edges = np.linspace(0.0, 1.0, m + 1)
    gaps = [Gap(1, float(edges[i]), float(edges[i + 1])) for i in range(m)]
    L = BarrierLayout(gaps, n, 1.0, label, "L8_learned_spline")
    # override per-gap energies to the learned shares, keeping total = n*E0
    L.est = (w * (n * E0)).astype(float)
    L.coef = L.est / L.wid
    L.cum = np.concatenate([[0.0], np.cumsum(L.est)])
    return L


def make_controller(family: str, *, n: int, B_total: float, eta: float,
                    gamma: float, seed: int = 0, harm_gate: bool = True,
                    max_norm: float | None = None,
                    spline_weights: np.ndarray | None = None
                    ) -> CantorGuardController:
    E0 = B_total / n
    if family == "L8_learned_spline":
        if spline_weights is None:
            spline_weights = np.ones(8)
        layout = learned_spline_layout(spline_weights, n, E0)
    else:
        layout = build_layout(family, n, E0, seed=seed)
    return CantorGuardController(layout, eta=eta, gamma=gamma,
                                 harm_gate=harm_gate, max_norm=max_norm)
