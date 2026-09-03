"""Hard-capped Cantor controller and a non-Cantor sensor-actuator baseline.

Two repairs over V3.4.0:

1. ``q_cap`` is now ENFORCED statewise, not merely declared.  V3.4.0 recorded
   q_max up to 0.0554 against a stated cap of 0.05 because the cap was never
   applied; here ``q_ctrl = min(eta * a(cell), q_cap)`` holds for every state.

2. ``LinearThresholdController`` uses the same sensor, actuator and budget but
   no recursive partition.  Without it, "the controller helps" and "the Cantor
   partition helps" cannot be told apart.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cantor_guard_v340.actuator import Actuator
from cantor_guard_v340.sensor_actuator_controller import (
    DEFAULT_LEAF_ACTIONS,
    OUTSIDE_WINDOW,
    SensorActuatorCantorController,
)
from cantor_guard_v340.sensor_distance import SensorHyperplane


@dataclass(frozen=True)
class CappedResult:
    h_corrected: np.ndarray
    delta_h: np.ndarray
    d_observed: np.ndarray
    r: np.ndarray
    actions: np.ndarray
    q_raw: np.ndarray
    q_ctrl: np.ndarray
    clipped: np.ndarray
    cell_kind: tuple
    cell_index: tuple
    delta_d_expected: np.ndarray


class _CapMixin:
    q_cap: float

    def _apply(self, residual, actions, kinds, indices, d, r):
        norms = np.linalg.norm(residual, axis=1)
        q_raw = self.eta * actions
        q = np.minimum(q_raw, self.q_cap)
        if np.any(q > self.q_cap + 1e-12):
            raise AssertionError("hard q cap violated")
        delta = q[:, None] * norms[:, None] * self.actuator.v_safe[None, :]
        return CappedResult(
            h_corrected=residual + delta, delta_h=delta, d_observed=d, r=r,
            actions=actions, q_raw=q_raw, q_ctrl=q, clipped=(q_raw > self.q_cap),
            cell_kind=tuple(kinds), cell_index=tuple(indices),
            delta_d_expected=q * norms * self.kappa,
        )

    def policy_record(self, h) -> list[dict]:
        """Name the frozen V3.4.0 generation hook calls. Alias of ``records``."""
        return self.records(h)

    def records(self, h) -> list[dict]:
        res = self.correct(h)
        return [
            {"d_observed": float(res.d_observed[i]),
             "r": None if not np.isfinite(res.r[i]) else float(res.r[i]),
             "cell_kind": res.cell_kind[i], "cell_index": res.cell_index[i],
             "action": float(res.actions[i]), "q_raw": float(res.q_raw[i]),
             "q_ctrl": float(res.q_ctrl[i]), "clipped": bool(res.clipped[i]),
             "delta_d_expected": float(res.delta_d_expected[i]),
             "outside_window": res.cell_kind[i] == "outside",
             "status": OUTSIDE_WINDOW if res.cell_kind[i] == "outside" else "INSIDE_WINDOW"}
            for i in range(len(res.d_observed))
        ]


class CappedCantorController(_CapMixin):
    """The V3.4.0 depth-3 Cantor policy with a real statewise budget cap."""

    name_prefix = "CANTOR"

    def __init__(self, *, sensor: SensorHyperplane, actuator: Actuator, W: float,
                 rho: float, eta: float, q_cap: float,
                 leaf_actions=DEFAULT_LEAF_ACTIONS, outside_action: float = 1.0):
        if not np.isfinite(q_cap) or q_cap <= 0:
            raise ValueError("q_cap must be finite and positive")
        self.inner = SensorActuatorCantorController(
            sensor=sensor, actuator=actuator, W=W, rho=rho, eta=1.0,
            leaf_actions=leaf_actions, outside_action=outside_action)
        self.sensor, self.actuator = sensor, actuator
        self.W, self.rho, self.eta, self.q_cap = float(W), float(rho), float(eta), float(q_cap)
        self.kappa = self.inner.kappa
        self.leaves, self.guards = self.inner.leaves, self.inner.guards

    def correct(self, h) -> CappedResult:
        residual = np.atleast_2d(np.asarray(h, dtype=float))
        d, r, cells = self.inner.classify_state(residual)
        actions = np.asarray([self.inner.action_for_cell(c) for c in cells], dtype=float)
        return self._apply(residual, actions, [c.kind for c in cells],
                           [c.index for c in cells], d, r)


class LinearThresholdController(_CapMixin):
    """Same sensor, actuator and budget; NO recursive partition.

    ``a = clip(r, 0, 1)`` -- the action rises smoothly with risk instead of
    stepping through eight Cantor leaves.  Outside the window it falls back to
    the same conservative action, so the only difference from the Cantor arm is
    the partition itself.
    """

    name_prefix = "LINEAR"

    def __init__(self, *, sensor: SensorHyperplane, actuator: Actuator, W: float,
                 eta: float, q_cap: float, outside_action: float = 1.0):
        if not np.isfinite(q_cap) or q_cap <= 0:
            raise ValueError("q_cap must be finite and positive")
        self.sensor, self.actuator = sensor, actuator
        self.W, self.eta, self.q_cap = float(W), float(eta), float(q_cap)
        self.outside_action = float(outside_action)
        self.kappa = float(sensor.coupling(actuator.v_safe))
        self.rho = None

    def correct(self, h) -> CappedResult:
        residual = np.atleast_2d(np.asarray(h, dtype=float))
        d = np.atleast_1d(self.sensor.distance(residual))
        inside = np.isfinite(d) & (np.abs(d) <= self.W)
        r = np.full(d.shape, np.nan, dtype=float)
        r[inside] = 0.5 - d[inside] / (2 * self.W)
        actions = np.where(inside, np.clip(r, 0.0, 1.0), self.outside_action)
        kinds = np.where(inside, "linear", "outside").tolist()
        return self._apply(residual, actions, kinds, [None] * len(d), d, r)
