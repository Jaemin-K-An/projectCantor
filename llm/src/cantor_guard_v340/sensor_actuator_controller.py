"""The V3.4.0 controller: SENSE with w, DECIDE with Cantor, ACTUATE with v.

This is the architecture the whole V3.3.5x lineage was missing.  Every earlier
controller read its state off the same direction it pushed on; here the two
roles are separate objects, and their only contact is the scalar coupling
``kappa = <w_hat, v_safe>``, which makes the induced sensor movement exact:

    delta_d = eta * a(cell) * ||h|| * kappa.

The controller consumes the ACTUAL residual vector -- never a precomputed
margin -- so no cross-calibration step can silently disagree with the model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .actuator import Actuator
from .cantor_geometry import Cell, classify, partition
from .sensor_distance import SensorHyperplane

OUTSIDE_WINDOW = "OUTSIDE_WINDOW"
DEFAULT_LEAF_ACTIONS = tuple(i / 7 for i in range(8))


@dataclass(frozen=True)
class ControlResult:
    h_corrected: np.ndarray
    delta_h: np.ndarray
    d_observed: np.ndarray
    r: np.ndarray
    cells: tuple[Cell, ...]
    actions: np.ndarray
    q_ctrl: np.ndarray
    delta_d_expected: np.ndarray


class SensorActuatorCantorController:
    def __init__(
        self,
        *,
        sensor: SensorHyperplane,
        actuator: Actuator,
        W: float,
        rho: float,
        eta: float,
        depth: int = 3,
        leaf_actions=DEFAULT_LEAF_ACTIONS,
        outside_action: float = 1.0,
    ):
        if depth != 3:
            raise ValueError("V3.4.0 freezes depth=3")
        if not np.isfinite(W) or W <= 0:
            raise ValueError("W must be finite and positive")
        if not np.isfinite(eta) or eta < 0:
            raise ValueError("eta must be finite and non-negative")
        actions = np.asarray(leaf_actions, dtype=float)
        if actions.shape != (8,) or np.any(np.diff(actions) < 0) or np.any((actions < 0) | (actions > 1)):
            raise ValueError("leaf_actions must be eight nondecreasing values in [0,1]")
        if not 0 <= outside_action <= 1:
            raise ValueError("outside_action must lie in [0,1]")
        self.sensor = sensor
        self.actuator = actuator
        self.W = float(W)
        self.rho = float(rho)
        self.eta = float(eta)
        self.depth = depth
        self.leaf_actions = actions
        self.outside_action = float(outside_action)
        self.leaves, self.guards = partition(self.rho, self.depth)
        self.kappa = float(sensor.coupling(actuator.v_safe))

    def risk_coordinate(self, d):
        """r = 1/2 - d/(2W); NaN outside the window, never clipped."""
        arr = np.atleast_1d(np.asarray(d, dtype=float))
        out = np.full(arr.shape, np.nan, dtype=float)
        inside = np.isfinite(arr) & (np.abs(arr) <= self.W)
        out[inside] = 0.5 - arr[inside] / (2 * self.W)
        return out

    def _guard_action(self, guard: Cell) -> float:
        left = [leaf for leaf in self.leaves if leaf.hi <= guard.lo + 1e-15]
        right = [leaf for leaf in self.leaves if leaf.lo >= guard.hi - 1e-15]
        adjacent = []
        if left:
            adjacent.append(self.leaf_actions[max(left, key=lambda c: c.hi).index])
        if right:
            adjacent.append(self.leaf_actions[min(right, key=lambda c: c.lo).index])
        return float(max(adjacent)) if adjacent else self.outside_action

    def action_for_cell(self, cell: Cell) -> float:
        if cell.kind == "leaf":
            return float(self.leaf_actions[cell.index])
        if cell.kind == "guard":
            return self._guard_action(cell)
        return self.outside_action

    def classify_state(self, h):
        d = np.atleast_1d(self.sensor.distance(h))
        r = self.risk_coordinate(d)
        cells = tuple(
            classify(float(ri), self.rho, self.depth) if np.isfinite(ri)
            else Cell("outside", float("nan"), float("nan"))
            for ri in r
        )
        return d, r, cells

    def correct(self, h) -> ControlResult:
        residual = np.asarray(h, dtype=float)
        squeeze = residual.ndim == 1
        if squeeze:
            residual = residual[None, :]
        if residual.ndim != 2 or residual.shape[1] != self.sensor.w.size:
            raise ValueError("h must be [d] or [batch,d] matching the sensor")
        norms = np.linalg.norm(residual, axis=1)
        if np.any(norms <= 0) or not np.isfinite(norms).all():
            raise ValueError("actual P0 residuals must have finite positive norm")
        d, r, cells = self.classify_state(residual)
        actions = np.asarray([self.action_for_cell(c) for c in cells], dtype=float)
        q = self.eta * actions
        delta = q[:, None] * norms[:, None] * self.actuator.v_safe[None, :]
        corrected = residual + delta
        expected = q * norms * self.kappa
        if squeeze:
            corrected, delta, d, r, actions, q, expected = (
                x[0] for x in (corrected, delta, d, r, actions, q, expected)
            )
        return ControlResult(corrected, delta, d, r, cells, actions, q, expected)

    def policy_record(self, h) -> list[dict]:
        result = self.correct(h)
        d = np.atleast_1d(result.d_observed)
        r = np.atleast_1d(result.r)
        a = np.atleast_1d(result.actions)
        q = np.atleast_1d(result.q_ctrl)
        dd = np.atleast_1d(result.delta_d_expected)
        return [
            {
                "d_observed": float(d[i]),
                "r": None if not np.isfinite(r[i]) else float(r[i]),
                "cell_kind": result.cells[i].kind,
                "cell_index": result.cells[i].index,
                "cell_level": result.cells[i].level,
                "action": float(a[i]),
                "q_ctrl": float(q[i]),
                "delta_d_expected": float(dd[i]),
                "outside_window": result.cells[i].kind == "outside",
                "status": OUTSIDE_WINDOW if result.cells[i].kind == "outside" else "INSIDE_WINDOW",
            }
            for i in range(len(d))
        ]
