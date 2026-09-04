"""Actual one-sided depth-3 Cantor policy acting through the frozen actuator."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cantor_guard_v340.actuator import Actuator
from cantor_guard_v340.sensor_distance import SensorHyperplane

from .one_sided_cantor import Cell, classify, partition
from .risk_coordinate import risk_magnitude, risk_ratio

OUTSIDE_RISK_WINDOW = "OUTSIDE_RISK_WINDOW"
DEFAULT_LEAF_ACTIONS = tuple(i / 7 for i in range(8))


@dataclass(frozen=True)
class RiskControlResult:
    h_corrected: np.ndarray
    delta_h: np.ndarray
    d_observed: np.ndarray
    x_risk: np.ndarray
    r_risk: np.ndarray
    actions: np.ndarray
    q_raw: np.ndarray
    q_ctrl: np.ndarray
    clipped: np.ndarray
    cell_kind: tuple
    cell_index: tuple
    delta_d_expected: np.ndarray


class RiskCantorController:
    """Cantor cells directly select monotone corrective actions on risk space."""

    name_prefix = "CANTOR"

    def __init__(self, *, sensor: SensorHyperplane, actuator: Actuator, W_R: float,
                 rho: float, eta: float, q_cap: float = 0.05, depth: int = 3,
                 leaf_actions=DEFAULT_LEAF_ACTIONS, outside_action: float = 1.0):
        if depth != 3:
            raise ValueError("V3.5.0 freezes depth=3")
        if not np.isfinite(W_R) or W_R <= 0:
            raise ValueError("W_R must be finite and positive")
        if not np.isfinite(eta) or eta < 0:
            raise ValueError("eta must be finite and non-negative")
        if not np.isfinite(q_cap) or q_cap <= 0:
            raise ValueError("q_cap must be finite and positive")
        actions = np.asarray(leaf_actions, dtype=float)
        if actions.shape != (8,) or np.any(np.diff(actions) < 0) or np.any((actions < 0) | (actions > 1)):
            raise ValueError("leaf_actions must be eight nondecreasing values in [0,1]")
        if not 0 <= outside_action <= 1:
            raise ValueError("outside_action must lie in [0,1]")
        self.sensor, self.actuator = sensor, actuator
        self.W_R, self.rho, self.eta = float(W_R), float(rho), float(eta)
        self.q_cap, self.depth = float(q_cap), int(depth)
        self.leaf_actions, self.outside_action = actions, float(outside_action)
        self.leaves, self.guards = partition(self.rho, self.depth)
        self.kappa = float(sensor.coupling(actuator.v_safe))

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
        residual = np.atleast_2d(np.asarray(h, dtype=float))
        d = np.atleast_1d(np.asarray(self.sensor.distance(residual), dtype=float))
        x = np.atleast_1d(np.asarray(risk_magnitude(d), dtype=float))
        r = np.atleast_1d(np.asarray(risk_ratio(x, self.W_R), dtype=float))
        cells, actions = [], []
        for di, xi, ri in zip(d, x, r):
            if di >= 0:
                cells.append(Cell("leaf", 0.0, self.leaves[0].hi, 0, self.depth))
                actions.append(0.0)
            elif xi > self.W_R:
                cells.append(Cell("outside", float("nan"), float("nan")))
                actions.append(self.outside_action)
            else:
                cell = classify(float(ri), self.rho, self.depth)
                cells.append(cell)
                actions.append(self.action_for_cell(cell))
        return d, x, r, tuple(cells), np.asarray(actions, dtype=float)

    def correct(self, h) -> RiskControlResult:
        residual = np.atleast_2d(np.asarray(h, dtype=float))
        if residual.ndim != 2 or residual.shape[1] != self.sensor.w.size:
            raise ValueError("h must match the frozen sensor dimension")
        norms = np.linalg.norm(residual, axis=1)
        if np.any(norms <= 0) or not np.isfinite(norms).all():
            raise ValueError("P0 residual norms must be finite and positive")
        d, x, r, cells, actions = self.classify_state(residual)
        q_raw = self.eta * actions
        q = np.minimum(q_raw, self.q_cap)
        if np.any(q < 0) or np.any(q > self.q_cap + 1e-12):
            raise AssertionError("hard q cap violated")
        if np.any(q[d >= 0] != 0):
            raise AssertionError("safe-side states must receive exactly zero action")
        delta = q[:, None] * norms[:, None] * self.actuator.v_safe[None, :]
        return RiskControlResult(
            h_corrected=residual + delta,
            delta_h=delta,
            d_observed=d,
            x_risk=x,
            r_risk=r,
            actions=actions,
            q_raw=q_raw,
            q_ctrl=q,
            clipped=q_raw > self.q_cap,
            cell_kind=tuple(c.kind for c in cells),
            cell_index=tuple(c.index for c in cells),
            delta_d_expected=q * norms * self.kappa,
        )

    def policy_record(self, h) -> list[dict]:
        res = self.correct(h)
        return [{
            "d_observed": float(res.d_observed[i]),
            "x_risk": float(res.x_risk[i]),
            "r_risk": None if not np.isfinite(res.r_risk[i]) else float(res.r_risk[i]),
            "cell_kind": res.cell_kind[i],
            "cell_index": res.cell_index[i],
            "action": float(res.actions[i]),
            "q_raw": float(res.q_raw[i]),
            "q_ctrl": float(res.q_ctrl[i]),
            "clipped": bool(res.clipped[i]),
            "delta_d_expected": float(res.delta_d_expected[i]),
            "outside_risk_window": res.cell_kind[i] == "outside",
            "status": OUTSIDE_RISK_WINDOW if res.cell_kind[i] == "outside" else "DEFINED_RISK_POLICY",
        } for i in range(len(res.d_observed))]
