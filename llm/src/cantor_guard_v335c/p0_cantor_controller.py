"""Actual P0 residual controller using a frozen affine Cantor partition."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .affine_coordinate import AffineCoordinate, OUTSIDE_WINDOW
from .cantor_geometry import Cell, classify, partition
from .p0_normalized_dose import normalize_direction


@dataclass(frozen=True)
class ControllerBatchResult:
    h_corrected: np.ndarray
    delta_h_controller: np.ndarray
    z_observed: np.ndarray
    r: np.ndarray
    cells: tuple[Cell, ...]
    actions: np.ndarray
    q_ctrl: np.ndarray


class P0CantorSafetyController:
    """Depth-3 rho-family controller that consumes actual residual vectors.

    The direction is shared across safe/unsafe motion.  ``safe_sign`` is frozen
    once; the affine orientation is its negative so larger r means higher risk.
    """

    def __init__(
        self,
        *,
        v,
        tau: float,
        W: float,
        rho: float,
        eta: float,
        safe_sign: int,
        depth: int = 3,
        leaf_actions=None,
        outside_action: float = 1.0,
    ):
        if depth != 3:
            raise ValueError("V3.3.5c freezes depth=3")
        if safe_sign not in (-1, 1):
            raise ValueError("safe_sign must be -1 or +1")
        if eta < 0 or not np.isfinite(eta):
            raise ValueError("eta must be finite and non-negative")
        self.v = normalize_direction(v)
        self.tau = float(tau)
        self.W = float(W)
        self.rho = float(rho)
        self.eta = float(eta)
        self.safe_sign = int(safe_sign)
        self.depth = depth
        self.coordinate = AffineCoordinate(self.tau, self.W, -self.safe_sign)
        self.leaves, self.guards = partition(self.rho, self.depth)
        actions = np.linspace(0, 1, 8) if leaf_actions is None else np.asarray(leaf_actions, dtype=float)
        if actions.shape != (8,) or np.any(np.diff(actions) < 0) or np.any((actions < 0) | (actions > 1)):
            raise ValueError("leaf_actions must be eight nondecreasing values in [0,1]")
        self.leaf_actions = actions
        if not 0 <= outside_action <= 1:
            raise ValueError("outside_action must lie in [0,1]")
        self.outside_action = float(outside_action)

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

    def correct(self, h) -> ControllerBatchResult:
        residual = np.asarray(h, dtype=float)
        squeeze = residual.ndim == 1
        if squeeze:
            residual = residual[None, :]
        if residual.ndim != 2 or residual.shape[1] != self.v.size:
            raise ValueError("h must have shape [d] or [batch,d] matching v")
        norms = np.linalg.norm(residual, axis=1)
        if np.any(norms <= 0) or not np.isfinite(norms).all():
            raise ValueError("actual P0 residuals must have finite positive norm")
        z = residual @ self.v
        r = np.asarray(self.coordinate.transform(z), dtype=float)
        cells = tuple(
            classify(float(ri), self.rho, self.depth)
            if np.isfinite(ri)
            else Cell("outside", float("nan"), float("nan"))
            for ri in r
        )
        actions = np.asarray([self.action_for_cell(cell) for cell in cells])
        q = self.eta * actions
        delta = self.safe_sign * q[:, None] * norms[:, None] * self.v[None, :]
        corrected = residual + delta
        if squeeze:
            corrected, delta, z, r, actions, q = (
                x[0] for x in (corrected, delta, z, r, actions, q)
            )
        return ControllerBatchResult(corrected, delta, z, r, cells, actions, q)

    def intervene(self, h, _v_ignored=None):
        """Compatibility adapter for the existing P0 generation hook."""
        result = self.correct(h)
        return result.delta_h_controller, np.linalg.norm(result.delta_h_controller, axis=-1)

    def policy_record(self, h) -> list[dict]:
        result = self.correct(h)
        z = np.atleast_1d(result.z_observed)
        r = np.atleast_1d(result.r)
        a = np.atleast_1d(result.actions)
        q = np.atleast_1d(result.q_ctrl)
        return [
            {
                "z_observed": float(z[i]),
                "r": None if not np.isfinite(r[i]) else float(r[i]),
                "cell_kind": result.cells[i].kind,
                "cell_index": result.cells[i].index,
                "cell_level": result.cells[i].level,
                "action": float(a[i]),
                "q_ctrl": float(q[i]),
                "outside_window": result.cells[i].kind == "outside",
                "status": OUTSIDE_WINDOW if result.cells[i].kind == "outside" else "INSIDE_WINDOW",
            }
            for i in range(len(z))
        ]
