"""Matched nonrecursive controller on the V3.5.0 one-sided risk coordinate."""
from __future__ import annotations

import numpy as np

from .risk_cantor_controller import RiskControlResult
from .risk_coordinate import risk_magnitude, risk_ratio


class LinearRiskController:
    name_prefix = "LINEAR"

    def __init__(self, *, sensor, actuator, W_R: float, eta: float,
                 q_cap: float = 0.05, outside_action: float = 1.0):
        if not np.isfinite(W_R) or W_R <= 0:
            raise ValueError("W_R must be finite and positive")
        if not np.isfinite(eta) or eta < 0:
            raise ValueError("eta must be finite and non-negative")
        if not np.isfinite(q_cap) or q_cap <= 0:
            raise ValueError("q_cap must be finite and positive")
        self.sensor, self.actuator = sensor, actuator
        self.W_R, self.eta, self.q_cap = float(W_R), float(eta), float(q_cap)
        self.outside_action = float(outside_action)
        self.kappa = float(sensor.coupling(actuator.v_safe))
        self.rho = None

    def correct(self, h) -> RiskControlResult:
        residual = np.atleast_2d(np.asarray(h, dtype=float))
        norms = np.linalg.norm(residual, axis=1)
        d = np.atleast_1d(np.asarray(self.sensor.distance(residual), dtype=float))
        x = np.atleast_1d(np.asarray(risk_magnitude(d), dtype=float))
        r = np.atleast_1d(np.asarray(risk_ratio(x, self.W_R), dtype=float))
        outside = x > self.W_R
        actions = np.where(d >= 0, 0.0, np.where(outside, self.outside_action, r))
        q_raw = self.eta * actions
        q = np.minimum(q_raw, self.q_cap)
        if np.any(q[d >= 0] != 0) or np.any(q > self.q_cap + 1e-12):
            raise AssertionError("one-sided safe policy or q cap violated")
        delta = q[:, None] * norms[:, None] * self.actuator.v_safe[None, :]
        kinds = np.where(d >= 0, "safe", np.where(outside, "outside", "linear"))
        return RiskControlResult(
            h_corrected=residual + delta,
            delta_h=delta,
            d_observed=d,
            x_risk=x,
            r_risk=r,
            actions=np.asarray(actions, dtype=float),
            q_raw=q_raw,
            q_ctrl=q,
            clipped=q_raw > self.q_cap,
            cell_kind=tuple(kinds.tolist()),
            cell_index=tuple([None] * len(d)),
            delta_d_expected=q * norms * self.kappa,
        )

    def policy_record(self, h) -> list[dict]:
        res = self.correct(h)
        return [{
            "d_observed": float(res.d_observed[i]),
            "x_risk": float(res.x_risk[i]),
            "r_risk": None if not np.isfinite(res.r_risk[i]) else float(res.r_risk[i]),
            "cell_kind": res.cell_kind[i],
            "cell_index": None,
            "action": float(res.actions[i]),
            "q_raw": float(res.q_raw[i]),
            "q_ctrl": float(res.q_ctrl[i]),
            "clipped": bool(res.clipped[i]),
            "delta_d_expected": float(res.delta_d_expected[i]),
            "outside_risk_window": res.cell_kind[i] == "outside",
            "status": "OUTSIDE_RISK_WINDOW" if res.cell_kind[i] == "outside" else "DEFINED_RISK_POLICY",
        } for i in range(len(res.d_observed))]
