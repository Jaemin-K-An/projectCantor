"""Split-conformal upper window calibrated conditional on the risk half-space."""
from __future__ import annotations

import numpy as np

from cantor_guard_v350.conformal_window import ConformalWindow, calibrate_upper_window


def calibrate_conditional_risk_window(d, *, n_risk: int = 200,
                                      alpha: float = 0.05) -> ConformalWindow:
    """Use the first ``n_risk`` negative distances and exclude all safe zeros."""
    values = np.asarray(d, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("sensor distances must be finite")
    risk = -values[values < 0]
    if risk.size < int(n_risk):
        raise ValueError(f"RISK_CAL_INSUFFICIENT: {risk.size} < {n_risk}")
    return calibrate_upper_window(risk[:int(n_risk)], alpha)


__all__ = ["ConformalWindow", "calibrate_conditional_risk_window"]
