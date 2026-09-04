"""Risk-conditional Cantor controller with positive actions in every risk leaf."""
from __future__ import annotations

from cantor_guard_v350.risk_cantor_controller import RiskCantorController

POSITIVE_LEAF_ACTIONS = tuple((i + 1) / 8 for i in range(8))


class RiskConditionalCantorController(RiskCantorController):
    """V3.5.1 policy: safe=0; every eligible leaf positive; outside=1."""

    def __init__(self, *, leaf_actions=POSITIVE_LEAF_ACTIONS, **kwargs):
        super().__init__(leaf_actions=leaf_actions, **kwargs)
        if not all(a > 0 for a in self.leaf_actions):
            raise ValueError("every V3.5.1 risk-leaf action must be strictly positive")

    def policy_record(self, h):
        rows = super().policy_record(h)
        for row in rows:
            if row["d_observed"] >= 0:
                row["cell_kind"] = "safe"
                row["cell_index"] = None
                row["status"] = "SAFE_SIDE"
            elif row["outside_risk_window"]:
                row["status"] = "OUTSIDE_RISK_WINDOW"
            else:
                row["status"] = "RISK_ELIGIBLE"
        return rows
