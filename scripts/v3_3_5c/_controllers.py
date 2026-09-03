from __future__ import annotations

import sys

sys.path.insert(0, "llm/src")
from cantor_guard_v335c.p0_cantor_controller import P0CantorSafetyController  # noqa: E402

from _common import CONFIG, read_json, rho_key


def controller_config() -> dict:
    return read_json(CONFIG / "controller.json")


def make_controller(rho: float, direction, *, eta: float | None = None) -> P0CantorSafetyController:
    config = controller_config()
    if config.get("tau") is None or config.get("W") is None:
        raise ValueError("tau and W must be calibrated before controller construction")
    if eta is None:
        table = config.get("eta_by_rho")
        if not table or rho_key(rho) not in table:
            raise ValueError("eta has not been calibrated for this rho")
        eta = float(table[rho_key(rho)])
    protocol = read_json(CONFIG / "behavioral_protocol.json")
    return P0CantorSafetyController(
        v=direction,
        tau=float(config["tau"]),
        W=float(config["W"]),
        rho=float(rho),
        eta=float(eta),
        safe_sign=int(protocol["safe_sign"]),
        depth=int(config["depth"]),
        leaf_actions=config["leaf_actions"],
        outside_action=float(config["outside_window_action"]),
    )
