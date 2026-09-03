"""Phase 11 -- one common intervention budget, one eta per rho.

Because q_ctrl = eta * a(cell), any target is reachable by scaling eta; the
real question is which target is MEANINGFUL.  The frozen rule picks the
smallest candidate that lets the controller move a typical state by at least
the Cantor certificate itself -- the same yardstick as the controllability
gate, and independent of which rho happens to do well.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "llm/src"))
from cantor_guard_v340.actuator import Actuator  # noqa: E402
from cantor_guard_v340.sensor_actuator_controller import SensorActuatorCantorController  # noqa: E402
from cantor_guard_v340.sensor_distance import SensorHyperplane  # noqa: E402

from _common import CONFIG, RESULTS, read_json, rho_key, write_json  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
TARGET_GRID = (0.01, 0.02, 0.03, 0.05)


def main() -> None:
    ctrl = read_json(CONFIG / "controller.json")
    act_cfg = read_json(CONFIG / "actuator.json")
    geom = read_json(RESULTS / "tables" / "sensor_window_and_coupling.json")
    fit = read_json(RESULTS / "tables" / "sensor_confirm.json")
    sensor = SensorHyperplane(np.load(RESULTS / "cache" / "sensor_w.npy"), float(fit["b"]))
    actuator = Actuator(np.load(ROOT / act_cfg["direction_file"]).astype(float).reshape(-1),
                        int(act_cfg["safe_sign"]))
    W, kappa, eps_c = float(geom["W"]), float(geom["coupling"]["kappa"]), float(geom["epsilon_cantor"])
    q_cap = float(ctrl["CONTROLLABILITY_GATE"]["q_cap"])

    H = np.load(RESULTS / "cache" / "h_D_controller_budget.npy")
    norms = np.linalg.norm(H, axis=1)
    median_norm = float(np.median(norms))

    q_needed = eps_c / (median_norm * abs(kappa))
    eligible = [t for t in TARGET_GRID if t >= q_needed and t <= q_cap]
    if not eligible:
        raise SystemExit(f"no candidate budget in {TARGET_GRID} satisfies q>={q_needed:.4f} and q<={q_cap}")
    target = float(min(eligible))

    per_rho = {}
    for rho in ctrl["rho_family"]:
        probe = SensorActuatorCantorController(
            sensor=sensor, actuator=actuator, W=W, rho=float(rho), eta=1.0,
            leaf_actions=ctrl["leaf_actions"],
        )
        actions = np.asarray([r["action"] for r in probe.policy_record(H)], dtype=float)
        rms_action = float(np.sqrt(np.mean(actions**2)))
        if rms_action <= 0:
            raise SystemExit(f"rho={rho}: controller never acts on the budget split")
        eta = target / rms_action
        final = SensorActuatorCantorController(
            sensor=sensor, actuator=actuator, W=W, rho=float(rho), eta=eta,
            leaf_actions=ctrl["leaf_actions"],
        )
        records = final.policy_record(H)
        q = np.asarray([r["q_ctrl"] for r in records], dtype=float)
        kinds = [r["cell_kind"] for r in records]
        per_rho[rho_key(rho)] = {
            "rho": float(rho), "eta": float(eta), "rms_action_at_eta1": rms_action,
            "q_rms": float(np.sqrt(np.mean(q**2))), "q_mean": float(q.mean()),
            "q_p95": float(np.quantile(q, 0.95)), "q_max": float(q.max()),
            "intervention_frequency": float(np.mean(q > 0)),
            "guard_frequency": float(np.mean([k == "guard" for k in kinds])),
            "leaf_frequency": float(np.mean([k == "leaf" for k in kinds])),
            "outside_frequency": float(np.mean([k == "outside" for k in kinds])),
            "mean_expected_delta_d": float(np.mean([r["delta_d_expected"] for r in records])),
        }
    write_json(RESULTS / "tables" / "controller_budgets.json", {
        "target_grid": list(TARGET_GRID), "q_cap": q_cap,
        "q_needed_for_one_certificate": q_needed,
        "selection_rule": ctrl["budget"]["target_selection_rule"],
        "selection_basis": "median ||h|| on D_controller_budget and |kappa|; no rho outcome is consulted",
        "median_h_norm": median_norm, "kappa": kappa, "epsilon_cantor": eps_c,
        "q_target_selected": target, "tolerance": float(ctrl["budget"]["final_tolerance"]),
        "per_rho": per_rho,
    })
    print(f"median ||h|| = {median_norm:.3f}, |kappa| = {abs(kappa):.4f}, epsilon_C = {eps_c:.4f}")
    print(f"q needed to move one certificate = {q_needed:.4f}  ->  target q_rms = {target}")
    print(f"\n{'rho':<7}{'eta':>10}{'q_rms':>9}{'q_p95':>9}{'act%':>8}{'guard%':>9}{'leaf%':>8}{'out%':>7}")
    for key, row in per_rho.items():
        print(f"{key:<7}{row['eta']:>10.4f}{row['q_rms']:>9.4f}{row['q_p95']:>9.4f}"
              f"{row['intervention_frequency']:>8.2f}{row['guard_frequency']:>9.2f}"
              f"{row['leaf_frequency']:>8.2f}{row['outside_frequency']:>7.2f}")


if __name__ == "__main__":
    main()
