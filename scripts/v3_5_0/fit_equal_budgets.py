"""Phase 8: fit eta for q_rms=.03 on the attacked deployment states."""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from cantor_guard_v340.attack import attack_v, attack_w  # noqa: E402
from cantor_guard_v350.linear_risk_controller import LinearRiskController  # noqa: E402
from cantor_guard_v350.risk_cantor_controller import RiskCantorController  # noqa: E402
from _common import CONFIG, Q_CAP, Q_TARGET, RESULTS, RHOS, ensure_final_absent, frozen_actuator, frozen_sensor, read_json, rho_key, write_json  # noqa: E402


def solve_eta(actions, target=Q_TARGET, q_cap=Q_CAP):
    actions = np.asarray(actions, dtype=float)
    def rms(eta): return float(np.sqrt(np.mean(np.minimum(eta * actions, q_cap) ** 2)))
    maximum = rms(1e9)
    if maximum < target: return float("nan"), maximum
    lo, hi = 0.0, 1.0
    while rms(hi) < target: hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if rms(mid) < target: lo = mid
        else: hi = mid
    return (lo + hi) / 2, maximum


def action_vector(controller, H, sensor, actuator, grid):
    rows = []
    for family in ("ATTACK_W", "ATTACK_V"):
        for eps in grid:
            if eps <= 0: continue
            delta = attack_w(sensor, eps, sign=-1) if family == "ATTACK_W" else attack_v(actuator, eps)
            rows.append(controller.correct(H + delta[None, :]).actions)
    return np.concatenate(rows)


def main() -> None:
    ensure_final_absent()
    if (RESULTS / "tables/budget_calibration.json").exists():
        raise SystemExit("budget already calibrated; refuse to refit")
    cfg = read_json(CONFIG / "controller.json"); W_R = float(read_json(CONFIG / "risk_coordinate.json")["W_R"])
    grid = cfg["attack_grid"]
    if not grid: raise SystemExit("attack grid must be frozen before budget fit")
    H = np.load(RESULTS / "cache/h_D_budget_v350.npy")
    sensor, actuator = frozen_sensor(), frozen_actuator()
    arms = {rho_key(r): RiskCantorController(sensor=sensor, actuator=actuator, W_R=W_R,
                                              rho=r, eta=1.0, q_cap=Q_CAP,
                                              leaf_actions=cfg["leaf_actions"])
            for r in RHOS}
    arms["LINEAR"] = LinearRiskController(sensor=sensor, actuator=actuator, W_R=W_R,
                                           eta=1.0, q_cap=Q_CAP)
    per_arm, etas = {}, {}
    for name, ctrl in arms.items():
        actions = action_vector(ctrl, H, sensor, actuator, grid)
        eta, maximum = solve_eta(actions)
        if not np.isfinite(eta):
            per_arm[name] = {"attainable": False, "eta": None,
                             "maximum_attainable_q_rms": maximum,
                             "positive_action_fraction": float(np.mean(actions > 0)),
                             "proof": "q_ctrl<=q_cap and action=0 states remain zero, so max RMS=q_cap*sqrt(P[action>0])"}
            continue
        q_raw = eta * actions; q = np.minimum(q_raw, Q_CAP)
        row = {"attainable": True, "eta": eta, "n_states": len(q),
               "q_rms": float(np.sqrt(np.mean(q**2))), "q_mean": float(q.mean()),
               "q_p95": float(np.quantile(q, .95)), "q_max": float(q.max()),
               "clipping_fraction": float(np.mean(q_raw > Q_CAP)),
               "intervention_frequency": float(np.mean(q > 0))}
        row["within_1pct"] = abs(row["q_rms"] / Q_TARGET - 1) <= .01
        row["cap_ok"] = row["q_max"] <= Q_CAP + 1e-12
        per_arm[name], etas[name] = row, eta
    passed = len(etas) == len(arms) and all(r["within_1pct"] and r["cap_ok"] for r in per_arm.values())
    payload = {"split": "D_budget_v350", "distribution": "both attacks at every nonzero frozen final-grid epsilon",
               "generation_performed": False, "outputs_or_labels_consulted": False,
               "q_target_rms": Q_TARGET, "q_cap": Q_CAP, "selection_targets": [Q_TARGET],
               "clipping_is_selection_gate": False, "per_arm": per_arm,
               "eta_per_arm": etas if passed else None,
               "verdict": "BUD1_MATCHED" if passed else "BUD2_MISMATCH"}
    write_json(RESULTS / "tables/budget_calibration.json", payload)
    cfg["eta_per_arm"] = payload["eta_per_arm"]
    write_json(CONFIG / "controller.json", cfg)
    for name, row in per_arm.items(): print(name, row)
    print(payload["verdict"])
    if not passed: raise SystemExit("q=.03 is mathematically unattainable under the hard cap")


if __name__ == "__main__": main()
