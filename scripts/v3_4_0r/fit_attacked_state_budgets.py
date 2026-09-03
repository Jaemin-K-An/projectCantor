"""Phase 5-6 -- fit eta per arm on the ATTACKED deployment distribution.

V3.4.0's defect: eta was solved from P(cell | CLEAN state) while the controller
runs on P(cell | ATTACKED state). Attacks push states into conservative guards,
high-action leaves and OUTSIDE_WINDOW, so E_clean[q^2] != E_attacked[q^2] and
the realised budget overshot by 14-25%.

Here the expectation is taken over the actual design distribution: every prompt
x attack family x epsilon on the frozen generation grid. With the hard cap,
q(eta) = mean over that grid of min(eta*a, q_cap)^2 is continuous and
non-decreasing in eta, so bisection is exact to tolerance.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from cantor_guard_v340.attack import attack_v, attack_w  # noqa: E402
from cantor_guard_v340r.controllers import CappedCantorController, LinearThresholdController  # noqa: E402

from _common import (CONFIG, Q_TARGET, RESULTS, frozen_actuator, frozen_sensor,
                     read_json, rho_key, write_json)  # noqa: E402

V340_FREEZE = ROOT / "configs/v3_4_0/PRE_ANALYSIS_FREEZE.json"


def action_matrix(controller, H, deltas) -> np.ndarray:
    """Raw actions a(cell) over every (prompt, attack) state. eta-independent."""
    rows = []
    for delta in deltas:
        states = H if delta is None else H + delta[None, :]
        rows.append(controller.correct(states).actions)
    return np.concatenate(rows)


def solve_eta(actions: np.ndarray, target: float, q_cap: float) -> float:
    """Bisection on a monotone, continuous q_rms(eta)."""
    def q_rms(eta: float) -> float:
        return float(np.sqrt(np.mean(np.minimum(eta * actions, q_cap) ** 2)))

    if q_rms(q_cap / max(actions.max(), 1e-12) * 1e6) < target:
        return float("nan")  # unreachable even at saturation
    lo, hi = 0.0, 1.0
    while q_rms(hi) < target:
        hi *= 2
        if hi > 1e6:
            return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if q_rms(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> None:
    window = read_json(RESULTS / "tables/external_window.json")
    if window["verdict"] != "ST1_PASS":
        raise SystemExit("STOP: external fixed-W applicability failed; budget fitting is forbidden")
    cfg = read_json(CONFIG / "controller.json")
    frozen = cfg["frozen_from_v340"]
    newcfg = cfg["NEW_IN_V340R"]
    q_cap = float(newcfg["hard_q_cap"]["q_cap"])
    grid = list(read_json(V340_FREEZE)["attacks"]["generation_grid_absolute"])
    sensor, actuator = frozen_sensor(), frozen_actuator()
    W = float(frozen["W"])
    H = np.load(RESULTS / "cache" / "h_D_budget_attacked_r.npy")

    deltas = [None]
    families = ["NONE"]
    for eps in grid:
        if eps <= 0:
            continue
        deltas.append(attack_w(sensor, eps, sign=-1)); families.append(f"ATTACK_W@{eps:.4f}")
        deltas.append(attack_v(actuator, eps)); families.append(f"ATTACK_V@{eps:.4f}")

    arms = {rho_key(r): CappedCantorController(sensor=sensor, actuator=actuator, W=W,
                                               rho=float(r), eta=1.0, q_cap=q_cap,
                                               leaf_actions=frozen["leaf_actions"])
            for r in frozen["rho_family"]}
    arms["LINEAR"] = LinearThresholdController(sensor=sensor, actuator=actuator, W=W,
                                               eta=1.0, q_cap=q_cap)
    actions = {name: action_matrix(c, H, deltas) for name, c in arms.items()}

    grid_targets = [Q_TARGET]
    feasibility = {}
    for target in grid_targets:
        rows = {}
        for name, a in actions.items():
            eta = solve_eta(a, target, q_cap)
            if not np.isfinite(eta):
                rows[name] = {"eta": None, "attainable": False}
                continue
            q = np.minimum(eta * a, q_cap)
            rows[name] = {"eta": float(eta), "attainable": True,
                          "q_rms": float(np.sqrt(np.mean(q**2))),
                          "clip_rate": float(np.mean(eta * a > q_cap)),
                          "q_max": float(q.max())}
        ok = all(r["attainable"] for r in rows.values())
        within = ok and all(abs(r["q_rms"] / target - 1) <= 0.01 for r in rows.values())
        cap_ok = ok and all(r["q_max"] <= q_cap + 1e-12 for r in rows.values())
        feasibility[str(target)] = {"per_arm": rows, "all_attainable": ok,
                                    "all_within_1pct": within, "q_cap_ok": cap_ok,
                                    "feasible": bool(within and cap_ok)}
        print(f"target {target}: attainable={ok} within1%={within} q_cap_ok={cap_ok} "
              f"max_clip={max((r.get('clip_rate', 1) for r in rows.values()), default=1):.3f}")

    feasible = [t for t in grid_targets if feasibility[str(t)]["feasible"]]
    chosen = Q_TARGET if feasible else None
    payload = {
        "split": "D_budget_attacked_r", "n_prompts": int(H.shape[0]),
        "n_states_per_arm": int(len(deltas) * H.shape[0]),
        "distribution": "ATTACKED states over the frozen generation grid, both families",
        "generation_performed": False, "labels_consulted": False,
        "q_cap": q_cap, "target_grid": grid_targets,
        "selection_rule": newcfg["budget_calibration"]["selection_rule"],
        "feasibility": feasibility, "feasible_targets": feasible,
        "q_target_selected": chosen,
        "eta_per_arm": ({k: v["eta"] for k, v in feasibility[str(chosen)]["per_arm"].items()}
                        if chosen is not None else None),
        "verdict": "BUD1_MATCHED" if chosen is not None else "BUD0_NOT_FEASIBLE",
    }
    write_json(RESULTS / "tables" / "budget_calibration.json", payload)
    print(f"\nfeasible targets: {feasible}")
    print(f"selected q_target = {chosen}   verdict {payload['verdict']}")
    if chosen is not None:
        print(f"\n{'arm':<8}{'eta':>10}{'q_rms':>9}{'clip%':>8}{'q_max':>9}")
        for name, row in feasibility[str(chosen)]["per_arm"].items():
            print(f"{name:<8}{row['eta']:>10.4f}{row['q_rms']:>9.4f}"
                  f"{row['clip_rate']:>8.3f}{row['q_max']:>9.4f}")


if __name__ == "__main__":
    main()
