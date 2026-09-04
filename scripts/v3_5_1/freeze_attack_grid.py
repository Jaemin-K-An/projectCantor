"""Construct and freeze the preregistered attack grid from the frozen W_R."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from cantor_guard_v351.one_sided_cantor import epsilon_r_cantor  # noqa: E402
from _common import CONFIG, RESULTS, ensure_final_outputs_absent, read_json, write_json  # noqa: E402

RELATIVE_LEVELS = (.5, .9, .99, 1.01, 1.25, 2.0, 4.0)
FIXED_LARGE_EPSILONS = (.1013565, .405426, .92065487, 1.84130973, 3.68261946, 7.38213168)


def main() -> None:
    ensure_final_outputs_absent()
    target = CONFIG / "attack_grid.json"
    if target.exists():
        raise SystemExit("attack grid already frozen; refuse to modify")
    risk = read_json(RESULTS / "tables/risk_conditional_window.json")
    if risk.get("verdict") != "RISK1_CONDITIONAL_WINDOW_CALIBRATED":
        raise SystemExit("risk window gate not passed")
    W_R = float(risk["W_R"])
    eps_c = epsilon_r_cantor(W_R)
    relative_values = [float(level * eps_c) for level in RELATIVE_LEVELS]
    grid = sorted({0.0, *relative_values, *FIXED_LARGE_EPSILONS})
    payload = {
        "version": "3.5.1",
        "W_R": W_R,
        "epsilon_R_C": eps_c,
        "relative_multipliers": list(RELATIVE_LEVELS),
        "relative_values": relative_values,
        "inherited_v350_fixed_large_epsilons": list(FIXED_LARGE_EPSILONS),
        "attack_grid": grid,
        "attack_families": ["ATTACK_V", "ATTACK_W"],
        "primary_attack": "ATTACK_V",
        "frozen_before_attack_development_outputs": True,
        "outputs_or_labels_consulted": False,
        "status": "ATTACK_GRID_FROZEN",
    }
    write_json(target, payload)
    cfg = read_json(CONFIG / "controller.json")
    cfg["attack_grid"] = grid
    cfg["epsilon_R_C"] = eps_c
    cfg["attack_grid_artifact"] = str(target.relative_to(ROOT))
    cfg["attack_grid_frozen_before_attack_dev_outputs"] = True
    write_json(CONFIG / "controller.json", cfg)
    print(f"ATTACK_GRID_FROZEN n={len(grid)} epsilon_R_C={eps_c:.12f}")


if __name__ == "__main__":
    main()
