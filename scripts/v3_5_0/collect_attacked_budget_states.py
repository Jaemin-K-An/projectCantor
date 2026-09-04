"""Phase 7: collect fresh clean P0 states for outcome-free attacked budgets."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from cantor_guard.io import seed_everything  # noqa: E402
from cantor_guard.models import load_model  # noqa: E402
from cantor_guard_v340.attack import attack_v, attack_w  # noqa: E402
from cantor_guard_v350.one_sided_cantor import epsilon_r_cantor  # noqa: E402
from cantor_guard_v350.risk_coordinate import risk_magnitude  # noqa: E402
from _common import CONFIG, RESULTS, ensure_final_absent, frozen_actuator, frozen_sensor, read_json, write_json  # noqa: E402
from _model import clean_residuals  # noqa: E402

RELATIVE_LEVELS = (0.5, 0.9, 0.99, 1.01, 1.25, 2.0, 4.0)
FIXED_CAUSAL_LEVELS = (0.1013565, 0.405426, 0.92065487, 1.84130973, 3.68261946, 7.38213168)


def attack_grid(W_R: float):
    eps_c = epsilon_r_cantor(W_R)
    values = {0.0, *(float(level) * eps_c for level in RELATIVE_LEVELS), *FIXED_CAUSAL_LEVELS}
    return sorted(round(x, 12) for x in values)


def main() -> None:
    ensure_final_absent()
    if (RESULTS / "tables/budget_states_collected.json").exists():
        raise SystemExit("attacked budget states already collected")
    cal = read_json(RESULTS / "tables/risk_window_calibration.json")
    W_R = float(cal["W_R"]); grid = attack_grid(W_R)
    cfg = read_json(CONFIG / "controller.json")
    cfg["attack_grid"] = grid; cfg["epsilon_R_C"] = epsilon_r_cantor(W_R)
    cfg["attack_grid_frozen_before_attack_dev_outputs"] = True
    write_json(CONFIG / "controller.json", cfg)

    seed_everything(20260904)
    prompts = pd.read_csv(RESULTS / "cache/D_budget_v350.csv")
    if len(prompts) < 300: raise SystemExit("D_budget_v350 requires n>=300")
    bundle = load_model("qwen2.5-0.5b-instruct")
    H = clean_residuals(bundle, prompts.prompt.tolist(), layer=14, batch_size=8)
    np.save(RESULTS / "cache/h_D_budget_v350.npy", H)
    sensor, actuator = frozen_sensor(), frozen_actuator()
    rows = []
    for family in ("ATTACK_W", "ATTACK_V"):
        for eps in grid:
            delta = attack_w(sensor, eps, sign=-1) if family == "ATTACK_W" else attack_v(actuator, eps)
            attacked = H + delta[None, :]
            d_clean = np.asarray(sensor.distance(H)); d_att = np.asarray(sensor.distance(attacked))
            x_clean = np.asarray(risk_magnitude(d_clean)); x_att = np.asarray(risk_magnitude(d_att))
            for i, pid in enumerate(prompts.pid):
                rows.append({"pid": pid, "family": family, "epsilon": eps,
                             "h_attacked_norm": float(np.linalg.norm(attacked[i])),
                             "d_clean": d_clean[i], "d_attacked": d_att[i],
                             "x_clean": x_clean[i], "x_attacked": x_att[i],
                             "delta_d": d_att[i] - d_clean[i], "delta_x": x_att[i] - x_clean[i],
                             "delta_r_R": (x_att[i] - x_clean[i]) / W_R})
    states = pd.DataFrame(rows)
    states.to_csv(RESULTS / "raw/attacked_budget_states.csv", index=False)
    write_json(RESULTS / "tables/budget_states_collected.json", {
        "split": "D_budget_v350", "n_prompts": len(prompts), "d_model": H.shape[1],
        "attack_families": ["ATTACK_W", "ATTACK_V"], "attack_grid": grid,
        "n_attacked_states": len(states), "generation_performed": False,
        "outputs_or_labels_consulted": False, "epsilon_R_C": epsilon_r_cantor(W_R),
        "mean_h_norm": float(np.linalg.norm(H, axis=1).mean()),
    })
    print(f"D_budget_v350 n={len(prompts)}, states={len(states)}, grid={grid}")
    print("no generation; no outputs; no labels")


if __name__ == "__main__": main()
