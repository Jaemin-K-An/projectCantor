"""Collect outcome-free pre-control attacked states for D_budget_v351."""
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
from cantor_guard_v351.risk_coordinate import risk_magnitude  # noqa: E402
from _common import (CONFIG, RESULTS, ensure_final_outputs_absent, frozen_actuator,
                     frozen_sensor, read_json, sha256, write_json)  # noqa: E402
from _model import clean_residuals  # noqa: E402


def main() -> None:
    ensure_final_outputs_absent()
    output = RESULTS / "tables/budget_states_collected.json"
    if output.exists():
        raise SystemExit("budget states already collected; refuse to replace")
    grid_cfg = read_json(CONFIG / "attack_grid.json")
    if grid_cfg.get("status") != "ATTACK_GRID_FROZEN":
        raise SystemExit("attack grid is not frozen")
    grid = list(map(float, grid_cfg["attack_grid"]))
    prompts_path = RESULTS / "cache/D_budget_v351.csv"
    prompts = pd.read_csv(prompts_path)
    if len(prompts) != 300:
        raise SystemExit("D_budget_v351 must contain exactly 300 prompts")

    seed_everything(20260905)
    bundle = load_model("qwen2.5-0.5b-instruct")
    H = clean_residuals(bundle, prompts.prompt.astype(str).tolist(), layer=14, batch_size=8)
    h_path = RESULTS / "cache/h_D_budget_v351.npy"
    np.save(h_path, H)
    sensor, actuator = frozen_sensor(), frozen_actuator()
    d_clean = np.asarray(sensor.distance(H), dtype=float)
    x_clean = np.asarray(risk_magnitude(d_clean), dtype=float)
    rows = []
    for family in ("ATTACK_V", "ATTACK_W"):
        for epsilon in grid:
            delta = attack_v(actuator, epsilon) if family == "ATTACK_V" else attack_w(sensor, epsilon, sign=-1)
            attacked = H + delta[None, :]
            d_attacked = np.asarray(sensor.distance(attacked), dtype=float)
            x_attacked = np.asarray(risk_magnitude(d_attacked), dtype=float)
            for i, pid in enumerate(prompts.pid.astype(str)):
                rows.append({
                    "pid": pid, "family": family, "epsilon": epsilon,
                    "h_attacked_norm": float(np.linalg.norm(attacked[i])),
                    "d_clean": d_clean[i], "d_attacked": d_attacked[i],
                    "x_clean": x_clean[i], "x_attacked": x_attacked[i],
                    "risk_eligible": bool(d_attacked[i] < 0),
                    "delta_d": d_attacked[i] - d_clean[i],
                    "delta_x": x_attacked[i] - x_clean[i],
                    "delta_r_R": (x_attacked[i] - x_clean[i]) / float(grid_cfg["W_R"]),
                })
    states = pd.DataFrame(rows)
    states_path = RESULTS / "raw/attacked_budget_states.csv"
    states.to_csv(states_path, index=False)
    write_json(output, {
        "split": "D_budget_v351", "n_prompts": len(prompts), "d_model": H.shape[1],
        "attack_families": ["ATTACK_V", "ATTACK_W"], "attack_grid": grid,
        "n_attacked_states": len(states),
        "risk_eligible_n": int(states.risk_eligible.sum()),
        "risk_eligible_prevalence": float(states.risk_eligible.mean()),
        "per_family_risk_eligible_prevalence": {
            key: float(group.risk_eligible.mean()) for key, group in states.groupby("family", sort=False)
        },
        "generation_performed": False, "outputs_or_labels_consulted": False,
        "epsilon_R_C": float(grid_cfg["epsilon_R_C"]),
        "mean_h_norm": float(np.linalg.norm(H, axis=1).mean()),
        "sensor_actuator_coupling": float(sensor.coupling(actuator.v_safe)),
        "hashes": {
            "budget_csv_sha256": sha256(prompts_path),
            "budget_residuals_sha256": sha256(h_path),
            "attacked_states_csv_sha256": sha256(states_path),
            "attack_grid_sha256": sha256(CONFIG / "attack_grid.json"),
        },
        "model": bundle.provenance(),
        "verdict": "BUDGET_STATES_COLLECTED_WITHOUT_ENDPOINTS",
    })
    print(f"D_budget_v351 n={len(prompts)} states={len(states)} "
          f"risk_prevalence={states.risk_eligible.mean():.6f}")
    print("no generation; no outputs; no labels")


if __name__ == "__main__":
    main()
