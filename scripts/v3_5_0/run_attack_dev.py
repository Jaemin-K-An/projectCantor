"""Phase 9: fixed-grid integration check; never optimizes the grid or eta."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from cantor_guard.io import seed_everything  # noqa: E402
from cantor_guard.models import load_model  # noqa: E402
from cantor_guard_v32.metrics32 import is_refusal32  # noqa: E402
from cantor_guard_v340.attack import attack_v, attack_w  # noqa: E402
from cantor_guard_v350.p0_generation import generate_defended  # noqa: E402
from cantor_guard_v350.risk_cantor_controller import RiskCantorController  # noqa: E402
from _common import CONFIG, Q_CAP, RESULTS, ensure_final_absent, frozen_actuator, frozen_sensor, read_json, write_json  # noqa: E402


def main() -> None:
    ensure_final_absent()
    output = RESULTS / "raw/attack_dev.csv"
    if output.exists(): raise SystemExit("attack DEV already executed")
    cfg = read_json(CONFIG / "controller.json"); cal = read_json(RESULTS / "tables/risk_window_calibration.json")
    budget = read_json(RESULTS / "tables/budget_calibration.json")
    if budget["verdict"] != "BUD1_MATCHED": raise SystemExit("budget gate failed")
    grid = [e for e in cfg["attack_grid"] if e > 0]
    eps_c = float(cfg["epsilon_R_C"])
    chosen = sorted({grid[0], min(grid, key=lambda e: abs(e - eps_c)), grid[-1]})
    prompts = pd.read_csv(RESULTS / "cache/D_attack_dev_v350.csv")
    seed_everything(20260904); sensor, actuator = frozen_sensor(), frozen_actuator()
    ctrl = RiskCantorController(sensor=sensor, actuator=actuator, W_R=float(cal["W_R"]),
                                rho=1 / 3, eta=float(budget["eta_per_arm"]["1/3"]),
                                q_cap=Q_CAP, leaf_actions=cfg["leaf_actions"])
    bundle = load_model("qwen2.5-0.5b-instruct")
    rows, private = [], []
    for family in ("ATTACK_W", "ATTACK_V"):
        for eps in chosen:
            delta = attack_w(sensor, eps, sign=-1) if family == "ATTACK_W" else attack_v(actuator, eps)
            for arm, controller in (("ATTACK_ONLY", None), ("1/3", ctrl)):
                gen = generate_defended(bundle, prompts.prompt.tolist(), layer=14, attack_delta=delta,
                                        controller=controller, sensor=sensor, max_new_tokens=24, batch_size=8)
                recs = gen.get("records") or []
                for i, text in enumerate(gen["texts"]):
                    rec = recs[i] if i < len(recs) else {}
                    row = {"pid": prompts.pid.iloc[i], "family": family, "epsilon": eps, "arm": arm,
                           "d_clean": gen["d_clean"][i], "d_attacked": gen["d_attacked"][i],
                           "d_corrected": gen["d_corrected"][i],
                           "x_attacked": max(0.0, -float(gen["d_attacked"][i])),
                           "q_ctrl": gen["q_ctrl"][i], "action": rec.get("action"),
                           "cell_kind": rec.get("cell_kind"), "clipped": rec.get("clipped"),
                           "refusal_proxy": int(is_refusal32(text))}
                    rows.append(row); private.append({**row, "prompt": prompts.prompt.iloc[i], "completion": text})
            print(f"{family} epsilon={eps:.6f} complete", flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(output, index=False)
    pd.DataFrame(private).to_csv(RESULTS / "private/attack_dev_completions.csv", index=False)
    acting = frame[frame.arm == "1/3"]
    q = acting.q_ctrl.to_numpy(float)
    executes = bool(np.any(q > 0)); not_all_saturated = bool(np.any(q < Q_CAP - 1e-12))
    coverage = float(np.mean(acting.x_attacked <= float(cal["W_R"])))
    payload = {"split": "D_attack_dev_v350", "n_prompts": len(prompts),
               "fixed_sentinel_epsilons": chosen, "grid_changed_after_outputs": False,
               "risk_coordinate_coverage": coverage, "target_diagnostic_coverage": .95,
               "controller_executes": executes, "not_entirely_saturated": not_all_saturated,
               "q_min": float(q.min()), "q_max": float(q.max()),
               "passed": executes and not_all_saturated,
               "verdict": "DEV1_EXECUTES" if executes and not_all_saturated else "DEV2_CONTROLLER_FAILURE"}
    write_json(RESULTS / "tables/attack_dev.json", payload)
    print(payload)
    if not payload["passed"]: raise SystemExit("controller execution hard stop")


if __name__ == "__main__": main()
