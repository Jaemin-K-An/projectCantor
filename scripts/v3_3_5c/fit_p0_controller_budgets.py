"""Choose one rho-independent controller budget and fit eta on D_budget only."""
from __future__ import annotations

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "llm/src")
from cantor_guard.models import load_model  # noqa: E402
from cantor_guard_v32.metrics32 import coherence32  # noqa: E402
from cantor_guard_v335a.p0_residual import last_valid_prompt_residuals  # noqa: E402
from cantor_guard_v335c.p0_attack_generation import generate_attacked_p0  # noqa: E402

from _common import CONFIG, RESULTS, behavioral_protocol, load_direction, read_json, rho_key, write_json
from _controllers import make_controller
from run_p0_attack_dev import common_grid


def eta_for_target(actions, target: float) -> float:
    rms = float(np.sqrt(np.mean(np.asarray(actions, dtype=float) ** 2)))
    if rms <= 0:
        raise ValueError("controller has no measurable actions on budget split")
    return float(target / rms)


def main() -> None:
    config = read_json(CONFIG / "controller.json")
    if config.get("tau") is None or config.get("W") is None:
        raise SystemExit("STOP: affine window not calibrated")
    protocol = behavioral_protocol()
    prompts = pd.read_csv(RESULTS / "cache/D_budget_P0_335c.csv")
    direction = load_direction(protocol)
    bundle = load_model(protocol["model"])
    residuals = last_valid_prompt_residuals(
        bundle, prompts.prompt.tolist(), int(protocol["layer"]), batch_size=8,
    )
    rhos = [float(x) for x in config["rho_family"]]
    prospective_grid = common_grid(rhos, float(config["W"]))
    unsafe_sign = int(read_json(CONFIG / "attack_grid.json")["unsafe_sign"])
    action_by_rho = {}
    for rho in rhos:
        attacked_family = np.concatenate(
            [residuals + unsafe_sign * epsilon * direction[None, :] for epsilon in prospective_grid],
            axis=0,
        )
        probe = make_controller(rho, direction, eta=1.0).correct(attacked_family)
        action_by_rho[rho_key(rho)] = np.asarray(probe.actions, dtype=float)

    candidates_report = []
    selected = None
    selected_eta = None
    causal_floor = float(config["prior_causal_floor"])
    cap = float(config["q_norm_fraction_cap"])
    for target in [float(x) for x in config["candidate_q_ctrl_rms"]]:
        if target < causal_floor:
            candidates_report.append({
                "target": target, "eligible": False,
                "reason": "below outcome-independent causal floor established before V3.3.5c",
            })
            continue
        by_rho = {}
        eta_map = {}
        all_pass = True
        for rho in rhos:
            key = rho_key(rho)
            eta = eta_for_target(action_by_rho[key], target)
            eta_map[key] = eta
            controller = make_controller(rho, direction, eta=eta)
            generated = generate_attacked_p0(
                bundle, prompts.prompt.tolist(), layer=int(protocol["layer"]),
                v=direction, controller=controller, attack_epsilon=0,
                max_new_tokens=int(protocol["max_new_tokens"]), batch_size=8,
            )
            coherence = np.asarray([coherence32(text) for text in generated["texts"]])
            degeneration = coherence < float(protocol["coherence_gate"])
            q_clean = np.asarray(generated["q_ctrl"], dtype=float)
            q_calibration = eta * action_by_rho[key]
            record = {
                "eta": eta,
                "intervention_frequency": float(np.mean(q_calibration > 0)),
                "q_mean": float(q_calibration.mean()),
                "q_rms": float(np.sqrt(np.mean(q_calibration**2))),
                "q_p95": float(np.quantile(q_calibration, 0.95)),
                "q_max": float(q_calibration.max()),
                "clean_deployment_q_rms": float(np.sqrt(np.mean(q_clean**2))),
                "coherence": float(coherence.mean()),
                "degeneration_rate": float(degeneration.mean()),
            }
            record["passes"] = bool(
                record["intervention_frequency"] > 0
                and record["coherence"] >= float(protocol["coherence_gate"])
                and record["degeneration_rate"] <= float(protocol["degeneration_rate_gate"])
                and record["q_max"] <= cap + 1e-12
                and abs(record["q_rms"] / target - 1) <= float(config["budget_tolerance_relative"])
            )
            all_pass &= record["passes"]
            by_rho[key] = record
            print(f"target={target:.3f} rho={key} q_rms={record['q_rms']:.4f} coh={record['coherence']:.3f} pass={record['passes']}", flush=True)
        candidates_report.append({"target": target, "eligible": all_pass, "rho": by_rho})
        if all_pass:
            selected, selected_eta = target, eta_map
            break
    if selected is None:
        write_json(RESULTS / "tables/controller_budget_calibration.json", {
            "status": "CONTROLLER_BUDGET_UNACHIEVABLE", "candidates": candidates_report,
        })
        raise SystemExit("STOP: no common meaningful non-degenerate controller budget")
    report = {
        "status": "CONTROLLER_BUDGET_FROZEN",
        "selection_uses_rho_outcomes": False,
        "selection_rule": config["budget_selection_rule"],
        "calibration_attack_grid": prospective_grid,
        "calibration_population": "D_budget_P0_335c crossed with the outcome-independent common absolute attack grid",
        "q_target": selected,
        "eta_by_rho": selected_eta,
        "candidates": candidates_report,
    }
    write_json(RESULTS / "tables/controller_budget_calibration.json", report)
    config["q_target"] = selected
    config["eta_by_rho"] = selected_eta
    write_json(CONFIG / "controller.json", config)
    print(f"FROZEN q_ctrl,rms target={selected:.3f}")


if __name__ == "__main__":
    main()
