"""Seal every V3.3.5c choice before D_final is touched."""
from __future__ import annotations

import datetime as dt
import hashlib
import pathlib
import subprocess

from _common import CONFIG, RESULTS, read_json, write_json


FILES = [
    "results/v3_3_5a/cache/v_p0.npy",
    "configs/v3_3_5c/splits.json",
    "configs/v3_3_5c/behavioral_protocol.json",
    "configs/v3_3_5c/evaluator.json",
    "configs/v3_3_5c/controller.json",
    "configs/v3_3_5c/attack_grid.json",
    "scripts/v3_3_5c/final_claim_check_v335c.py",
    "scripts/v3_3_5c/analyse_generation.py",
    "scripts/v3_3_5c/run_final_p0_cantor.py",
    "scripts/v3_3_5c/run_certificate_validation.py",
    "scripts/v3_3_5c/audit_final_budgets.py",
    "scripts/v3_3_5c/score_semantic_generation.py",
    "scripts/v3_3_5c/analyse_failure_thresholds.py",
    "scripts/v3_3_5c/analyse_first_token.py",
    "scripts/v3_3_5c/analyse_utility.py",
    "results/v3_3_5c/tables/p0_dose_grid_freeze.json",
    "results/v3_3_5c/tables/p0_behavioral_boundary.json",
    "results/v3_3_5c/tables/p0_window.json",
    "results/v3_3_5c/tables/controller_budget_calibration.json",
    "results/v3_3_5c/tables/semantic_evaluator_validation.json",
    "llm/src/cantor_guard_v335c/p0_normalized_dose.py",
    "llm/src/cantor_guard_v335c/p0_behavioral_boundary.py",
    "llm/src/cantor_guard_v335c/affine_coordinate.py",
    "llm/src/cantor_guard_v335c/cantor_geometry.py",
    "llm/src/cantor_guard_v335c/p0_cantor_controller.py",
    "llm/src/cantor_guard_v335c/p0_attack_generation.py",
    "llm/src/cantor_guard_v335c/semantic_eval.py",
]


def sha256(path: str) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def verify_freeze(manifest: dict) -> None:
    if manifest.get("status") != "PRE_ANALYSIS_FROZEN":
        raise RuntimeError(
            "final-stage execution requires status=PRE_ANALYSIS_FROZEN; "
            f"observed {manifest.get('status', 'MISSING')}"
        )
    for path, expected in manifest["files"].items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"frozen file changed: {path}")


def main() -> None:
    boundary = read_json(RESULTS / "tables/p0_behavioral_boundary.json")
    window = read_json(RESULTS / "tables/p0_window.json")
    budget = read_json(RESULTS / "tables/controller_budget_calibration.json")
    evaluator = read_json(CONFIG / "evaluator.json")
    controller = read_json(CONFIG / "controller.json")
    attack = read_json(CONFIG / "attack_grid.json")
    blockers = []
    if boundary["verdict"] not in {"B1_P0_BEHAVIORAL_BOUNDARY_IDENTIFIED", "B2_NONPARAMETRIC_BOUNDARY_ONLY"}:
        blockers.append("behavioral boundary")
    if boundary.get("tau_beh_P0") is None:
        blockers.append("tau")
    if window.get("W") is None:
        blockers.append("W")
    if budget.get("status") != "CONTROLLER_BUDGET_FROZEN" or not controller.get("eta_by_rho"):
        blockers.append("controller budget")
    if not attack.get("common_absolute_epsilon_grid"):
        blockers.append("common attack grid")
    final_paths = [RESULTS / "raw/final_p0_cantor.csv", RESULTS / "private/final_p0_cantor_completions.csv"]
    if any(path.exists() for path in final_paths):
        blockers.append("D_final was already touched")
    if blockers:
        raise SystemExit("STOP: cannot freeze: " + ", ".join(blockers))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "status": "PRE_ANALYSIS_FROZEN",
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_git_sha_before_freeze_commit": head,
        "files": {path: sha256(path) for path in FILES},
        "choices": {
            "model": read_json(CONFIG / "behavioral_protocol.json")["model"],
            "layer": read_json(CONFIG / "behavioral_protocol.json")["layer"],
            "direction_sha256": read_json(CONFIG / "behavioral_protocol.json")["direction_sha256"],
            "safe_sign": read_json(CONFIG / "behavioral_protocol.json")["safe_sign"],
            "semantic_evaluator": evaluator["selected"],
            "semantic_evaluator_status": evaluator["status"],
            "tau_beh_P0": boundary["tau_beh_P0"],
            "tau_method": boundary["tau_method"],
            "W": window["W"],
            "rho_family": controller["rho_family"],
            "cell_actions": controller["leaf_actions"],
            "guard_action": controller["guard_action"],
            "outside_action": controller["outside_window_action"],
            "eta_by_rho": controller["eta_by_rho"],
            "q_target": controller["q_target"],
            "budget_tolerance": controller["budget_tolerance_relative"],
            "attack_grid": attack["common_absolute_epsilon_grid"],
            "generation": {
                "max_new_tokens": read_json(CONFIG / "behavioral_protocol.json")["max_new_tokens"],
                "deterministic": True,
            },
            "semantic_endpoint": "harmful_compliance" if evaluator["status"] == "S1_VALID" else "refusal_proxy_only",
            "sesoi": controller["semantic_auc_sesoi"],
            "bootstrap_replicates": read_json(CONFIG / "behavioral_protocol.json")["bootstrap_replicates"],
            "bootstrap_seed": read_json(CONFIG / "behavioral_protocol.json")["bootstrap_seed"],
            "classifier": "scripts/v3_3_5c/final_claim_check_v335c.py",
        },
        "D_final_touched_before_freeze": False,
        "requires_commit_before_final_generation": True,
    }
    write_json(CONFIG / "PRE_ANALYSIS_FREEZE.json", manifest)
    print("PRE_ANALYSIS_FROZEN")
    print("Commit this manifest and all frozen files before running D_final.")


if __name__ == "__main__":
    main()
