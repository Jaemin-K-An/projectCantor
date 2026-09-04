"""Phase 0: reproduce and freeze the V3.5.0 domain-mismatch diagnosis."""
from __future__ import annotations

import math
import pathlib
import subprocess

import numpy as np
import pandas as pd

from _common import (ACTUATOR_SHA, BASE_COMMIT, Q_CAP, Q_TARGET, RESULTS,
                     ROOT, SENSOR_SHA, read_json, sha256, write_json)

PRESERVED = ["configs/v3_5_0", "results/v3_5_0", "docs/v3_5_0",
             "scripts/v3_5_0", "llm/src/cantor_guard_v350"]


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def main():
    trees = {}
    for path in PRESERVED:
        base = git("rev-parse", f"{BASE_COMMIT}:{path}")
        head = git("rev-parse", f"HEAD:{path}")
        trees[path] = {"base_tree": base, "head_tree": head, "identical": base == head}
    risk = pd.read_csv(ROOT / "results/v3_5_0/raw/risk_window_calibration.csv")
    x = risk.x_risk.to_numpy(float); d = risk.d_clean.to_numpy(float)
    k_all = math.ceil((len(x) + 1) * .95); W_all = float(np.sort(x)[k_all - 1])
    xr = -d[d < 0]; k_risk_old = math.ceil((len(xr) + 1) * .95)
    W_risk_old = float(np.sort(xr)[min(k_risk_old, len(xr)) - 1])
    states = pd.read_csv(ROOT / "results/v3_5_0/raw/attacked_budget_states.csv")
    deployed = states[states.epsilon > 0]
    risk_prevalence = float(np.mean(deployed.d_attacked < 0))
    unconditional_cap_upper = Q_CAP * math.sqrt(risk_prevalence)
    budget = read_json(ROOT / "results/v3_5_0/tables/budget_calibration.json")
    freeze = read_json(ROOT / "configs/v3_5_0/PRE_ANALYSIS_FREEZE.json")
    final_paths = [ROOT / "results/v3_5_0/raw/final_D_final_v350_harmful.csv",
                   ROOT / "results/v3_5_0/raw/utility_D_final_v350_benign.csv",
                   ROOT / "results/v3_5_0/private/final_D_final_v350_harmful_completions.csv",
                   ROOT / "results/v3_5_0/private/utility_D_final_v350_benign_completions.csv"]
    final_untouched = freeze.get("D_final_v350_touched") is False and not any(p.exists() for p in final_paths)
    sensor = sha256(ROOT / "results/v3_4_0/cache/sensor_w.npy")
    actuator = sha256(ROOT / "results/v3_3_5a/cache/v_p0.npy")
    passed = (all(v["identical"] for v in trees.values()) and sensor == SENSOR_SHA and
              actuator == ACTUATOR_SHA and budget["verdict"] == "BUD2_MISMATCH" and final_untouched)
    payload = {
        "base_commit": BASE_COMMIT, "historical_trees": trees,
        "sensor_sha256": sensor, "actuator_sha256": actuator,
        "sensor_hash_match": sensor == SENSOR_SHA, "actuator_hash_match": actuator == ACTUATOR_SHA,
        "v350_unconditional_risk_calibration": {
            "n": len(x), "zero_risk_fraction": float(np.mean(x == 0)),
            "risk_prevalence": float(np.mean(d < 0)), "order_index": k_all,
            "W_R_recomputed": W_all, "conditional_coverage_at_old_W_R": float(np.mean(xr <= W_all)),
            "risk_states_available": len(xr), "counterfactual_conditional_order_index": k_risk_old,
            "counterfactual_conditional_W_R_on_same_small_sample": W_risk_old,
        },
        "v350_unconditional_budget": {
            "risk_eligible_prevalence": risk_prevalence,
            "safe_side_fraction": 1 - risk_prevalence,
            "q_cap_times_sqrt_risk_prevalence_upper_bound": unconditional_cap_upper,
            "target": Q_TARGET,
            "per_arm_maximum": {k: v["maximum_attainable_q_rms"] for k, v in budget["per_arm"].items()},
            "verdict": budget["verdict"],
        },
        "diagnosis": "one-sided sparse-controller domain mismatch: unconditional scale and budget were calibrated over safe zero-mass plus risk states",
        "geometry_failure": False,
        "v350_final_untouched": final_untouched,
        "final_paths_checked": [str(p.relative_to(ROOT)) for p in final_paths],
        "passed": passed, "verdict": "AUDIT1_V350_DOMAIN_MISMATCH_REPRODUCED" if passed else "AUDIT2_FAIL",
    }
    write_json(RESULTS / "tables/v350_failure_audit.json", payload)
    print(payload["verdict"])
    print(payload["v350_unconditional_risk_calibration"])
    print(payload["v350_unconditional_budget"])
    if not passed: raise SystemExit("V3.5.0 audit hard stop")


if __name__ == "__main__": main()
