"""Calibrate W_R from the first 200 clean d<0 states in fixed order."""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from cantor_guard.io import seed_everything  # noqa: E402
from cantor_guard.models import load_model  # noqa: E402
from cantor_guard_v351.conformal_window import calibrate_conditional_risk_window  # noqa: E402
from _common import (ALPHA, CONFIG, K_RISK, RESULTS, ensure_final_outputs_absent,
                     frozen_sensor, read_json, sha256, write_json)  # noqa: E402
from _model import clean_residuals  # noqa: E402

MODEL_SEED = 20260905
BOOTSTRAP_SEED = 35101
N_BOOT = 20_000
SCAN_CHUNK = 128


def list_hash(values) -> str:
    return hashlib.sha256(json.dumps(list(values), separators=(",", ":")).encode()).hexdigest()


def quantiles(values) -> dict:
    arr = np.asarray(values, dtype=float)
    return {str(p): float(np.quantile(arr, p)) for p in (0, .05, .25, .5, .75, .9, .95, 1)}


def main() -> None:
    ensure_final_outputs_absent()
    output = RESULTS / "tables/risk_conditional_window.json"
    if output.exists():
        raise SystemExit("risk-conditional window already calibrated; refuse to retune")
    leakage = read_json(RESULTS / "tables/historical_leakage_audit.json")
    if leakage.get("verdict") != "LEAK1_PASS":
        raise SystemExit("historical leakage gate not passed")
    pool_path = RESULTS / "cache/D_risk_cal_candidate_order_v351.csv"
    pool = pd.read_csv(pool_path).sort_values("candidate_order", kind="stable").reset_index(drop=True)
    if pool.candidate_order.tolist() != list(range(len(pool))):
        raise SystemExit("candidate order is not a contiguous frozen ordering")

    seed_everything(MODEL_SEED)
    bundle = load_model("qwen2.5-0.5b-instruct")
    sensor = frozen_sensor()
    scanned_frames, scanned_h = [], []
    risk_seen = 0
    for start in range(0, len(pool), SCAN_CHUNK):
        chunk = pool.iloc[start:start + SCAN_CHUNK].copy()
        H = clean_residuals(bundle, chunk.prompt.astype(str).tolist(), layer=14, batch_size=8)
        d = np.asarray(sensor.distance(H), dtype=float)
        chunk["d_clean"] = d
        chunk["x_risk"] = np.maximum(0.0, -d)
        chunk["risk_eligible"] = d < 0
        scanned_frames.append(chunk)
        scanned_h.append(H)
        risk_seen += int(np.sum(d < 0))
        print(f"scanned={start + len(chunk)}/{len(pool)} risk_seen={risk_seen}/{K_RISK}", flush=True)
        if risk_seen >= K_RISK:
            break

    scanned = pd.concat(scanned_frames, ignore_index=True)
    H_scanned = np.concatenate(scanned_h, axis=0)
    risk_positions = np.flatnonzero(scanned.risk_eligible.to_numpy(bool))
    if len(risk_positions) < K_RISK:
        write_json(output, {
            "candidate_pool_size": len(pool), "n_scanned": len(scanned),
            "n_risk": len(risk_positions), "K_RISK": K_RISK,
            "uses_labels": False, "uses_generated_endpoints": False,
            "verdict": "RISK_CAL_INSUFFICIENT",
        })
        raise SystemExit(f"RISK_CAL_INSUFFICIENT: {len(risk_positions)} < {K_RISK}")

    selected_positions = risk_positions[:K_RISK]
    selected = scanned.iloc[selected_positions].copy()
    selected["risk_calibration_selected"] = True
    d_selected = selected.d_clean.to_numpy(float)
    x_selected = -d_selected
    calibration = calibrate_conditional_risk_window(
        scanned.d_clean.to_numpy(float), n_risk=K_RISK, alpha=ALPHA)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, K_RISK, size=(N_BOOT, K_RISK))
    bootstrap_samples = x_selected[indices]
    bootstrap_thresholds = np.partition(
        bootstrap_samples, calibration.order_index_one_based - 1, axis=1
    )[:, calibration.order_index_one_based - 1]

    scanned["risk_calibration_selected"] = False
    scanned.loc[selected_positions, "risk_calibration_selected"] = True
    scanned.to_csv(RESULTS / "raw/risk_conditional_window_scanned.csv", index=False)
    selected.to_csv(RESULTS / "cache/D_risk_cal_v351.csv", index=False)
    np.save(RESULTS / "cache/h_D_risk_cal_scanned_v351.npy", H_scanned)
    np.save(RESULTS / "cache/h_D_risk_cal_v351.npy", H_scanned[selected_positions])

    d_scanned = scanned.d_clean.to_numpy(float)
    x_scanned = scanned.x_risk.to_numpy(float)
    scanned_ids = scanned.pid.astype(str).tolist()
    selected_ids = selected.pid.astype(str).tolist()
    payload = {
        "split": "D_risk_cal_v351",
        "selection": "first 200 d<0 clean states in frozen candidate order",
        "candidate_pool_size": len(pool),
        "n_scanned": len(scanned),
        "scanned_d_lt_0_prevalence": float(np.mean(d_scanned < 0)),
        "n_risk": K_RISK,
        "K_RISK": K_RISK,
        "alpha": ALPHA,
        "target_conditional_coverage": 1 - ALPHA,
        "order_index_one_based": calibration.order_index_one_based,
        "order_index_rule": "ceil((n+1)*(1-alpha))",
        "W_R": calibration.W_R,
        "conditional_empirical_coverage": calibration.empirical_coverage,
        "unconditional_empirical_coverage_over_scanned_states": float(np.mean(x_scanned <= calibration.W_R)),
        "selected_x_quantiles": quantiles(x_selected),
        "selected_d_quantiles": quantiles(d_selected),
        "scanned_d_quantiles": quantiles(d_scanned),
        "sampling_uncertainty_exploratory": {
            "method": "nonparametric bootstrap of the frozen conformal order statistic",
            "n_boot": N_BOOT,
            "seed": BOOTSTRAP_SEED,
            "percentile_95_interval": [float(np.quantile(bootstrap_thresholds, .025)),
                                       float(np.quantile(bootstrap_thresholds, .975))],
            "used_to_adjust_W_R": False,
        },
        "data_hashes": {
            "candidate_pool_csv_sha256": sha256(pool_path),
            "scanned_pid_order_sha256": list_hash(scanned_ids),
            "selected_pid_order_sha256": list_hash(selected_ids),
            "scanned_residuals_sha256": sha256(RESULTS / "cache/h_D_risk_cal_scanned_v351.npy"),
            "selected_residuals_sha256": sha256(RESULTS / "cache/h_D_risk_cal_v351.npy"),
        },
        "model": bundle.provenance(),
        "uses_labels": False,
        "uses_generated_endpoints": False,
        "final_data_used": False,
        "retuning_allowed": False,
        "verdict": "RISK1_CONDITIONAL_WINDOW_CALIBRATED",
    }
    write_json(output, payload)

    risk_cfg = read_json(CONFIG / "risk_coordinate.json")
    risk_cfg.update({
        "W_R": calibration.W_R,
        "order_index_one_based": calibration.order_index_one_based,
        "conditional_empirical_coverage": calibration.empirical_coverage,
        "calibration_artifact": str(output.relative_to(ROOT)),
    })
    write_json(CONFIG / "risk_coordinate.json", risk_cfg)
    splits = read_json(CONFIG / "splits.json")
    splits["risk_cal_scanned_pids"] = scanned_ids
    splits["risk_cal_scanned_pid_order_sha256"] = list_hash(scanned_ids)
    splits["risk_cal_selected_pids"] = selected_ids
    splits["risk_cal_selected_pid_order_sha256"] = list_hash(selected_ids)
    splits["risk_cal_scanned_n"] = len(scanned)
    write_json(CONFIG / "splits.json", splits)
    print(f"RISK1_CONDITIONAL_WINDOW_CALIBRATED n={K_RISK} scanned={len(scanned)} "
          f"W_R={calibration.W_R:.12f} coverage={calibration.empirical_coverage:.6f}")


if __name__ == "__main__":
    main()
