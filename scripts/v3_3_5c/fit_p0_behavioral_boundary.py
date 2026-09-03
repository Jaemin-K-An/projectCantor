"""Fit and gate DEV/CONFIRM P0 boundaries using realised z_after."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "llm/src")
from cantor_guard_v335c.p0_behavioral_boundary import fit_behavioral_boundary, fit_logistic  # noqa: E402

from _common import CONFIG, RESULTS, behavioral_protocol, read_json, write_json


def _endpoint(frame: pd.DataFrame, evaluator: dict) -> tuple[np.ndarray, str]:
    if evaluator["status"] == "S1_VALID":
        semantic = frame.semantic_harmful_compliance
        if semantic.isna().any():
            raise ValueError("semantic evaluator is marked valid but semantic labels are missing")
        return (1 - semantic.astype(int).to_numpy()), "semantic_refusal"
    return frame.refusal_proxy.astype(int).to_numpy(), "refusal_proxy"


def _serializable(fit) -> dict:
    result = asdict(fit)
    return result


def main() -> None:
    protocol = behavioral_protocol()
    evaluator = read_json(CONFIG / "evaluator.json")
    freeze = read_json(RESULTS / "tables/p0_dose_grid_freeze.json")
    if freeze["status"] != "READY_FOR_CONFIRM":
        raise SystemExit("STOP: dose-grid gate failed")
    grid = freeze["confirm_u_grid"]
    dev = pd.read_csv(RESULTS / "raw/symmetric_D_beh_P0_dev_335c.csv")
    confirm = pd.read_csv(RESULTS / "raw/symmetric_D_beh_P0_confirm_335c.csv")
    dev = dev[dev.u.isin(grid)].copy()
    confirm = confirm[confirm.u.isin(grid)].copy()
    y_dev, endpoint = _endpoint(dev, evaluator)
    y_confirm, _ = _endpoint(confirm, evaluator)
    _, slope_dev = fit_logistic(dev.z_after, y_dev)
    _, slope_confirm = fit_logistic(confirm.z_after, y_confirm)
    safe_sign = int(protocol["safe_sign"])
    directions_agree = bool(np.sign(slope_dev) == np.sign(slope_confirm) == safe_sign)
    common = {
        "safe_sign": safe_sign,
        "confirm_direction_agrees": directions_agree,
        "n_boot": int(protocol["bootstrap_replicates"]),
        "seed": int(protocol["bootstrap_seed"]),
        "coherence_gate": float(protocol["coherence_gate"]),
        "degeneration_rate_gate": float(protocol["degeneration_rate_gate"]),
        "crossing_agreement_sigma": float(protocol["logistic_isotonic_agreement_sigma"]),
    }
    dev_fit = fit_behavioral_boundary(
        z_after=dev.z_after, z_clean=dev.z_clean, outcome=y_dev, pid=dev.pid, u=dev.u,
        coherence=dev.coherence, degeneration=dev.degeneration, **common,
    )
    confirm_fit = fit_behavioral_boundary(
        z_after=confirm.z_after, z_clean=confirm.z_clean, outcome=y_confirm, pid=confirm.pid, u=confirm.u,
        coherence=confirm.coherence, degeneration=confirm.degeneration, **common,
    )
    write_json(RESULTS / "tables/p0_boundary_dev.json", {"endpoint": endpoint, **_serializable(dev_fit)})
    write_json(RESULTS / "tables/p0_boundary_confirm.json", {"endpoint": endpoint, **_serializable(confirm_fit)})
    if confirm_fit.verdict == "B1_P0_BEHAVIORAL_BOUNDARY_IDENTIFIED":
        tau, method = confirm_fit.tau_logistic, "confirm_logistic"
    elif confirm_fit.verdict == "B2_NONPARAMETRIC_BOUNDARY_ONLY" and protocol["nonparametric_tau_admissible"]:
        tau, method = confirm_fit.tau_isotonic, "confirm_isotonic_preregistered_fallback"
    else:
        tau, method = None, None
    final = {
        "verdict": confirm_fit.verdict,
        "endpoint": endpoint,
        "semantic_evaluator_status": evaluator["status"],
        "tau_beh_P0": tau,
        "tau_method": method,
        "dev_verdict": dev_fit.verdict,
        "dev_confirm_crossing_direction_agrees": directions_agree,
        "confirm": _serializable(confirm_fit),
    }
    write_json(RESULTS / "tables/p0_behavioral_boundary.json", final)
    print(f"DEV={dev_fit.verdict} CONFIRM={confirm_fit.verdict}")
    print(f"endpoint={endpoint} tau={tau} method={method}")
    if tau is None:
        raise SystemExit("STOP: fresh P0 behavioural boundary is unidentifiable")


if __name__ == "__main__":
    main()
