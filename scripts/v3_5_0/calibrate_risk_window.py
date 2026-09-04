"""Phases 2-4: collect clean calibration states and freeze conformal W_R."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from cantor_guard.io import seed_everything  # noqa: E402
from cantor_guard.models import load_model  # noqa: E402
from cantor_guard_v350.conformal_window import calibrate_upper_window  # noqa: E402
from cantor_guard_v350.risk_coordinate import risk_magnitude  # noqa: E402
from _common import ALPHA, CONFIG, RESULTS, ensure_final_absent, frozen_sensor, read_json, write_json  # noqa: E402
from _model import clean_residuals  # noqa: E402


def main() -> None:
    ensure_final_absent()
    output = RESULTS / "tables/risk_window_calibration.json"
    if output.exists(): raise SystemExit("risk window already calibrated; refuse to retune")
    seed_everything(20260904)
    prompts = pd.read_csv(RESULTS / "cache/D_risk_window_cal.csv")
    if len(prompts) < 300: raise SystemExit("D_risk_window_cal requires n>=300")
    bundle = load_model("qwen2.5-0.5b-instruct")
    H = clean_residuals(bundle, prompts.prompt.tolist(), layer=14, batch_size=8)
    np.save(RESULTS / "cache/h_D_risk_window_cal.npy", H)
    sensor = frozen_sensor()
    d = np.asarray(sensor.distance(H), dtype=float)
    x = np.asarray(risk_magnitude(d), dtype=float)
    cal = calibrate_upper_window(x, ALPHA)
    rows = pd.DataFrame({"pid": prompts.pid, "d_clean": d, "x_risk": x,
                         "inside_calibrated_window": x <= cal.W_R})
    rows.to_csv(RESULTS / "raw/risk_window_calibration.csv", index=False)
    payload = {
        "split": "D_risk_window_cal", "n": cal.n, "alpha": cal.alpha,
        "target_coverage": 1 - cal.alpha, "order_index_one_based": cal.order_index_one_based,
        "order_index_rule": "ceil((n+1)*(1-alpha))", "W_R": cal.W_R,
        "empirical_coverage": cal.empirical_coverage, "zero_risk_fraction": float(np.mean(x == 0)),
        "d_min": float(d.min()), "d_max": float(d.max()), "x_max": float(x.max()),
        "uses_labels": False, "uses_generated_endpoints": False, "final_data_used": False,
        "retuning_allowed": False, "verdict": "RISK_WINDOW_CALIBRATED",
    }
    write_json(output, payload)
    cfg = read_json(CONFIG / "risk_coordinate.json"); cfg["W_R"] = cal.W_R
    cfg["order_index_one_based"] = cal.order_index_one_based
    cfg["calibration_empirical_coverage"] = cal.empirical_coverage
    write_json(CONFIG / "risk_coordinate.json", cfg)
    print(f"n={cal.n} alpha={cal.alpha} k={cal.order_index_one_based} W_R={cal.W_R:.9f} coverage={cal.empirical_coverage:.6f}")


if __name__ == "__main__": main()
