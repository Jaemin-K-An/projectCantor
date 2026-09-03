"""Evaluate frozen-W applicability on the external sensor-check split.

This gate is deliberately separate from sensor discrimination: a sensor may
rank refusal states well while too many deployment states fall outside the
fixed affine window.  W is read from the immutable V3.4.0 freeze and is never
estimated here.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from _common import CONFIG, RESULTS, frozen_sensor, read_json, write_json  # noqa: E402


def main() -> None:
    cfg = read_json(CONFIG / "controller.json")
    gate = cfg["EXTERNAL_WINDOW_GATE"]
    historical = read_json(ROOT / "configs/v3_4_0/PRE_ANALYSIS_FREEZE.json")
    W = float(historical["geometry"]["W"])
    if W != float(gate["W"]) or W != float(cfg["frozen_from_v340"]["W"]):
        raise RuntimeError("V3.4.0R W differs from the immutable V3.4.0 value")
    H = np.load(RESULTS / "cache/h_D_sensor_transfer_r.npy")
    d = np.asarray(frozen_sensor().distance(H), dtype=float)
    inside = np.abs(d) <= W
    coverage = float(inside.mean())
    passed = coverage >= float(gate["coverage_min"])
    payload = {
        "split": gate["split"],
        "n": int(d.size),
        "W": W,
        "W_source": "configs/v3_4_0/PRE_ANALYSIS_FREEZE.json",
        "W_recalibrated": False,
        "inside_count": int(inside.sum()),
        "outside_count": int((~inside).sum()),
        "coverage": coverage,
        "coverage_min": float(gate["coverage_min"]),
        "passed": passed,
        "verdict": "ST1_PASS" if passed else "ST3_WINDOW_SHIFT",
        "stop_controller_final_testing": not passed,
        "distance_quantiles": {
            str(q): float(np.quantile(d, q)) for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
        },
    }
    write_json(RESULTS / "tables/external_window.json", payload)
    print(f"fixed W={W:.16f} coverage={coverage:.4f} ({inside.sum()}/{d.size})")
    print(payload["verdict"])
    if not passed:
        print("STOP: fixed-W applicability gate failed; do not freeze or run controller finals")


if __name__ == "__main__":
    main()
