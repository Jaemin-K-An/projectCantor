"""Calibrate one affine P0 operating window on the independent window split."""
from __future__ import annotations

import pandas as pd

import sys
sys.path.insert(0, "llm/src")
from cantor_guard.models import load_model  # noqa: E402
from cantor_guard_v335a.p0_residual import last_valid_prompt_residuals  # noqa: E402
from cantor_guard_v335c.affine_coordinate import AffineCoordinate, calibrate_window  # noqa: E402

from _common import CONFIG, RESULTS, behavioral_protocol, load_direction, read_json, write_json


def main() -> None:
    boundary = read_json(RESULTS / "tables/p0_behavioral_boundary.json")
    tau = boundary.get("tau_beh_P0")
    if tau is None:
        raise SystemExit("STOP: no admissible behavioural tau")
    protocol = behavioral_protocol()
    prompts = pd.read_csv(RESULTS / "cache/D_window_P0_335c.csv")
    bundle = load_model(protocol["model"])
    direction = load_direction(protocol)
    residuals = last_valid_prompt_residuals(
        bundle, prompts.prompt.tolist(), int(protocol["layer"]), batch_size=8,
    )
    z = residuals @ direction
    W = calibrate_window(z, float(tau), quantile=0.99, padding=1.05)
    coordinate = AffineCoordinate(float(tau), W, -int(protocol["safe_sign"]))
    table = pd.DataFrame({"pid": prompts.pid, "z_clean": z, "inside_window": coordinate.inside(z).astype(int)})
    (RESULTS / "raw").mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "raw/window_calibration.csv", index=False)
    report = {
        "tau_beh_P0": float(tau), "W": W, "rule": "1.05 * Q_0.99(|z_clean-tau|)",
        "n": len(z), "calibration_coverage": coordinate.coverage(z),
        "same_tau_every_rho": True, "same_W_every_rho": True,
        "outside_policy": "OUTSIDE_WINDOW_CONSERVATIVE_FALLBACK", "clipping": False,
    }
    write_json(RESULTS / "tables/p0_window.json", report)
    controller = read_json(CONFIG / "controller.json")
    controller["tau"] = float(tau)
    controller["W"] = W
    write_json(CONFIG / "controller.json", controller)
    print(f"tau={tau:.8g} W={W:.8g} calibration coverage={coordinate.coverage(z):.3f}")


if __name__ == "__main__":
    main()
