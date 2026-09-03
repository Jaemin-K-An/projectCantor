"""Phases 8-9 -- coupling, controllability gate, and the fixed affine window.

W is calibrated on an independent split and is the SAME for every rho, so the
certificate ordering epsilon_h(rho) = 2W M_3(rho) is decided by geometry alone.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "llm/src"))
from cantor_guard_v340.actuator import Actuator, coupling  # noqa: E402
from cantor_guard_v340.cantor_geometry import epsilon_h, epsilon_h_cantor  # noqa: E402
from cantor_guard_v340.sensor_distance import SensorHyperplane  # noqa: E402

from _common import CONFIG, RESULTS, read_json, write_json  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> None:
    ctrl = read_json(CONFIG / "controller.json")
    act_cfg = read_json(CONFIG / "actuator.json")
    fit = read_json(RESULTS / "tables" / "sensor_confirm.json")
    sensor = SensorHyperplane(np.load(RESULTS / "cache" / "sensor_w.npy"), float(fit["b"]))
    actuator = Actuator(np.load(ROOT / act_cfg["direction_file"]).astype(float).reshape(-1),
                        int(act_cfg["safe_sign"]))
    coup = coupling(sensor, actuator)

    H_win = np.load(RESULTS / "cache" / "h_D_window_cal.npy")
    d_win = np.atleast_1d(sensor.distance(H_win))
    W = float(1.05 * np.quantile(np.abs(d_win), 0.99))
    if W <= 0:
        raise SystemExit("calibrated W must be positive")

    H_bud = np.load(RESULTS / "cache" / "h_D_controller_budget.npy")
    d_bud = np.atleast_1d(sensor.distance(H_bud))
    norms = np.linalg.norm(H_bud, axis=1)
    q_cap = float(ctrl["CONTROLLABILITY_GATE"]["q_cap"])
    achievable = q_cap * norms * abs(coup["kappa"])
    eps_c = epsilon_h_cantor(W)
    fraction = float(np.mean(achievable >= eps_c))
    required = float(ctrl["CONTROLLABILITY_GATE"]["fraction_required"])
    controllable = bool(fraction >= required)

    coverage = {}
    for split in ["D_window_cal", "D_sensor_confirm", "D_controller_budget", "D_attack_dev"]:
        H = np.load(RESULTS / "cache" / f"h_{split}.npy")
        d = np.atleast_1d(sensor.distance(H))
        coverage[split] = {
            "n": int(d.size), "mean_d": float(d.mean()), "sd_d": float(d.std(ddof=1)),
            "q01": float(np.quantile(d, 0.01)), "q99": float(np.quantile(d, 0.99)),
            "inside_window_fraction": float(np.mean(np.abs(d) <= W)),
        }

    certificates = {}
    for rho in ctrl["rho_family"]:
        key = "1/3" if abs(rho - 1 / 3) < 1e-12 else f"{rho:.2f}"
        certificates[key] = {"rho": float(rho), "M3": float(rho**2 * (1 - 2 * rho)),
                             "epsilon_h": float(epsilon_h(rho, W))}
    best = max(certificates, key=lambda k: certificates[k]["epsilon_h"])

    payload = {
        "coupling": coup,
        "W": W, "window_rule": "1.05 * Q_0.99(|d(h_clean)|) on D_window_cal",
        "window_split_n": int(d_win.size), "same_W_every_rho": True, "clipping": False,
        "epsilon_cantor": eps_c,
        "certificates": certificates,
        "argmax_certificate": best,
        "controllability": {
            "q_cap": q_cap, "kappa": coup["kappa"],
            "mean_h_norm": float(norms.mean()),
            "mean_achievable_delta_d": float(achievable.mean()),
            "epsilon_cantor": eps_c,
            "fraction_meeting_criterion": fraction,
            "fraction_required": required,
            "passed": controllable,
            "verdict": "COUP1_CONTROLLABLE" if controllable else "COUP2_WEAK_SENSOR_ACTUATOR_COUPLING",
        },
        "coverage": coverage,
    }
    write_json(RESULTS / "tables" / "sensor_window_and_coupling.json", payload)
    print(f"kappa = {coup['kappa']:+.4f}   angle(w,v) = {coup['angle_w_v_deg']:.1f} deg")
    print(f"W = {W:.4f}   epsilon_Cantor = 2W/27 = {eps_c:.4f}")
    print("\ncertificates:")
    for key, row in certificates.items():
        print(f"  rho={key:<5} M3={row['M3']:.6f}  epsilon_h={row['epsilon_h']:.4f}"
              f"{'   <- max' if key == best else ''}")
    print(f"\ncontrollability: mean achievable |dd| at q<={q_cap} = {achievable.mean():.4f} "
          f"vs epsilon_C = {eps_c:.4f}")
    print(f"  fraction meeting criterion = {fraction:.3f} (need {required})"
          f"  -> {payload['controllability']['verdict']}")
    print("\nwindow coverage:")
    for split, row in coverage.items():
        print(f"  {split:<22} inside={row['inside_window_fraction']:.3f} "
              f"mean_d={row['mean_d']:+.3f} sd={row['sd_d']:.3f}")


if __name__ == "__main__":
    main()
