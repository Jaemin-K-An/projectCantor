"""Validate the depth-3 direct terminal risk-policy-switch certificate."""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from cantor_guard_v340.attack import attack_w  # noqa: E402
from cantor_guard_v351.one_sided_cantor import (classify, epsilon_r,
                                                epsilon_r_cantor, margin_derivative_m3,
                                                margin_m3, unique_grid_max)  # noqa: E402
from _common import CONFIG, RESULTS, RHOS, frozen_sensor, read_json, rho_key, write_json  # noqa: E402


def main() -> None:
    H = np.load(RESULTS / "cache/h_D_risk_cal_v351.npy")
    sensor = frozen_sensor()
    W_R = float(read_json(CONFIG / "risk_coordinate.json")["W_R"])
    d0 = np.asarray(sensor.distance(H), dtype=float)
    x0 = np.maximum(0.0, -d0)
    r0 = x0 / W_R
    per_rho, total = {}, 0
    for rho in RHOS:
        cert = epsilon_r(rho, W_R)
        base = [classify(float(r), rho, 3) if 0 < x <= W_R else None for r, x in zip(r0, x0)]
        tested = violations = 0
        for eps in np.linspace(cert / 100, cert * .999, 64):
            for sign in (-1, 1):
                delta = attack_w(sensor, eps, sign=sign)
                d1 = np.asarray(sensor.distance(H + delta[None, :]), dtype=float)
                x1 = np.maximum(0.0, -d1)
                r1 = x1 / W_R
                for cell0, x_after, r_after in zip(base, x1, r1):
                    if cell0 is None or cell0.kind != "leaf":
                        continue
                    tested += 1
                    if x_after <= 0 or x_after > W_R:
                        continue
                    cell1 = classify(float(r_after), rho, 3)
                    if cell1.kind == "leaf" and cell1.index != cell0.index:
                        violations += 1
        per_rho[rho_key(rho)] = {
            "rho": rho, "epsilon_R": cert,
            "tests_below_certificate": tested, "violations": violations,
        }
        total += violations

    dense = np.linspace(.000001, .499999, 500_000)
    dense_values = dense ** 2 * (1 - 2 * dense)
    dense_max = float(dense[np.argmax(dense_values)])
    family_max = unique_grid_max(RHOS)
    derivative_left = margin_derivative_m3(1/3 - 1e-6)
    derivative_right = margin_derivative_m3(1/3 + 1e-6)
    unique = (family_max == 1/3 and abs(dense_max - 1/3) < 2e-6 and
              derivative_left > 0 > derivative_right and
              np.isclose(margin_m3(1/3), 1/27))
    verdict = "GEO1_ONE_SIDED_CANTOR_CERTIFICATE_VALID" if total == 0 and unique else "GEO2_IMPLEMENTATION_FAILURE"
    payload = {
        "W_R": W_R,
        "formula": "epsilon_R(rho)=W_R*rho^2*(1-2*rho)",
        "analytic_derivative": "d/drho [rho^2(1-2rho)] = 2rho(1-3rho)",
        "analytic_unique_maximum_on_0<rho<1/2": 1/3,
        "rho_family_unique_maximum": family_max,
        "dense_grid_points": len(dense),
        "dense_grid_argmax": dense_max,
        "middle_third_factor": margin_m3(1/3),
        "epsilon_R_C": epsilon_r_cantor(W_R),
        "n_states": len(H),
        "per_rho": per_rho,
        "total_violations": total,
        "unique_middle_third_maximum_verified": bool(unique),
        "scope": "residual-L2 direct terminal risk-policy switch; not semantic safety",
        "verdict": verdict,
    }
    write_json(RESULTS / "tables/certificate_validation.json", payload)
    print(verdict, "violations", total, "epsilon_R_C", payload["epsilon_R_C"])
    if verdict != "GEO1_ONE_SIDED_CANTOR_CERTIFICATE_VALID":
        raise SystemExit("Cantor certificate implementation failure")


if __name__ == "__main__":
    main()
