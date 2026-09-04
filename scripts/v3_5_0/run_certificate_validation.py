"""Phase 11: validate the exact one-sided terminal-policy certificate."""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from cantor_guard_v340.attack import attack_w  # noqa: E402
from cantor_guard_v350.one_sided_cantor import classify, epsilon_r  # noqa: E402
from cantor_guard_v350.risk_cantor_controller import RiskCantorController  # noqa: E402
from _common import CONFIG, Q_CAP, RESULTS, frozen_actuator, frozen_sensor, read_json, rho_key, write_json  # noqa: E402


def main() -> None:
    H = np.load(RESULTS / "cache/h_D_budget_v350.npy")
    cfg = read_json(CONFIG / "controller.json")
    sensor, actuator = frozen_sensor(), frozen_actuator()
    W_R = float(read_json(RESULTS / "tables/risk_window_calibration.json")["W_R"])
    per_rho, total = {}, 0
    d0 = np.asarray(sensor.distance(H)); x0 = np.maximum(0, -d0); r0 = x0 / W_R
    for rho in cfg["rho_family"]:
        ctrl = RiskCantorController(sensor=sensor, actuator=actuator, W_R=W_R, rho=rho, eta=0, q_cap=Q_CAP)
        cert = epsilon_r(rho, W_R); tested = violations = 0
        base = [classify(float(r), rho, 3) if x <= W_R else None for r, x in zip(r0, x0)]
        # Dense deterministic below-certificate radii and both sensor-normal directions.
        radii = np.linspace(cert / 100, cert * .999, 64)
        for eps in radii:
            for sign in (-1, 1):
                d1 = np.asarray(sensor.distance(H + attack_w(sensor, eps, sign=sign)[None, :]))
                x1 = np.maximum(0, -d1); r1 = x1 / W_R
                for c0, xi, ri in zip(base, x1, r1):
                    if c0 is None or c0.kind != "leaf": continue
                    tested += 1
                    if xi > W_R or ri == 0: continue
                    c1 = classify(float(ri), rho, 3)
                    if c1.kind == "leaf" and c1.index != c0.index: violations += 1
        per_rho[rho_key(rho)] = {"rho": rho, "epsilon_R": cert, "tests_below_certificate": tested, "violations": violations}
        total += violations
    verdict = "GEO1_ONE_SIDED_CANTOR_CERTIFICATE_VALID" if total == 0 else "GEO2_IMPLEMENTATION_FAILURE"
    write_json(RESULTS / "tables/certificate_validation.json", {
        "W_R": W_R, "formula": "epsilon_R=W_R*rho^2*(1-2rho)", "n_states": len(H),
        "per_rho": per_rho, "total_violations": total, "verdict": verdict,
        "scope": "residual-L2 direct terminal-risk-policy switch; not semantic safety"})
    print(verdict, "violations", total)
    if total: raise SystemExit("certificate implementation failure")


if __name__ == "__main__": main()
