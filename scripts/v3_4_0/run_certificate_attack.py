"""Phase 14 -- certificate validation.  Pure geometry, no generation.

Claim under test: for ATTACK-W, if the clean state sits in a terminal leaf and
``eps < epsilon_h(rho) = 2W M_3(rho)``, the attacked state cannot land in a
DIFFERENT terminal leaf.  Landing in the same leaf, in a guard, or outside the
window is allowed.

This validates an analytic theorem's IMPLEMENTATION.  Zero violations is not
independent evidence of semantic robustness and is never reported as such.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "llm/src"))
from cantor_guard_v340.actuator import Actuator  # noqa: E402
from cantor_guard_v340.cantor_geometry import Cell, classify, epsilon_h  # noqa: E402
from cantor_guard_v340.sensor_actuator_controller import SensorActuatorCantorController  # noqa: E402
from cantor_guard_v340.sensor_distance import SensorHyperplane  # noqa: E402

from _common import CONFIG, RESULTS, read_json, rho_key, write_json  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> None:
    freeze = read_json(CONFIG / "PRE_ANALYSIS_FREEZE.json")
    sensor = SensorHyperplane(np.load(RESULTS / "cache" / "sensor_w.npy"), float(freeze["sensor"]["b"]))
    actuator = Actuator(np.load(ROOT / freeze["actuator"]["file"]).astype(float).reshape(-1),
                        int(freeze["actuator"]["safe_sign"]))
    W = float(freeze["geometry"]["W"])
    kappa = float(freeze["coupling"]["kappa"])
    grid = list(freeze["attacks"]["certificate_validation_grid_absolute"])

    H = np.load(RESULTS / "cache" / "h_D_final_harmful.npy")
    per_rho, rows = {}, []
    total_violations = 0
    for rho in freeze["geometry"]["rho_family"]:
        key = rho_key(rho)
        controller = SensorActuatorCantorController(
            sensor=sensor, actuator=actuator, W=W, rho=float(rho), eta=0.0,
            leaf_actions=freeze["geometry"]["leaf_actions"],
        )
        cert = float(epsilon_h(float(rho), W))
        _, _, clean_cells = controller.classify_state(H)
        violations = below_cert = 0
        for family in ("ATTACK_W", "ATTACK_V"):
            for eps in grid:
                # Both signs of the sensor-normal attack; ATTACK-V moves d by
                # only -eps*kappa, so it is reported but is not the tight test.
                for sign in ((-1, 1) if family == "ATTACK_W" else (1,)):
                    shift = sign * eps if family == "ATTACK_W" else -eps * kappa
                    d_att = np.atleast_1d(sensor.distance(H)) + shift
                    r_att = controller.risk_coordinate(d_att)
                    att_cells = tuple(
                        classify(float(x), float(rho), 3) if np.isfinite(x)
                        else Cell("outside", float("nan"), float("nan")) for x in r_att
                    )
                    for c0, c1 in zip(clean_cells, att_cells):
                        if c0.kind != "leaf":
                            continue
                        crossed = c1.kind == "leaf" and c1.index != c0.index
                        if eps < cert:
                            below_cert += 1
                            if family == "ATTACK_W" and crossed:
                                violations += 1
                        rows.append({"rho_key": key, "family": family, "epsilon": float(eps),
                                     "sign": sign, "below_certificate": bool(eps < cert),
                                     "direct_leaf_crossing": bool(crossed)})
        total_violations += violations
        crossings_by_eps = {}
        for eps in grid:
            sel = [r for r in rows if r["rho_key"] == key and r["family"] == "ATTACK_W"
                   and abs(r["epsilon"] - eps) < 1e-12]
            crossings_by_eps[f"{eps:.6f}"] = float(np.mean([r["direct_leaf_crossing"] for r in sel])) if sel else None
        per_rho[key] = {
            "rho": float(rho), "epsilon_cert": cert,
            "n_clean_in_leaf": int(sum(c.kind == "leaf" for c in clean_cells)),
            "n_clean_in_guard": int(sum(c.kind == "guard" for c in clean_cells)),
            "n_clean_outside": int(sum(c.kind == "outside" for c in clean_cells)),
            "tests_below_certificate": below_cert,
            "violations_below_certificate": violations,
            "attack_w_crossing_rate_by_epsilon": crossings_by_eps,
        }
    verdict = "CERT1_CANTOR_SENSOR_CERTIFICATE_VALID" if total_violations == 0 else "CERT2_IMPLEMENTATION_FAILURE"
    write_json(RESULTS / "tables" / "certificate_validation.json", {
        "W": W, "grid_absolute": grid, "n_prompts": int(H.shape[0]),
        "per_rho": per_rho, "total_violations": int(total_violations),
        "verdict": verdict,
        "scope": "Implementation validation of an analytic theorem. Zero violations "
                 "is NOT independent evidence of semantic robustness.",
    })
    print(f"{'rho':<7}{'eps_cert':>10}{'leaf':>6}{'guard':>7}{'out':>5}{'tests<cert':>12}{'violations':>12}")
    for key, row in per_rho.items():
        print(f"{key:<7}{row['epsilon_cert']:>10.4f}{row['n_clean_in_leaf']:>6}"
              f"{row['n_clean_in_guard']:>7}{row['n_clean_outside']:>5}"
              f"{row['tests_below_certificate']:>12}{row['violations_below_certificate']:>12}")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
