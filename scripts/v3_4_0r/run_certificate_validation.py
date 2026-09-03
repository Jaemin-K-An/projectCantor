"""Phase 10 -- certificate implementation check on the new population."""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from cantor_guard_v340.attack import attack_v, attack_w  # noqa: E402
from cantor_guard_v340.cantor_geometry import Cell, classify, epsilon_h  # noqa: E402
from cantor_guard_v340r.controllers import CappedCantorController  # noqa: E402

from _common import CONFIG, RESULTS, frozen_actuator, frozen_sensor, read_json, rho_key, write_json  # noqa: E402


def main() -> None:
    freeze = read_json(CONFIG / "PRE_ANALYSIS_FREEZE.json")
    frozen = freeze["inherited_frozen"]
    W = float(frozen["W"])
    kappa_sensor, actuator = frozen_sensor(), frozen_actuator()
    kappa = float(kappa_sensor.coupling(actuator.v_safe))
    grid = [e for e in freeze["attacks"]["grid_absolute"] if e > 0]
    H = np.load(RESULTS / "cache" / "h_D_final_r_harmful.npy") \
        if (RESULTS / "cache" / "h_D_final_r_harmful.npy").exists() \
        else np.load(RESULTS / "cache" / "h_D_budget_attacked_r.npy")

    per_rho, total = {}, 0
    d0 = np.atleast_1d(kappa_sensor.distance(H))
    for rho in frozen["rho_family"]:
        key = rho_key(rho)
        ctrl = CappedCantorController(sensor=kappa_sensor, actuator=actuator, W=W,
                                      rho=float(rho), eta=0.0, q_cap=0.05,
                                      leaf_actions=frozen["leaf_actions"])
        cert = float(epsilon_h(float(rho), W))
        r0 = ctrl.inner.risk_coordinate(d0)
        base = [classify(float(x), float(rho), 3) if np.isfinite(x)
                else Cell("outside", float("nan"), float("nan")) for x in r0]
        violations = tested = 0
        for eps in grid:
            for family in ("ATTACK_W", "ATTACK_V"):
                for sign in ((-1, 1) if family == "ATTACK_W" else (1,)):
                    shift = sign * eps if family == "ATTACK_W" else -eps * kappa
                    r1 = ctrl.inner.risk_coordinate(d0 + shift)
                    att = [classify(float(x), float(rho), 3) if np.isfinite(x)
                           else Cell("outside", float("nan"), float("nan")) for x in r1]
                    for c0, c1 in zip(base, att):
                        if c0.kind != "leaf" or eps >= cert:
                            continue
                        tested += 1
                        if family == "ATTACK_W" and c1.kind == "leaf" and c1.index != c0.index:
                            violations += 1
        total += violations
        per_rho[key] = {"rho": float(rho), "epsilon_cert": cert,
                        "tests_below_certificate": tested, "violations": violations}
    verdict = "CERT1_VALID" if total == 0 else "CERT2_IMPLEMENTATION_FAILURE"
    write_json(RESULTS / "tables" / "certificate_validation.json", {
        "W": W, "n_states": int(H.shape[0]), "grid": grid, "per_rho": per_rho,
        "total_violations": int(total), "verdict": verdict,
        "scope": "certified residual-L2 radius against direct terminal-policy switching in "
                 "the frozen sensor coordinate. NOT a semantic safety or jailbreak guarantee, "
                 "and zero violations is implementation validation, not empirical robustness.",
    })
    print(f"{'rho':<7}{'eps_cert':>10}{'tests<cert':>12}{'violations':>12}")
    for key, row in per_rho.items():
        print(f"{key:<7}{row['epsilon_cert']:>10.4f}{row['tests_below_certificate']:>12}"
              f"{row['violations']:>12}")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
