"""Analytic and numerical audit of the inherited one-sided risk transform."""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from cantor_guard_v351.risk_coordinate import lipschitz_slack  # noqa: E402
from _common import RESULTS, frozen_sensor, write_json  # noqa: E402


def main() -> None:
    sensor = frozen_sensor()
    rng = np.random.default_rng(35102)
    h = rng.normal(size=(10_000, sensor.w.size))
    delta = rng.normal(size=h.shape)
    slack = lipschitz_slack(sensor, h, delta)
    h_tight = np.tile(-10.0 * sensor.w_hat, (100, 1))
    scales = np.linspace(.001, 1.0, 100)
    delta_tight = -scales[:, None] * sensor.w_hat[None, :]
    tight_slack = lipschitz_slack(sensor, h_tight, delta_tight)
    violations = int(np.sum(slack < -1e-10) + np.sum(tight_slack < -1e-10))
    payload = {
        "theorem": "|x(h+Delta h)-x(h)| <= ||Delta h||_2",
        "analytic_proof": "d is 1-Lipschitz; phi(z)=max(0,-z) is 1-Lipschitz; the composition is 1-Lipschitz",
        "random_seed": 35102,
        "random_trials": len(slack),
        "minimum_random_slack": float(slack.min()),
        "tight_direction_trials": len(tight_slack),
        "maximum_tight_direction_abs_slack": float(np.max(np.abs(tight_slack))),
        "violations": violations,
        "scope": "residual-L2 risk-coordinate stability; not semantic safety",
        "verdict": "GEO1_RISK_COORDINATE_LIPSCHITZ" if violations == 0 else "GEO2_IMPLEMENTATION_FAILURE",
    }
    write_json(RESULTS / "tables/risk_lipschitz_proof.json", payload)
    print(payload)
    if violations:
        raise SystemExit("risk-coordinate Lipschitz implementation failure")


if __name__ == "__main__":
    main()
