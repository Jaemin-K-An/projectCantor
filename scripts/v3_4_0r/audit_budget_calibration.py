"""Audit a permitted attacked-state calibration against frozen q=.03/cap=.05."""
from __future__ import annotations

from _common import Q_CAP, Q_TARGET, RESULTS, read_json, require_external_window_pass, write_json


def main() -> None:
    require_external_window_pass()
    row = read_json(RESULTS / "tables/budget_calibration.json")
    selected = row.get("q_target_selected")
    per_arm = row.get("feasibility", {}).get(str(Q_TARGET), {}).get("per_arm", {})
    checks = {
        "target_exact": selected == Q_TARGET,
        "all_arms_present": len(per_arm) == 8,
        "all_within_1pct": bool(per_arm) and all(abs(x["q_rms"] / Q_TARGET - 1) <= .01 for x in per_arm.values()),
        "hard_cap": bool(per_arm) and all(x["q_max"] <= Q_CAP + 1e-12 for x in per_arm.values()),
        "no_generation": row.get("generation_performed") is False,
        "no_labels": row.get("labels_consulted") is False,
    }
    payload = {"checks": checks, "verdict": "BUD1_MATCHED" if all(checks.values()) else "BUD0_NOT_FEASIBLE"}
    write_json(RESULTS / "tables/budget_calibration_audit.json", payload)
    print(payload["verdict"], checks)


if __name__ == "__main__":
    main()
