"""Audit actual D_final controller budgets and policy occupancy by rho."""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import CONFIG, RESULTS, read_json, rho_key, write_json


def main() -> None:
    frame = pd.read_csv(RESULTS / "raw/final_p0_cantor.csv")
    config = read_json(CONFIG / "controller.json")
    target = float(config["q_target"])
    tolerance = float(config["budget_tolerance_relative"])
    rows = []
    for rho, group in frame.groupby("rho"):
        q = group.q_ctrl.to_numpy(float)
        kinds = group.cell_kind.fillna("unknown")
        record = {
            "rho": float(rho),
            "rho_key": rho_key(float(rho)),
            "q_target": target,
            "q_mean": float(q.mean()),
            "q_rms": float(np.sqrt(np.mean(q**2))),
            "q_p95": float(np.quantile(q, 0.95)),
            "q_max": float(q.max()),
            "intervention_frequency": float(np.mean(q > 0)),
            "guard_frequency": float(np.mean(kinds == "guard")),
            "leaf_frequency": float(np.mean(kinds == "leaf")),
            "outside_window_frequency": float(np.mean(kinds == "outside")),
        }
        record["relative_error"] = record["q_rms"] / target - 1
        record["matched"] = abs(record["relative_error"]) <= tolerance
        rows.append(record)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "tables/final_budget_audit.csv", index=False)
    mismatches = table.loc[~table.matched, "rho_key"].tolist()
    write_json(RESULTS / "tables/final_budget_audit.json", {
        "status": "BUDGET_MATCHED" if not mismatches else "BUDGET_MISMATCH",
        "target": target, "tolerance_relative": tolerance,
        "unmatched_rho": mismatches,
        "all_rho_eligible": not mismatches,
        "rho": rows,
    })
    print(table.to_string(index=False))
    print("BUDGET_MATCHED" if not mismatches else f"BUDGET_MISMATCH: {mismatches}")


if __name__ == "__main__":
    main()
