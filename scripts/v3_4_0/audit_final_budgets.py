"""Phase 16 -- did every rho actually spend the same intervention energy?"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "llm/src"))
from _common import CONFIG, RESULTS, read_json, write_json  # noqa: E402


def main(split: str = "D_final_harmful") -> None:
    freeze = read_json(CONFIG / "PRE_ANALYSIS_FREEZE.json")
    target = float(freeze["budget"]["q_target_rms"])
    tol = float(freeze["budget"]["tolerance"])
    frame = pd.read_csv(RESULTS / "raw" / f"final_{split}.csv")
    rows = {}
    for key, group in frame.groupby("rho_key"):
        q = group.q_ctrl.to_numpy(dtype=float)
        q_rms = float(np.sqrt(np.mean(q**2)))
        rows[key] = {
            "rho": float(group.rho.iloc[0]), "n_rows": int(len(group)),
            "q_rms": q_rms, "q_mean": float(q.mean()),
            "q_p95": float(np.quantile(q, 0.95)), "q_max": float(q.max()),
            "relative_deviation": float(q_rms / target - 1),
            "within_tolerance": bool(abs(q_rms / target - 1) <= tol),
            "intervention_frequency": float(np.mean(q > 0)),
            "guard_frequency": float(np.mean(group.cell_kind == "guard")),
            "leaf_frequency": float(np.mean(group.cell_kind == "leaf")),
            "outside_frequency": float(np.mean(group.cell_kind == "outside")),
        }
    excluded = [k for k, v in rows.items() if not v["within_tolerance"]]
    write_json(RESULTS / "tables" / "final_budget_audit.json", {
        "split": split, "target_q_rms": target, "tolerance": tol,
        "per_rho": rows, "excluded_rho": excluded,
        "all_matched": len(excluded) == 0,
        "note": "A rho outside tolerance cannot enter equal-budget comparisons.",
    })
    print(f"{'rho':<7}{'q_rms':>9}{'dev':>9}{'ok':>5}{'act%':>7}{'guard%':>8}{'leaf%':>7}{'out%':>7}")
    for key, row in rows.items():
        print(f"{key:<7}{row['q_rms']:>9.4f}{row['relative_deviation']:>+9.4f}"
              f"{'Y' if row['within_tolerance'] else 'N':>5}{row['intervention_frequency']:>7.2f}"
              f"{row['guard_frequency']:>8.2f}{row['leaf_frequency']:>7.2f}{row['outside_frequency']:>7.2f}")
    print(f"\nexcluded: {excluded or 'none'}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
