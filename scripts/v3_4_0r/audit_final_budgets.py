"""Phase 13 -- stratified budget audit. A global mean can hide family imbalance."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from _common import CONFIG, RESULTS, read_json, write_json  # noqa: E402


def main(split: str = "D_final_r_harmful") -> None:
    freeze = read_json(CONFIG / "PRE_ANALYSIS_FREEZE.json")
    stats = read_json(CONFIG / "statistics.json")["equal_budget_gate"]
    target = float(freeze["budget"]["q_target_rms"])
    q_cap = float(freeze["hard_q_cap"]["q_cap"])
    frame = pd.read_csv(RESULTS / "raw" / f"final_{split}.csv")
    acting = frame[~frame.arm.isin(["CLEAN", "ATTACK_ONLY"])]

    rows, excluded = {}, []
    for arm, group in acting.groupby("arm"):
        q = group.q_ctrl.to_numpy(dtype=float)
        q_rms = float(np.sqrt(np.mean(q**2)))
        by_family = {}
        for family, sub in group.groupby("family"):
            qf = sub.q_ctrl.to_numpy(dtype=float)
            by_family[family] = {"q_rms": float(np.sqrt(np.mean(qf**2))),
                                 "relative_deviation": float(np.sqrt(np.mean(qf**2)) / target - 1)}
        by_eps = {f"{e:.4f}": float(np.sqrt(np.mean(s.q_ctrl.to_numpy(dtype=float) ** 2)))
                  for e, s in group.groupby("epsilon")}
        checks = {
            "overall_within_tolerance": bool(abs(q_rms / target - 1) <= float(stats["overall_tolerance"])),
            "family_within_tolerance": bool(all(abs(v["relative_deviation"]) <= float(stats["per_family_tolerance"])
                                                for v in by_family.values())),
            "q_max_within_cap": bool(q.max() <= q_cap + 1e-12),
        }
        rows[arm] = {"n_rows": int(len(group)), "q_rms": q_rms, "q_mean": float(q.mean()),
                     "q_p95": float(np.quantile(q, 0.95)), "q_max": float(q.max()),
                     "relative_deviation": float(q_rms / target - 1),
                     "by_family": by_family, "q_rms_by_epsilon": by_eps,
                     "clipping_fraction": float(group.clipped.fillna(False).astype(bool).mean()),
                     "intervention_frequency": float(np.mean(q > 0)),
                     "guard_frequency": float(np.mean(group.cell_kind == "guard")),
                     "leaf_frequency": float(np.mean(group.cell_kind == "leaf")),
                     "outside_frequency": float(np.mean(group.cell_kind == "outside")),
                     "checks": checks, "valid": all(checks.values())}
        if not all(checks.values()):
            excluded.append(arm)

    primary = ["1/3", "0.30", "0.36", "0.40", "LINEAR"]
    all_matched = not excluded
    primary_valid = all(rows[a]["valid"] for a in primary if a in rows)
    write_json(RESULTS / "tables" / "final_budget_audit.json", {
        "split": split, "target_q_rms": target, "q_cap": q_cap, "gate": stats,
        "per_arm": rows, "excluded_arms": excluded,
        "all_matched": all_matched, "primary_arms_valid": primary_valid,
        "verdict": "BUD1_MATCHED" if (all_matched and primary_valid) else "BUD2_MISMATCH",
    })
    print(f"{'arm':<8}{'q_rms':>9}{'dev':>9}{'q_max':>9}{'clip%':>8}{'ok':>4}"
          f"{'W dev':>9}{'V dev':>9}")
    for arm, row in rows.items():
        fw = row["by_family"].get("ATTACK_W", {}).get("relative_deviation", float("nan"))
        fv = row["by_family"].get("ATTACK_V", {}).get("relative_deviation", float("nan"))
        print(f"{arm:<8}{row['q_rms']:>9.4f}{row['relative_deviation']:>+9.4f}{row['q_max']:>9.4f}"
              f"{row['clipping_fraction']:>8.3f}{'Y' if row['valid'] else 'N':>4}{fw:>+9.4f}{fv:>+9.4f}")
    print(f"\nexcluded: {excluded or 'none'}   verdict: "
          f"{'BUD1_MATCHED' if (all_matched and primary_valid) else 'BUD2_MISMATCH'}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
