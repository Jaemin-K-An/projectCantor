"""Phase 18 -- censor-aware failure analysis.

V3.4.0 reported a median over observed events only while 62-82% were right
censored. That is not a population median and the "11-70x the certificate"
sentence rested on it. Here every prompt that never fails is censored at
epsilon_max, the survival curve is estimated on the grid, and a median is
reported ONLY if the curve crosses 0.5.

Also fixes the direction bug: r = 1/2 - d/(2W), so HIGHER leaf index is riskier.
Guard and outside states get their own categories and are never counted as
direct terminal failures -- the guard is deliberately conservative.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from cantor_guard_v340.cantor_geometry import epsilon_h  # noqa: E402

from _common import CONFIG, RESULTS, read_json, require_confirmatory_freeze, write_json  # noqa: E402
from _stats import discrete_survival  # noqa: E402
from analyse_controller_effect import endpoint_column  # noqa: E402


def classify_transition(base_kind, base_index, kind, index) -> str:
    if kind == "outside":
        return "OUTSIDE_CAPTURE"
    if kind == "guard":
        return "GUARD_CAPTURE"
    if kind != "leaf" or base_kind != "leaf":
        return "SAME_TERMINAL"
    if index == base_index:
        return "SAME_TERMINAL"
    # r = 1/2 - d/(2W): higher index means higher risk
    return "RISKIER_TERMINAL" if index > base_index else "SAFER_TERMINAL"


def main(split: str = "D_final_r_harmful") -> None:
    freeze = require_confirmatory_freeze()
    W = float(freeze["inherited_frozen"]["W"])
    grid = [e for e in freeze["attacks"]["grid_absolute"] if e > 0]
    frame = pd.read_csv(RESULTS / "raw" / f"final_{split}.csv")
    col = endpoint_column(frame)

    baseline = frame[frame.arm == "CLEAN"].set_index("pid")[col]
    attacked = frame[(frame.arm != "CLEAN") & (frame.family != "NONE")]

    rows = []
    for (arm, family, pid), group in attacked.groupby(["arm", "family", "pid"]):
        if int(baseline.get(pid, 0)) != 1:
            continue  # can only fail if it started safe
        g = group.sort_values("epsilon")
        unsafe = g[g[col] == 0]
        first = float(unsafe.epsilon.iloc[0]) if len(unsafe) else np.nan
        # reversion: became unsafe then safe again at a larger epsilon
        reverted = False
        if len(unsafe):
            after = g[g.epsilon > unsafe.epsilon.iloc[0]]
            reverted = bool((after[col] == 1).any())
        rows.append({"arm": arm, "family": family, "pid": pid,
                     "first_failure": first, "censored": not np.isfinite(first),
                     "reverted": reverted})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "tables" / "failure_events.csv", index=False)

    summary = {}
    for (arm, family), group in table.groupby(["arm", "family"]):
        surv = discrete_survival(group.first_failure.to_numpy(dtype=float), max(grid), grid)
        surv["reversion_rate"] = float(group.reverted.mean())
        surv["arm"], surv["family"] = arm, family
        if arm not in ("ATTACK_ONLY", "LINEAR"):
            rho = 1 / 3 if arm == "1/3" else float(arm)
            surv["epsilon_cert"] = float(epsilon_h(rho, W))
        summary[f"{family}|{arm}"] = surv

    geometry = {}
    for (arm, family), group in attacked.groupby(["arm", "family"]):
        if arm in ("ATTACK_ONLY",) or "cell_kind" not in group:
            continue
        base = frame[(frame.arm == arm) & (frame.family == "NONE")]
        base_map = {}
        if len(base):
            base_map = base.set_index("pid")[["cell_kind", "cell_index"]].to_dict("index")
        kinds = []
        for _, r in group.iterrows():
            b = base_map.get(r.pid, {})
            kinds.append(classify_transition(b.get("cell_kind"), b.get("cell_index"),
                                             r.cell_kind, r.cell_index))
        counts = pd.Series(kinds).value_counts(normalize=True).to_dict()
        geometry[f"{family}|{arm}"] = {k: float(v) for k, v in counts.items()}

    write_json(RESULTS / "tables" / "failure_survival.json", {
        "split": split, "endpoint": col, "grid": grid, "W": W,
        "excluded_unsafe_at_baseline": int(
            (baseline == 0).sum() * attacked.groupby(["arm", "family"]).ngroups),
        "by_arm_and_family": summary,
        "transition_categories": geometry,
        "direction_note": "r = 1/2 - d/(2W), so a HIGHER leaf index is riskier. "
                          "Guard and outside captures are separate categories and are "
                          "never counted as direct terminal failures.",
        "median_policy": "reported only if the survival curve crosses 0.5",
    })
    print(f"{'family':<10}{'arm':<12}{'n':>5}{'events':>8}{'censor%':>9}"
          f"{'median':>12}{'RMST':>9}{'revert%':>9}")
    for key, row in summary.items():
        med = row["median"] if row["median"] is not None else "NOT_IDENT"
        print(f"{row['family']:<10}{row['arm']:<12}{row['n']:>5}{row['n_events']:>8}"
              f"{row['censoring_rate']:>9.2f}{str(med):>12}"
              f"{row['restricted_mean_failure_free_epsilon']:>9.3f}{row['reversion_rate']:>9.2f}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
