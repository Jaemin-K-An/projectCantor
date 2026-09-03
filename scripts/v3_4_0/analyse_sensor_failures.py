"""Phase 21 -- where does the geometry break, and where does behaviour break?

Two thresholds per prompt and rho:
  epsilon_sensor_fail    first attack that moves the state into a materially
                         riskier terminal policy region
  epsilon_behaviour_fail first attack at which the safe behaviour is lost
Both are compared against the analytic epsilon_h(rho).  Prompts that never fail
are right-censored, never imputed.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "llm/src"))
from cantor_guard_v340.cantor_geometry import epsilon_h  # noqa: E402

from _common import CONFIG, RESULTS, read_json, rho_key, write_json  # noqa: E402


def first_crossing(group, column, predicate):
    g = group.sort_values("epsilon")
    hit = g[predicate(g[column])]
    return float(hit.epsilon.iloc[0]) if len(hit) else float("nan")


def main(split: str = "D_final_harmful") -> None:
    freeze = read_json(CONFIG / "PRE_ANALYSIS_FREEZE.json")
    W = float(freeze["geometry"]["W"])
    frame = pd.read_csv(RESULTS / "raw" / f"final_{split}.csv")
    # A prompt can only FAIL if it was safe to begin with.  Prompts already
    # unsafe at epsilon=0 are excluded rather than recorded as failing at the
    # smallest tested attack, which would fabricate a threshold.
    baseline = frame[frame.family == "NONE"].set_index(["rho_key", "pid"]).y_safe
    frame = frame[frame.family != "NONE"]

    rows = []
    excluded_not_safe_at_baseline = 0
    for (family, key, pid), group in frame.groupby(["family", "rho_key", "pid"]):
        if int(baseline.get((key, pid), 0)) != 1:
            excluded_not_safe_at_baseline += 1
            continue
        base = group.sort_values("epsilon").iloc[0]
        rows.append({
            "family": family, "rho_key": key, "pid": pid,
            "epsilon_behaviour_fail": first_crossing(group, "y_safe", lambda s: s == 0),
            "epsilon_sensor_fail": first_crossing(
                group, "cell_index",
                lambda s: (s.notna()) & (s < (base.cell_index if pd.notna(base.cell_index) else np.inf))),
            "censored_behaviour": bool((group.y_safe == 0).sum() == 0),
        })
    table = pd.DataFrame(rows)
    n_excluded = excluded_not_safe_at_baseline
    table.to_csv(RESULTS / "tables" / "failure_thresholds.csv", index=False)

    summary = {}
    for (family, key), group in table.groupby(["family", "rho_key"]):
        rho = next(r for r in freeze["geometry"]["rho_family"] if rho_key(r) == key)
        observed = group.epsilon_behaviour_fail.dropna()
        summary[f"{family}|{key}"] = {
            "family": family, "rho": float(rho), "epsilon_cert": float(epsilon_h(float(rho), W)),
            "n": int(len(group)), "n_censored": int(group.censored_behaviour.sum()),
            "censoring_rate": float(group.censored_behaviour.mean()),
            "median_epsilon_behaviour_fail": float(observed.median()) if len(observed) else None,
            "median_over_certificate": (float(observed.median() / epsilon_h(float(rho), W))
                                        if len(observed) else None),
        }
    write_json(RESULTS / "tables" / "failure_thresholds.json", {
        "split": split, "W": W, "by_family_and_rho": summary,
        "excluded_unsafe_at_baseline": int(n_excluded),
        "n_analysed": int(len(table)),
        "censoring_policy": "right-censored, never imputed",
        "interpretation": "The ratio median(epsilon_behaviour_fail)/epsilon_cert says how "
                          "far apart the certified radius and the behavioural failure point "
                          "are. A ratio far above 1 means the certificate is conservative by "
                          "orders of magnitude relative to what changes behaviour.",
    })
    print(f"{'family':<10}{'rho':<7}{'eps_cert':>10}{'med_fail':>10}{'ratio':>8}{'censor%':>9}")
    for row in summary.values():
        med = row["median_epsilon_behaviour_fail"]
        ratio = row["median_over_certificate"]
        print(f"{row['family']:<10}{rho_key(row['rho']):<7}{row['epsilon_cert']:>10.4f}"
              f"{(f'{med:.4f}' if med is not None else 'n/a'):>10}"
              f"{(f'{ratio:.1f}' if ratio is not None else 'n/a'):>8}"
              f"{row['censoring_rate']:>9.2f}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
