"""Estimate first common-grid safety failure and compare it with certificates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import CONFIG, RESULTS, read_json, write_json


def main() -> None:
    evaluator = read_json(CONFIG / "evaluator.json")
    attack = read_json(CONFIG / "attack_grid.json")
    frame = pd.read_csv(RESULTS / "tables/final_generation_scored.csv")
    endpoint = "semantic_harmful_compliance" if evaluator["status"] == "S1_VALID" else "harmful_compliance_proxy"
    threshold = float(attack["failure_criterion"]["semantic_harmful_compliance_threshold"])
    max_epsilon = float(frame.epsilon.max())
    rows = []
    for (pid, rho), group in frame.groupby(["pid", "rho"]):
        group = group.sort_values("epsilon")
        failed = group[group[endpoint] >= threshold]
        censored = failed.empty
        epsilon_fail = max_epsilon if censored else float(failed.epsilon.iloc[0])
        certificate = float(attack["certificate_by_rho"]["1/3" if abs(rho - 1 / 3) < 1e-9 else f"{rho:.2f}"])
        rows.append({
            "pid": pid, "rho": rho, "epsilon_fail": epsilon_fail,
            "right_censored": censored, "epsilon_cert": certificate,
            "fail_minus_certificate": epsilon_fail - certificate,
        })
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "tables/failure_thresholds.csv", index=False)
    summary = table.groupby("rho").agg(
        median_epsilon_fail=("epsilon_fail", "median"),
        censor_rate=("right_censored", "mean"),
        epsilon_cert=("epsilon_cert", "first"),
        median_fail_minus_certificate=("fail_minus_certificate", "median"),
    ).reset_index()
    summary.to_csv(RESULTS / "tables/failure_threshold_summary.csv", index=False)
    correlation = float(summary[["median_epsilon_fail", "epsilon_cert"]].corr(method="spearman").iloc[0, 1])
    write_json(RESULTS / "tables/failure_threshold_analysis.json", {
        "endpoint": endpoint, "right_censor_value": max_epsilon,
        "descriptive_spearman_certificate_vs_median_failure": correlation,
        "relationship_assumed": False, "rho": summary.to_dict(orient="records"),
    })
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
