"""Mechanical V3.3.5c classifier for the corrected temporal analysis."""
from __future__ import annotations

import argparse
import json
import pathlib

import pandas as pd


DISTRIBUTED = ("S4_EARLY_4", "S5_EARLY_8")
P0 = "S1_P0_ONLY"
G1 = "S2_G1_ONLY"


def td1_candidates(table: pd.DataFrame, sesoi: float) -> list[dict]:
    """A candidate must beat BOTH P0_ONLY and G1_ONLY at the same B2."""
    winners: list[dict] = []
    for budget in sorted(table.B2.unique()):
        for schedule in DISTRIBUTED:
            rows = table[(table.B2 == budget) & (table.distributed == schedule)]
            lower = {str(r.single): float(r.simult_lo) for r in rows.itertuples()}
            if lower.get(P0, float("-inf")) > sesoi and lower.get(G1, float("-inf")) > sesoi:
                winners.append({"B2": float(budget), "schedule": schedule, "lower": lower})
    return winners


def classify_temporal(table: pd.DataFrame, sesoi: float) -> dict:
    candidates = td1_candidates(table, sesoi)
    if candidates:
        return {
            "verdict": "T2_DISTRIBUTION_SUPPORTED",
            "td1_candidates": candidates,
            "reason": "a distributed schedule materially outperforms both P0_ONLY and G1_ONLY at one matched B2",
        }

    vs_p0 = table[table.single == P0]
    p0_supported = bool((vs_p0.simult_hi < -sesoi).any())
    if p0_supported:
        return {
            "verdict": "T1_P0_CONCENTRATION_SUPPORTED",
            "td1_candidates": [],
            "reason": "at least one preregistered distributed-minus-P0 interval lies wholly below -SESOI",
        }

    equivalent = bool(
        len(table)
        and (table.simult_lo >= -sesoi).all()
        and (table.simult_hi <= sesoi).all()
    )
    if equivalent:
        return {
            "verdict": "T3_EQUIVALENT",
            "td1_candidates": [],
            "reason": "all preregistered simultaneous intervals lie within the equivalence region",
        }
    return {
        "verdict": "T4_INCONCLUSIVE",
        "td1_candidates": [],
        "reason": "neither the distribution, P0-concentration, nor equivalence rule is satisfied",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table",
        type=pathlib.Path,
        default=pathlib.Path("results/v3_3_5c/tables/temporal_contrasts_corrected.csv"),
    )
    parser.add_argument("--sesoi", type=float, default=0.03)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("results/v3_3_5c/tables/temporal_classifier.json"),
    )
    args = parser.parse_args()
    result = classify_temporal(pd.read_csv(args.table), args.sesoi)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(result["verdict"])
    print(result["reason"])


if __name__ == "__main__":
    main()
