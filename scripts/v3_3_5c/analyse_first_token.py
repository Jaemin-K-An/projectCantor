"""Aggregate preregistered first-token mechanistic endpoints."""
from __future__ import annotations

import pandas as pd

from _common import RESULTS, write_json


def main() -> None:
    frame = pd.read_csv(RESULTS / "raw/final_p0_cantor.csv")
    metrics = [
        "kl_clean_attacked", "kl_clean_corrected", "attacked_top1_flip",
        "corrected_top1_flip", "attacked_topk_overlap", "corrected_topk_overlap",
    ]
    summary = frame.groupby(["rho", "epsilon"])[metrics].mean().reset_index()
    summary["kl_restoration"] = summary.kl_clean_attacked - summary.kl_clean_corrected
    summary["topk_restoration"] = summary.corrected_topk_overlap - summary.attacked_topk_overlap
    summary.to_csv(RESULTS / "tables/first_token_analysis.csv", index=False)
    write_json(RESULTS / "tables/first_token_analysis.json", {
        "role": "secondary_mechanistic_only",
        "does_not_establish_semantic_safety": True,
        "mean_kl_restoration": float(summary.kl_restoration.mean()),
        "mean_topk_restoration": float(summary.topk_restoration.mean()),
    })
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
