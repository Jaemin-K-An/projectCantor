"""Shared final-data validation and AUC reshaping."""
from __future__ import annotations

import pandas as pd

from _stats import auc_per_prompt


def endpoint_column(frame):
    if "y_safe" not in frame:
        raise RuntimeError("frozen external refusal labels missing; run score_refusal.py")
    return "y_safe"


def auc_wide(frame, expected_grid):
    endpoint = endpoint_column(frame)
    counts = frame.groupby(["family", "arm", "pid"]).epsilon.nunique()
    if not (counts == len(expected_grid)).all():
        raise RuntimeError("incomplete harmful factorial grid")
    auc = auc_per_prompt(frame, endpoint)
    wide = {
        family: block.pivot(index="pid", columns="arm", values="auc").sort_index()
        for family, block in auc.groupby("family", sort=False)
    }
    expected_pids = None
    for family, table in wide.items():
        if table.isna().any().any():
            raise RuntimeError(f"missing AUC cell in {family}")
        pids = tuple(table.index)
        if expected_pids is None:
            expected_pids = pids
        elif pids != expected_pids:
            raise RuntimeError("prompt pairing differs across attack families")
    return auc, wide
