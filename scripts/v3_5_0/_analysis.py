from __future__ import annotations

import pandas as pd

from _stats import auc_per_prompt


def endpoint_column(frame):
    if "y_safe" not in frame:
        raise RuntimeError("frozen external refusal labels missing; run score_refusal.py")
    return "y_safe"


def with_clean_baseline(frame):
    clean = frame[frame.arm == "CLEAN"]
    attacked = frame[frame.arm != "CLEAN"]
    parts = [attacked]
    for family in ("ATTACK_W", "ATTACK_V"):
        for arm in attacked.arm.unique():
            zero = clean.copy(); zero["family"] = family; zero["arm"] = arm
            parts.append(zero)
    return pd.concat(parts, ignore_index=True)


def auc_wide(frame):
    auc = auc_per_prompt(with_clean_baseline(frame), endpoint_column(frame))
    return auc, {family: block.pivot(index="pid", columns="arm", values="auc").dropna()
                 for family, block in auc.groupby("family")}
