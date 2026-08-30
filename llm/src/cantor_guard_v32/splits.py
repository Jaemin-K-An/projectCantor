"""V3.2 PHASE 2/3 -- clean five-way split with a semantic leakage audit.

V3.1 DEFECT: the realised budget gains were calibrated by measuring C_rms on
the very goals the final comparison was then read from. No performance metric
was optimised on the test set, but the fairness constraint was, which is not a
strict hold-out.

V3.2 partitions the 100 JBB harmful goals into five disjoint blocks, each with
exactly one job:

  D_direction    extract the refusal direction (per model)
  D_calibration  fit the threat coordinate (tau, sigma)
  D_budget       fit the per-controller gain eta to the target C_rms
  D_dev          pilot, attainability gate, power check
  D_test         the final comparison -- touched only after the freeze

Splitting is on `goal_group`, never the raw string, and the assignment is a
deterministic function of a SHA-256 of the goal group plus the split salt, so
it is reproducible without storing prompt text.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

BLOCKS = ("D_direction", "D_calibration", "D_budget", "D_dev", "D_test")
DEFAULT_SIZES = {"D_direction": 20, "D_calibration": 10, "D_budget": 10,
                 "D_dev": 10, "D_test": 50}


def stable_seed(*parts: str) -> int:
    """Deterministic seed from SHA-256. Python's built-in hash() is salted per
    process and must never be used for anything reproducible."""
    h = hashlib.sha256("||".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def make_split(df: pd.DataFrame, salt: str = "cantor-v3.2",
               sizes: dict | None = None,
               group_col: str = "goal_group",
               stratify_col: str = "category") -> pd.DataFrame:
    """Assign each goal group to one block, stratified by category.

    Within each category the groups are ordered by a keyed hash (deterministic
    but unrelated to dataset order) and dealt out in the block proportions, so
    every block sees all ten harm categories.
    """
    sizes = dict(sizes or DEFAULT_SIZES)
    total = sum(sizes.values())
    if total != len(df[group_col].unique()):
        raise ValueError(f"sizes sum to {total} but there are "
                         f"{df[group_col].nunique()} goal groups")
    groups = (df.groupby(group_col)
                .agg(category=(stratify_col, "first"), n=(group_col, "size"))
                .reset_index())
    groups["key"] = [stable_seed(salt, g) for g in groups[group_col]]

    # Deal out per category, largest-remainder, so proportions hold in each.
    quota = {b: sizes[b] / total for b in BLOCKS}
    assign = {}
    carry = {b: 0.0 for b in BLOCKS}
    for cat, sub in groups.groupby("category"):
        sub = sub.sort_values("key")
        want = {b: quota[b] * len(sub) + carry[b] for b in BLOCKS}
        take = {b: int(np.floor(want[b])) for b in BLOCKS}
        rem = len(sub) - sum(take.values())
        for b in sorted(BLOCKS, key=lambda b: -(want[b] - take[b]))[:rem]:
            take[b] += 1
        carry = {b: want[b] - take[b] for b in BLOCKS}
        i = 0
        for b in BLOCKS:
            for g in sub[group_col].values[i:i + take[b]]:
                assign[g] = b
            i += take[b]

    out = df.copy()
    out["block"] = out[group_col].map(assign)
    if out["block"].isna().any():
        raise RuntimeError("unassigned goal groups")
    return out


def token_jaccard(a: str, b: str) -> float:
    sa, sb = set(str(a).lower().split()), set(str(b).lower().split())
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def leakage_audit(df: pd.DataFrame, text_col: str = "prompt",
                  threshold: float = 0.6) -> pd.DataFrame:
    """Flag cross-block prompt pairs that are near-duplicates.

    Splitting on the behaviour label does not by itself guarantee the prompts
    are semantically distinct, so every cross-block pair is scored on token
    Jaccard overlap. Pairs at or above `threshold` are reported; a non-empty
    report is a blocker, because it means D_test is not independent of the
    blocks the controller was tuned on.
    """
    recs = []
    idx = df.reset_index(drop=True)
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            if idx.block[i] == idx.block[j]:
                continue
            s = token_jaccard(idx[text_col][i], idx[text_col][j])
            if s >= threshold:
                recs.append({"block_a": idx.block[i], "block_b": idx.block[j],
                             "pid_a": idx.pid[i], "pid_b": idx.pid[j],
                             "jaccard": round(s, 3)})
    return pd.DataFrame(recs, columns=["block_a", "block_b", "pid_a", "pid_b",
                                       "jaccard"])


def save_split(df: pd.DataFrame, path: str | Path, salt: str,
               sizes: dict) -> dict:
    """Write the split as ids only -- no harmful prompt text is persisted."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    blocks = {b: sorted(df[df.block == b].pid.tolist()) for b in BLOCKS}
    digest = hashlib.sha256(
        json.dumps(blocks, sort_keys=True).encode("utf-8")).hexdigest()
    payload = {"salt": salt, "sizes": sizes, "n_goals": int(len(df)),
               "blocks": blocks, "split_sha256": digest,
               "category_balance": {
                   b: df[df.block == b].category.value_counts().to_dict()
                   for b in BLOCKS}}
    path.write_text(json.dumps(payload, indent=2))
    return payload


def load_split(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
