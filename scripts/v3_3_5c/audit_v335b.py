"""Read-only source audit for V3.3.5b.

This script prints machine-derived facts only.  The narrative audit is kept in
docs/v3_3_5c/V335B_AUDIT.md and deliberately distinguishes the historical
official label from the corrected interpretation.
"""
from __future__ import annotations

import json
import pathlib
import subprocess

import pandas as pd


def main() -> None:
    raw = pd.read_csv("results/v3_3_5b/raw/temporal_D_temporal_confirm.csv")
    official = json.loads(pathlib.Path("results/v3_3_5b/tables/verdict_v335b.json").read_text())
    temporal = json.loads(pathlib.Path("results/v3_3_5b/tables/temporal_verdict.json").read_text())
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    effect = pd.DataFrame(temporal["effect_table"])
    print(f"current branch: {branch}")
    print(f"current HEAD: {head}")
    print("audited source: cantor-guard-v3.3.5b @ b5d4b1f0bde73d9cd643736e0eecaf0129645e9f")
    print("official verdict:", json.dumps(official, indent=2))
    print("matched B2: sqrt(sum q_t^2), uniform schedule weights, K=8")
    print(f"confirm prompts: {raw.pid.nunique()}")
    print(f"budget mismatches: {temporal['budget_mismatch_rows']}")
    print("effects (baseline refusal - condition refusal):")
    print(effect.to_string())
    print(f"original max-T critical value: {temporal['maxT_crit']:.6f}")
    print("semantic evaluator: lexical refusal proxy only; no validated harmful-compliance endpoint")
    print("untouched final sets: V3.3.5a D_final_P0=90; V3.3.5b D_final_traj=90")


if __name__ == "__main__":
    main()
