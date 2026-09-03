"""Phases 15-16 -- does the controller help, and does Cantor add anything?

Two questions V3.4.0 could not separate:
  QUESTION A  Cantor 1/3 vs ATTACK_ONLY  -- does the controller help at all?
  QUESTION B  Cantor 1/3 vs LINEAR       -- does the recursive partition add
                                            anything beyond the architecture?
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from _common import CONFIG, RESULTS, read_json, require_confirmatory_freeze, write_json  # noqa: E402
from _stats import auc_per_prompt, max_t_intervals, shared_index  # noqa: E402
from patch_claim_classifier import controller_verdict  # noqa: E402


def endpoint_column(frame) -> str:
    return "y_safe" if "y_safe" in frame.columns else "refusal_proxy"


def build_curve(frame):
    """Attach the CLEAN arm at epsilon=0 to every arm and family."""
    clean = frame[frame.arm == "CLEAN"]
    parts = [frame[frame.arm != "CLEAN"]]
    for family in ("ATTACK_W", "ATTACK_V"):
        for arm in frame[frame.arm != "CLEAN"].arm.unique():
            dup = clean.copy()
            dup["arm"] = arm
            dup["family"] = family
            parts.append(dup)
    return pd.concat(parts, ignore_index=True)


def main(split: str = "D_final_r_harmful") -> None:
    require_confirmatory_freeze()
    stats_cfg = read_json(CONFIG / "statistics.json")
    sesoi = float(stats_cfg["efficacy_SESOI"])
    frame = pd.read_csv(RESULTS / "raw" / f"final_{split}.csv")
    col = endpoint_column(frame)
    curve = build_curve(frame)

    results = {}
    for family, block in curve.groupby("family"):
        auc = auc_per_prompt(block, value_col=col, x_col="epsilon", group_cols=("arm", "pid"))
        wide = auc.pivot(index="pid", columns="arm", values="auc").dropna()
        idx = shared_index(len(wide), n_boot=int(stats_cfg["n_boot"]), seed=int(stats_cfg["seed"]))
        per_prompt = {a: wide[a].to_numpy(dtype=float) for a in wide.columns}
        contrasts = [tuple(c) for c in stats_cfg["controller_efficacy_contrasts"]
                     if c[0] in per_prompt and c[1] in per_prompt]
        maxt = max_t_intervals(per_prompt, contrasts, idx)
        for row in maxt["contrasts"]:
            lo, hi = row["simultaneous_lo"], row["simultaneous_hi"]
            row["favours_arm"] = bool(lo > sesoi)
            row["favours_reference"] = bool(hi < -sesoi)
            row["within_sesoi"] = bool(lo >= -sesoi and hi <= sesoi)
        results[family] = {
            "n_prompts": int(len(wide)), "endpoint": col,
            "mean_auc_by_arm": {a: float(wide[a].mean()) for a in wide.columns},
            "max_t": maxt,
        }
        auc.to_csv(RESULTS / "tables" / f"auc_per_prompt_{family}.csv", index=False)

    def find(family, arm, ref):
        for row in results.get(family, {}).get("max_t", {}).get("contrasts", []):
            if row["arm"] == arm and row["reference"] == ref:
                return row
        return None

    family_verdicts = {}
    for family in ("ATTACK_W", "ATTACK_V"):
        primary = find(family, "1/3", "ATTACK_ONLY")
        family_verdicts[family] = controller_verdict(
            interval_lo=None if primary is None else primary["simultaneous_lo"],
            interval_hi=None if primary is None else primary["simultaneous_hi"],
            efficacy_sesoi=sesoi,
        )
    values = set(family_verdicts.values())
    controller = values.pop() if len(values) == 1 else "CTRL4_INCONCLUSIVE"

    write_json(RESULTS / "tables" / "controller_effect.json", {
        "split": split, "endpoint": col, "sesoi": sesoi,
        "primary_efficacy_contrast": "1/3 vs ATTACK_ONLY, classified separately for ATTACK_W and ATTACK_V",
        "family_verdicts": family_verdicts,
        "by_family": results, "controller_verdict": controller,
        "note": "Similarity among rho arms is NOT evidence of inertness; only this "
                "contrast against the attacked no-controller baseline can establish it.",
    })
    for family, row in results.items():
        print(f"\n=== {family}  (n={row['n_prompts']}, endpoint {col})")
        for arm, value in sorted(row["mean_auc_by_arm"].items(), key=lambda kv: -kv[1]):
            print(f"   {arm:<12} mean AUC {value:.4f}")
        print(f"   max-T critical value {row['max_t']['critical_value']:.4f}")
        for c in row["max_t"]["contrasts"]:
            tag = ("arm better" if c["favours_arm"] else
                   "reference better" if c["favours_reference"] else
                   "within SESOI" if c["within_sesoi"] else "inconclusive")
            print(f"   {c['arm']:<12} - {c['reference']:<12} {c['mean_difference']:+.4f} "
                  f"[{c['simultaneous_lo']:+.4f}, {c['simultaneous_hi']:+.4f}]  {tag}")
    print(f"\nCONTROLLER VERDICT: {controller}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
