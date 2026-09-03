"""Phase 19 -- paired prompt-level max-T inference on the final endpoint.

The endpoint is REFUSAL robustness AUC, not semantic safety: Phase 1 found no
evaluator meeting the 0.85 harmful-compliance gate.  Comparisons are the three
preregistered Cantor-vs-neighbour contrasts; the comparator is never chosen
after seeing the data.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "llm/src"))
from _common import CONFIG, RESULTS, read_json, rho_key, write_json  # noqa: E402
from _stats import auc_per_prompt, max_t_intervals, shared_index  # noqa: E402


def main(split: str = "D_final_harmful") -> None:
    freeze = read_json(CONFIG / "PRE_ANALYSIS_FREEZE.json")
    audit = read_json(RESULTS / "tables" / "final_budget_audit.json")
    stats_cfg = freeze["statistics"]
    sesoi = float(stats_cfg["SESOI"])
    frame = pd.read_csv(RESULTS / "raw" / f"final_{split}.csv")

    # The unattacked arm belongs to both attack curves at epsilon = 0.
    zero = frame[frame.family == "NONE"].copy()
    parts = [frame[frame.family != "NONE"]]
    for family in ("ATTACK_W", "ATTACK_V"):
        dup = zero.copy()
        dup["family"] = family
        parts.append(dup)
    curve = pd.concat(parts, ignore_index=True)

    results = {}
    for family, block in curve.groupby("family"):
        auc = auc_per_prompt(block, value_col="y_safe", x_col="epsilon",
                             group_cols=("rho_key", "pid"))
        wide = auc.pivot(index="pid", columns="rho_key", values="auc").dropna()
        arms = [rho_key(r) for r in freeze["geometry"]["rho_family"]]
        # Budget-mismatched rho cannot enter a CONFIRMATORY equal-budget
        # comparison (frozen rule).  They are still reported descriptively --
        # suppressing measured numbers would be worse than labelling them --
        # but the confirmatory verdict stays blocked by the classifier.
        usable = [a for a in arms if a in wide.columns]
        idx = shared_index(len(wide), n_boot=int(stats_cfg["n_boot"]), seed=int(stats_cfg["seed"]))
        per_prompt = {a: wide[a].to_numpy(dtype=float) for a in usable}
        contrasts = [("1/3", rho_key(b)) for _, b in stats_cfg["primary_comparisons"]
                     if rho_key(b) in usable]
        maxt = max_t_intervals(per_prompt, contrasts, idx)
        for row in maxt["contrasts"]:
            lo, hi = row["simultaneous_lo"], row["simultaneous_hi"]
            row["favours_cantor"] = bool(lo > sesoi)
            row["favours_other"] = bool(hi < -sesoi)
            row["within_sesoi"] = bool(lo >= -sesoi and hi <= sesoi)
        results[family] = {
            "n_prompts": int(len(wide)),
            "mean_auc_by_rho": {a: float(wide[a].mean()) for a in usable},
            "rank_by_mean_auc": sorted(usable, key=lambda a: -float(wide[a].mean())),
            "max_t": maxt,
            "all_within_sesoi": bool(all(r["within_sesoi"] for r in maxt["contrasts"])),
            "any_favours_cantor": bool(any(r["favours_cantor"] for r in maxt["contrasts"])),
            "all_favour_cantor": bool(maxt["contrasts"]) and bool(
                all(r["favours_cantor"] for r in maxt["contrasts"])),
            "any_favours_other": bool(any(r["favours_other"] for r in maxt["contrasts"])),
        }
        auc.to_csv(RESULTS / "tables" / f"auc_per_prompt_{family}.csv", index=False)

    blocked = len(audit["excluded_rho"]) > 0
    write_json(RESULTS / "tables" / "generation_analysis.json", {
        "split": split, "endpoint": "refusal robustness AUC (SEM3_PROXY_ONLY)",
        "confirmatory_comparison_blocked": blocked,
        "blocked_reason": (
            "Every rho missed the frozen +/-3% budget tolerance on D_final "
            "(realised q_rms 0.0343-0.0374 against a 0.0300 target). eta was "
            "calibrated on CLEAN states but the controller runs on ATTACKED "
            "states, which sit in higher-action cells and fall outside the "
            "window more often (18% vs 12%). The deviation also grows with rho "
            "(+14% at 0.25 to +25% at 0.44), so larger rho spent more energy. "
            "The numbers below are DESCRIPTIVE only; no equal-budget claim "
            "follows from them, and nothing was retuned after D_final."
        ) if blocked else None,
        "sesoi": sesoi, "primary_comparisons": stats_cfg["primary_comparisons"],
        "excluded_rho": audit["excluded_rho"], "by_family": results,
        "no_post_hoc_comparator_selection": True,
    })
    for family, row in results.items():
        print(f"\n=== {family}  (n={row['n_prompts']} prompts)")
        for arm, value in sorted(row["mean_auc_by_rho"].items(), key=lambda kv: -kv[1]):
            print(f"   rho={arm:<6} mean AUC {value:.4f}")
        print(f"   max-T critical value {row['max_t']['critical_value']:.4f}")
        for c in row["max_t"]["contrasts"]:
            tag = ("CANTOR" if c["favours_cantor"] else
                   "OTHER" if c["favours_other"] else
                   "within SESOI" if c["within_sesoi"] else "inconclusive")
            print(f"   1/3 - {c['reference']:<5} {c['mean_difference']:+.4f} "
                  f"[{c['simultaneous_lo']:+.4f}, {c['simultaneous_hi']:+.4f}]  {tag}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
