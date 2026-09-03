"""Phase 17 -- rho-family comparison, valid only under a matched budget."""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
sys.path.insert(0, str(ROOT / "scripts/v3_4_0r"))
from patch_claim_classifier import cantor_verdict, generation_verdict  # noqa: E402

from _common import CONFIG, RESULTS, read_json, write_json  # noqa: E402
from _stats import auc_per_prompt, max_t_intervals, shared_index  # noqa: E402
from analyse_controller_effect import build_curve, endpoint_column  # noqa: E402


def main(split: str = "D_final_r_harmful") -> None:
    stats_cfg = read_json(CONFIG / "statistics.json")
    audit = read_json(RESULTS / "tables" / "final_budget_audit.json")
    sesoi = float(stats_cfg["SESOI"])
    frame = pd.read_csv(RESULTS / "raw" / f"final_{split}.csv")
    col = endpoint_column(frame)
    curve = build_curve(frame)

    blocked = audit["verdict"] == "BUD2_MISMATCH"
    results = {}
    for family, block in curve.groupby("family"):
        auc = auc_per_prompt(block, value_col=col, x_col="epsilon", group_cols=("arm", "pid"))
        wide = auc.pivot(index="pid", columns="arm", values="auc").dropna()
        idx = shared_index(len(wide), n_boot=int(stats_cfg["n_boot"]), seed=int(stats_cfg["seed"]))
        per_prompt = {a: wide[a].to_numpy(dtype=float) for a in wide.columns}
        contrasts = [tuple(c) for c in stats_cfg["rho_primary_contrasts"]
                     if c[0] in per_prompt and c[1] in per_prompt]
        maxt = max_t_intervals(per_prompt, contrasts, idx)
        for row in maxt["contrasts"]:
            lo, hi = row["simultaneous_lo"], row["simultaneous_hi"]
            row["favours_cantor"] = bool(lo > sesoi)
            row["favours_other"] = bool(hi < -sesoi)
            row["within_sesoi"] = bool(lo >= -sesoi and hi <= sesoi)
        results[family] = {
            "n_prompts": int(len(wide)),
            "mean_auc_by_rho": {a: float(wide[a].mean()) for a in wide.columns
                                if a not in ("ATTACK_ONLY", "LINEAR")},
            "max_t": maxt,
            "all_within_sesoi": bool(all(r["within_sesoi"] for r in maxt["contrasts"])),
            "all_favour_cantor": bool(maxt["contrasts"]) and bool(all(r["favours_cantor"] for r in maxt["contrasts"])),
            "any_favours_other": bool(any(r["favours_other"] for r in maxt["contrasts"])),
        }

    primary = results.get("ATTACK_V") or next(iter(results.values()), {})
    kwargs = dict(budget_all_matched=audit["all_matched"] and audit["primary_arms_valid"],
                  comparison_blocked=blocked,
                  all_within_sesoi=primary.get("all_within_sesoi", False),
                  all_favour_cantor=primary.get("all_favour_cantor", False),
                  any_favours_other=primary.get("any_favours_other", False),
                  have_contrasts=bool(primary.get("max_t", {}).get("contrasts")))
    write_json(RESULTS / "tables" / "rho_family.json", {
        "split": split, "endpoint": col, "sesoi": sesoi,
        "budget_verdict": audit["verdict"], "confirmatory_comparison_blocked": blocked,
        "by_family": results,
        "generation_verdict": generation_verdict(**kwargs),
        "cantor_verdict": cantor_verdict(**kwargs),
    })
    for family, row in results.items():
        print(f"\n=== {family}")
        for arm, value in sorted(row["mean_auc_by_rho"].items(), key=lambda kv: -kv[1]):
            print(f"   rho={arm:<6} mean AUC {value:.4f}")
        for c in row["max_t"]["contrasts"]:
            print(f"   1/3 - {c['reference']:<5} {c['mean_difference']:+.4f} "
                  f"[{c['simultaneous_lo']:+.4f}, {c['simultaneous_hi']:+.4f}]")
    print(f"\nGENERATION: {generation_verdict(**kwargs)}")
    print(f"CANTOR:     {cantor_verdict(**kwargs)}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
