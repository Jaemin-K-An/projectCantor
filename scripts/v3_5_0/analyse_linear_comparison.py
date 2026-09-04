"""Phase 17: Cantor 1/3 versus matched continuous risk controller."""
from __future__ import annotations

import pandas as pd

from _analysis import auc_wide
from _common import CONFIG, RESULTS, read_json, require_freeze, write_json
from _stats import max_t_intervals, shared_index


def classify(lo, hi, sesoi):
    if lo > sesoi: return "LIN1_CANTOR_BEATS_LINEAR"
    if hi < -sesoi: return "LIN3_LINEAR_BETTER"
    if lo >= -sesoi and hi <= sesoi: return "LIN2_EQUIVALENT"
    return "LIN4_INCONCLUSIVE"


def main() -> None:
    require_freeze(); audit = read_json(RESULTS / "tables/final_budget_audit.json")
    if audit["verdict"] != "BUD1_MATCHED": raise SystemExit("equal-budget comparison blocked")
    frame = pd.read_csv(RESULTS / "raw/final_D_final_v350_harmful.csv"); _auc, wide = auc_wide(frame)
    idx = shared_index(len(next(iter(wide.values()))), n_boot=20000, seed=int(read_json(CONFIG / "statistics.json")["seed"]))
    per, contrasts = {}, []
    for family, table in wide.items():
        for arm in ("1/3", "LINEAR"): per[f"{family}:{arm}"] = table[arm].to_numpy(float)
        contrasts.append((f"{family}:1/3", f"{family}:LINEAR"))
    mt = max_t_intervals(per, contrasts, idx); by_family = {}
    for row in mt["contrasts"]:
        family = row["arm"].split(":")[0]
        by_family[family] = {"contrast": row, "verdict": classify(row["simultaneous_lo"], row["simultaneous_hi"], .03)}
    verdict = by_family["ATTACK_V"]["verdict"]
    write_json(RESULTS / "tables/linear_comparison.json", {"primary_attack_family": "ATTACK_V", "SESOI": .03,
               "shared_bootstrap": True, "max_t": mt, "by_family": by_family, "linear_verdict": verdict})
    print(verdict)


if __name__ == "__main__": main()
