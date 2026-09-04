"""Phase 18: preregistered middle-third versus neighbouring rho controls."""
from __future__ import annotations

import pandas as pd

from _analysis import auc_wide
from _common import CONFIG, RESULTS, read_json, require_freeze, write_json
from _stats import max_t_intervals, shared_index

PRIMARY = ("0.30", "0.36", "0.40")


def family_verdict(rows, sesoi=.03):
    if rows and all(r["simultaneous_lo"] > sesoi for r in rows): return "RHO1_CANTOR_EMPIRICAL_GAIN"
    if any(r["simultaneous_hi"] < -sesoi for r in rows): return "RHO3_OTHER_BETTER"
    if rows and all(r["simultaneous_lo"] >= -sesoi and r["simultaneous_hi"] <= sesoi for r in rows): return "RHO2_EQUIVALENT"
    return "RHO4_INCONCLUSIVE"


def main() -> None:
    require_freeze(); audit = read_json(RESULTS / "tables/final_budget_audit.json")
    if audit["verdict"] != "BUD1_MATCHED": raise SystemExit("equal-budget comparison blocked")
    frame = pd.read_csv(RESULTS / "raw/final_D_final_v350_harmful.csv"); _auc, wide = auc_wide(frame)
    idx = shared_index(len(next(iter(wide.values()))), n_boot=20000, seed=int(read_json(CONFIG / "statistics.json")["seed"]))
    per, contrasts = {}, []
    for family, table in wide.items():
        per[f"{family}:1/3"] = table["1/3"].to_numpy(float)
        for arm in PRIMARY:
            per[f"{family}:{arm}"] = table[arm].to_numpy(float)
            contrasts.append((f"{family}:1/3", f"{family}:{arm}"))
    mt = max_t_intervals(per, contrasts, idx); by_family = {}
    for family in wide:
        rows = [r for r in mt["contrasts"] if r["arm"].startswith(family + ":")]
        by_family[family] = {"contrasts": rows, "verdict": family_verdict(rows)}
    verdict = by_family["ATTACK_V"]["verdict"]
    write_json(RESULTS / "tables/rho_family.json", {"primary_attack_family": "ATTACK_V", "SESOI": .03,
               "primary_controls": list(PRIMARY), "shared_bootstrap": True,
               "max_t": mt, "by_family": by_family, "rho_verdict": verdict})
    print(verdict)


if __name__ == "__main__": main()
