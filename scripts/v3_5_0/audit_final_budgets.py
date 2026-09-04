"""Phase 14: final q summaries and frozen equal-budget gate."""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import CONFIG, Q_CAP, Q_TARGET, RESULTS, read_json, require_freeze, write_json


def main() -> None:
    require_freeze(); frame = pd.read_csv(RESULTS / "raw/final_D_final_v350_harmful.csv")
    acting = frame[~frame.arm.isin(["CLEAN", "ATTACK_ONLY"])]
    rows, failed = {}, []
    for arm, group in acting.groupby("arm"):
        q = group.q_ctrl.to_numpy(float); qrms = float(np.sqrt(np.mean(q*q)))
        by_family = {}; by_epsilon = {}
        for family, sub in group.groupby("family"):
            z = sub.q_ctrl.to_numpy(float); by_family[family] = {"q_rms": float(np.sqrt(np.mean(z*z))), "q_mean": float(z.mean()), "q_max": float(z.max())}
        for (family, eps), sub in group.groupby(["family", "epsilon"]):
            z = sub.q_ctrl.to_numpy(float); by_epsilon[f"{family}@{eps:.12g}"] = float(np.sqrt(np.mean(z*z)))
        valid = abs(qrms / Q_TARGET - 1) <= .03 and q.max() <= Q_CAP + 1e-12
        rows[arm] = {"q_rms": qrms, "q_mean": float(q.mean()), "q_p95": float(np.quantile(q, .95)),
                     "q_max": float(q.max()), "clipping_fraction": float(group.clipped.fillna(False).astype(bool).mean()),
                     "intervention_frequency": float(np.mean(q > 0)), "by_family": by_family,
                     "by_family_epsilon": by_epsilon, "valid": valid}
        if not valid: failed.append(arm)
    verdict = "BUD1_MATCHED" if not failed else "BUD2_MISMATCH"
    write_json(RESULTS / "tables/final_budget_audit.json", {"q_target": Q_TARGET, "q_cap": Q_CAP, "per_arm": rows, "failed": failed, "verdict": verdict})
    print(verdict)


if __name__ == "__main__": main()
