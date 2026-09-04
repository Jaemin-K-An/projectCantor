"""Audit realized final risk-conditional budgets and global deployment burden."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from _common import Q_CAP, Q_TARGET, RESULTS, require_freeze, write_json


def main() -> None:
    require_freeze()
    frame = pd.read_csv(RESULTS / "raw/final_D_final_v351_harmful.csv")
    keys = ["pid", "family", "epsilon"]
    common = frame.groupby(keys).agg(
        risk_mask_values=("risk_eligible", "nunique"),
        d_values=("d_attacked", lambda x: np.ptp(np.asarray(x, float))),
    )
    common_mask_ok = bool((common.risk_mask_values == 1).all() and (common.d_values < 1e-10).all())
    acting = frame[~frame.arm.isin(["ATTACK_ONLY"])]
    rows, failed = {}, []
    for arm, group in acting.groupby("arm", sort=False):
        risk = group.risk_eligible.astype(bool).to_numpy()
        q = group.q_ctrl.to_numpy(float)
        q_raw = group.q_raw.to_numpy(float)
        risk_rms = float(np.sqrt(np.mean(q[risk] ** 2)))
        global_rms = float(np.sqrt(np.mean(q ** 2)))
        safe_frequency = float(np.mean(q[~risk] > 0)) if (~risk).any() else 0.0
        valid = (abs(risk_rms / Q_TARGET - 1) <= .03 and
                 q.max() <= Q_CAP + 1e-12 and safe_frequency == 0)
        by_family, by_epsilon = {}, {}
        for family, sub in group.groupby("family", sort=False):
            mask = sub.risk_eligible.astype(bool).to_numpy()
            z = sub.q_ctrl.to_numpy(float)
            by_family[family] = {
                "risk_q_rms": float(np.sqrt(np.mean(z[mask] ** 2))),
                "global_q_rms": float(np.sqrt(np.mean(z ** 2))),
                "risk_prevalence": float(mask.mean()),
            }
        for (family, epsilon), sub in group.groupby(["family", "epsilon"], sort=False):
            mask = sub.risk_eligible.astype(bool).to_numpy()
            z = sub.q_ctrl.to_numpy(float)
            by_epsilon[f"{family}@{epsilon:.12g}"] = {
                "risk_q_rms": float(np.sqrt(np.mean(z[mask] ** 2))) if mask.any() else None,
                "global_q_rms": float(np.sqrt(np.mean(z ** 2))),
                "risk_n": int(mask.sum()),
            }
        rows[arm] = {
            "risk_q_rms": risk_rms, "global_q_rms": global_rms,
            "risk_eligible_prevalence": float(risk.mean()),
            "q_mean": float(q.mean()), "q_p95": float(np.quantile(q, .95)),
            "q_max": float(q.max()), "clipping_fraction": float(np.mean(q_raw > Q_CAP)),
            "intervention_frequency": float(np.mean(q > 0)),
            "safe_side_intervention_frequency": safe_frequency,
            "eligibility_mask_sha256": hashlib.sha256(risk.astype(np.uint8).tobytes()).hexdigest(),
            "by_family": by_family, "by_family_epsilon": by_epsilon, "valid": valid,
        }
        if not valid:
            failed.append(arm)
    verdict = "BUD1_RISK_CONDITIONAL_MATCHED" if common_mask_ok and not failed else "BUD2_RISK_CONDITIONAL_MISMATCH"
    write_json(RESULTS / "tables/final_budget_audit.json", {
        "q_target_risk_rms": Q_TARGET, "q_cap": Q_CAP,
        "global_rms_is_selection_target": False,
        "common_precontrol_eligibility_identical_across_arms": common_mask_ok,
        "per_arm": rows, "failed": failed, "verdict": verdict,
    })
    print(verdict, "failed", failed)


if __name__ == "__main__":
    main()
