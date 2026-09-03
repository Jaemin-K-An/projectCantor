"""Freeze the largest symmetric, contiguous, non-degenerate DEV dose range."""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import RESULTS, behavioral_protocol, write_json


def select_confirm_grid(frame: pd.DataFrame, protocol: dict) -> dict:
    summary = frame.groupby("u").agg(
        coherence=("coherence", "mean"),
        degeneration_rate=("degeneration", "mean"),
        refusal=("refusal_proxy", "mean"),
        q_median=("relative_norm_realised", "median"),
        q_p95=("relative_norm_realised", lambda x: float(np.quantile(x, 0.95))),
        q_max=("relative_norm_realised", "max"),
    ).reset_index()
    passing = {
        float(row.u): bool(
            row.coherence >= float(protocol["coherence_gate"])
            and row.degeneration_rate <= float(protocol["degeneration_rate_gate"])
        )
        for row in summary.itertuples()
    }
    candidates = sorted(float(x) for x in protocol["candidate_u_grid"])
    magnitudes = sorted({abs(x) for x in candidates if x > 0})
    selected = [0.0] if passing.get(0.0, False) else []
    for magnitude in magnitudes:
        inner = [x for x in candidates if abs(x) <= magnitude + 1e-12]
        symmetric = (-magnitude in candidates) and (magnitude in candidates)
        if symmetric and all(passing.get(x, False) for x in inner):
            selected = inner
        else:
            break
    selected_rates = summary[summary.u.isin(selected)].sort_values("u")
    bracketed = bool(
        len(selected_rates)
        and selected_rates.refusal.min() < 0.5 < selected_rates.refusal.max()
    )
    status = "READY_FOR_CONFIRM" if selected and bracketed else "DEV_BEHAVIORAL_GATE_FAILED"
    return {
        "status": status,
        "selection_uses_behavior_only_for_required_bracketing": True,
        "selection_uses_rho": False,
        "confirm_u_grid": selected,
        "transition_0_5_bracketed_on_dev": bracketed,
        "dose_summary": summary.to_dict(orient="records"),
        "excluded_doses": [x for x in candidates if x not in selected],
    }


def main() -> None:
    protocol = behavioral_protocol()
    frame = pd.read_csv(RESULTS / "raw/symmetric_D_beh_P0_dev_335c.csv")
    result = select_confirm_grid(frame, protocol)
    write_json(RESULTS / "tables/p0_dose_grid_freeze.json", result)
    print(result["status"])
    print("confirm grid:", result["confirm_u_grid"])
    if result["status"] != "READY_FOR_CONFIRM":
        raise SystemExit("STOP: DEV did not provide a non-degenerate bracketed range")


if __name__ == "__main__":
    main()
