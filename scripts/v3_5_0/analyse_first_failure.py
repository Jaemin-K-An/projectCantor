"""Phase 20: secondary censor-aware first-failure and reversion analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import RESULTS, require_freeze, write_json
from _stats import discrete_survival


def main() -> None:
    freeze = require_freeze(); frame = pd.read_csv(RESULTS / "raw/final_D_final_v350_harmful.csv")
    if "y_safe" not in frame: raise SystemExit("run frozen refusal scoring first")
    grid = freeze["attacks"]["grid_absolute"]; rows = {}
    for (family, arm), group in frame[frame.arm != "CLEAN"].groupby(["family", "arm"]):
        first, reverted = [], []
        for _pid, p in group.sort_values("epsilon").groupby("pid"):
            failures = p[p.y_safe == 0]
            if failures.empty: first.append(np.nan); reverted.append(False); continue
            e = float(failures.epsilon.iloc[0]); first.append(e)
            reverted.append(bool((p[p.epsilon > e].y_safe == 1).any()))
        row = discrete_survival(first, grid); row["reversion_rate_after_observed_failure"] = float(np.mean(reverted))
        rows[f"{family}:{arm}"] = row
    write_json(RESULTS / "tables/first_failure.json", {"secondary_only": True,
               "no_dropna_median": True, "per_family_arm": rows})


if __name__ == "__main__": main()
