import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path("scripts/v3_3_5c").resolve()))
from reanalyse_temporal_maxT import corrected_max_t, shared_prompt_index_matrix


CONTRASTS = [
    ["S4_EARLY_4", "S1_P0_ONLY"],
    ["S4_EARLY_4", "S2_G1_ONLY"],
    ["S5_EARLY_8", "S1_P0_ONLY"],
    ["S5_EARLY_8", "S2_G1_ONLY"],
]


def test_one_shared_prompt_index_matrix_is_deterministic():
    a = shared_prompt_index_matrix(25, 7, 19)
    b = shared_prompt_index_matrix(25, 7, 19)
    assert a.shape == (25, 7)
    np.testing.assert_array_equal(a, b)


def test_same_indices_drive_every_budget_and_contrast():
    rows = []
    for pid in range(8):
        for budget in (0.1, 0.2):
            values = {
                "S1_P0_ONLY": (pid + budget) % 2,
                "S2_G1_ONLY": pid % 2,
                "S4_EARLY_4": (pid // 2) % 2,
                "S5_EARLY_8": (pid // 3) % 2,
            }
            for schedule, refusal in values.items():
                rows.append({"pid": str(pid), "B2_target": budget, "schedule": schedule, "refusal": refusal})
    result = corrected_max_t(pd.DataFrame(rows), contrasts=CONTRASTS, sesoi=0.03, n_boot=40, seed=3)
    assert result.bootstrap_indices.shape == (40, 8)
    assert result.bootstrap_statistics.shape == (40, 8)
    # Reconstruct the first contrast with the shared IDX: no hidden per-budget RNG.
    wide = pd.DataFrame(rows).pivot_table(index="pid", columns=["B2_target", "schedule"], values="refusal")
    wide = wide.reindex(result.prompt_ids)
    d = (wide[(0.1, "S1_P0_ONLY")] - wide[(0.1, "S4_EARLY_4")]).to_numpy()
    np.testing.assert_allclose(result.bootstrap_statistics[:, 0], d[result.bootstrap_indices].mean(axis=1))


def test_corrected_analysis_preserves_family_dependence():
    df = pd.read_csv("results/v3_3_5b/raw/temporal_D_temporal_confirm.csv")
    result = corrected_max_t(df, contrasts=CONTRASTS, sesoi=0.03, n_boot=250, seed=7)
    assert np.isfinite(result.critical_value)
    assert result.critical_value > 1.96
    assert result.bootstrap_statistics.shape[1] == 16
