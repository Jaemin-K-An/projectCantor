import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path("scripts/v3_3_5c").resolve()))
from fix_temporal_classifier import classify_temporal, td1_candidates


def _rows(lo_p0: float, lo_g1: float, hi_p0: float = 0.2, hi_g1: float = 0.2):
    return pd.DataFrame(
        [
            {"B2": 0.4, "distributed": "S4_EARLY_4", "single": "S1_P0_ONLY", "simult_lo": lo_p0, "simult_hi": hi_p0},
            {"B2": 0.4, "distributed": "S4_EARLY_4", "single": "S2_G1_ONLY", "simult_lo": lo_g1, "simult_hi": hi_g1},
        ]
    )


def test_td1_requires_distributed_to_beat_both_p0_and_g1():
    assert not td1_candidates(_rows(0.04, 0.01), 0.03)
    assert not td1_candidates(_rows(0.01, 0.04), 0.03)
    assert td1_candidates(_rows(0.04, 0.05), 0.03)


def test_distribution_rule_maps_to_revised_t2_label():
    assert classify_temporal(_rows(0.04, 0.05), 0.03)["verdict"] == "T2_DISTRIBUTION_SUPPORTED"


def test_no_automatic_global_accumulation_label():
    verdict = classify_temporal(_rows(-0.2, -0.1, hi_p0=-0.04, hi_g1=-0.01), 0.03)["verdict"]
    assert verdict == "T1_P0_CONCENTRATION_SUPPORTED"
    assert "GLOBAL" not in verdict and "ACCUMULATION" not in verdict
