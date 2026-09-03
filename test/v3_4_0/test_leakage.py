import itertools
import json
import pathlib

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/v3_4_0"
SPLITS = json.loads((ROOT / "configs/v3_4_0/splits.json").read_text())


def test_no_overlap_with_any_prior_version():
    assert SPLITS["hash_overlap_with_all_prior"] == 0
    assert SPLITS["prior_hashes_excluded"] >= 1900


def test_all_v340_splits_are_mutually_disjoint():
    blocks = SPLITS["blocks"]
    for a, b in itertools.combinations(blocks, 2):
        assert not (set(blocks[a]) & set(blocks[b])), f"{a} overlaps {b}"
    assert SPLITS["within_v340_overlap"] == 0


def test_sizes_match_the_recorded_blocks():
    for name, size in SPLITS["sizes"].items():
        assert len(SPLITS["blocks"][name]) == size


def test_final_split_meets_the_minimum():
    assert SPLITS["sizes"]["D_final_harmful"] >= 60
    assert SPLITS["sizes"]["D_final_benign"] >= 50


def test_final_prompts_never_appear_in_any_calibration_stage():
    final = set(SPLITS["blocks"]["D_final_harmful"]) | set(SPLITS["blocks"]["D_final_benign"])
    for name in ("D_sensor_train", "D_sensor_tune", "D_sensor_confirm",
                 "D_actuator_validate", "D_window_cal", "D_controller_budget", "D_attack_dev"):
        assert not (final & set(SPLITS["blocks"][name]))


def test_raw_tables_carry_no_prompt_or_completion_text():
    """Only hashes and scalars are tracked; text lives in gitignored private/."""
    for path in sorted((RESULTS / "raw").glob("*.csv")):
        columns = set(pd.read_csv(path, nrows=1).columns)
        assert "prompt" not in columns, path.name
        assert "completion" not in columns, path.name


def test_private_directory_is_gitignored():
    ignore = (ROOT / ".gitignore").read_text()
    assert "results/v3_4_0/private/" in ignore
    assert "results/v3_4_0/cache/" in ignore


def test_no_retuning_after_the_freeze():
    freeze = json.loads((ROOT / "configs/v3_4_0/PRE_ANALYSIS_FREEZE.json").read_text())
    assert freeze["D_final_touched"] is False
    fit = json.loads((RESULTS / "tables" / "sensor_confirm.json").read_text())
    assert freeze["sensor"]["C"] == pytest.approx(fit["C_selected"])
    assert freeze["sensor"]["b"] == pytest.approx(fit["b"])
    budgets = json.loads((RESULTS / "tables" / "controller_budgets.json").read_text())
    for key, eta in freeze["budget"]["eta_per_rho"].items():
        assert eta == pytest.approx(budgets["per_rho"][key]["eta"])


def test_window_was_calibrated_on_its_own_split():
    geom = json.loads((RESULTS / "tables" / "sensor_window_and_coupling.json").read_text())
    assert geom["window_split_n"] == SPLITS["sizes"]["D_window_cal"]
    assert geom["same_W_every_rho"] is True
