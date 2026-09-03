import itertools
import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPLITS = json.loads((ROOT / "configs/v3_4_0r/splits.json").read_text())
RESULTS = ROOT / "results/v3_4_0r"


def test_no_overlap_with_any_prior_version():
    assert SPLITS["hash_overlap_with_all_prior"] == 0
    assert SPLITS["prior_hashes_excluded"] >= 2500


def test_all_v340r_splits_are_mutually_disjoint():
    blocks = SPLITS["blocks"]
    for a, b in itertools.combinations(blocks, 2):
        assert not (set(blocks[a]) & set(blocks[b])), f"{a} overlaps {b}"
    assert SPLITS["within_v340r_overlap"] == 0


def test_final_never_touches_calibration_sets():
    blocks = SPLITS["blocks"]
    final = set(blocks["D_final_r_harmful"]) | set(blocks["D_final_r_benign"])
    for name in ("D_sensor_transfer_r", "D_eval_val_r", "D_eval_val_benign_r",
                 "D_budget_attacked_r"):
        assert not (final & set(blocks[name]))


def test_evaluator_validation_is_independent_of_budget_and_final():
    blocks = SPLITS["blocks"]
    val = set(blocks["D_eval_val_r"]) | set(blocks["D_eval_val_benign_r"])
    assert not (val & set(blocks["D_budget_attacked_r"]))
    assert not (val & set(blocks["D_final_r_harmful"]))


def test_population_change_is_disclosed():
    d = SPLITS["POPULATION_CHANGE_DISCLOSURE"]
    assert d["old_population"] == "declare-lab/HarmfulQA"
    assert "exhausted" in d["why"] or "28" in d["why"]
    assert "OUT OF ITS" in d["consequence"] or "out of its" in d["consequence"].lower()
    assert "D_sensor_transfer_r" in d["mitigation"]


def test_raw_tables_carry_no_prompt_or_completion_text():
    for path in sorted((RESULTS / "raw").glob("*.csv")):
        cols = set(pd.read_csv(path, nrows=1).columns)
        assert "prompt" not in cols, path.name
        assert "completion" not in cols, path.name


def test_private_and_cache_are_gitignored():
    ignore = (ROOT / ".gitignore").read_text()
    assert "results/v3_4_0r/private/" in ignore
    assert "results/v3_4_0r/cache/" in ignore


def test_final_split_meets_the_minimum():
    n = len(pd.read_csv(RESULTS / "cache" / "D_final_r_harmful.csv"))
    assert n >= 80
    assert SPLITS["sizes"]["D_final_r_benign"] >= 60
