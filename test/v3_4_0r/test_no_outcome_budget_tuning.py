import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_budget_was_frozen_before_the_final():
    freeze = json.loads((ROOT / "configs/v3_4_0r/PRE_ANALYSIS_FREEZE.json").read_text())
    assert freeze["status"] == "NOT_FROZEN_EXTERNAL_WINDOW_SHIFT"
    assert freeze["formal_freeze_valid"] is False
    assert freeze["D_final_r_touch_was_invalid"] is True


def test_frozen_eta_matches_the_calibration_output():
    freeze = json.loads((ROOT / "configs/v3_4_0r/PRE_ANALYSIS_FREEZE.json").read_text())
    assert freeze["budget"]["eta_per_arm"] is None
    assert freeze["budget"]["q_target_rms"] == 0.03
    assert freeze["budget"]["status"] == "NOT_RUN_EXTERNAL_WINDOW_SHIFT"


def test_attack_grid_was_inherited_not_reselected():
    freeze = json.loads((ROOT / "configs/v3_4_0r/PRE_ANALYSIS_FREEZE.json").read_text())
    old = json.loads((ROOT / "configs/v3_4_0/PRE_ANALYSIS_FREEZE.json").read_text())
    assert freeze["attacks"]["grid_absolute"] == old["attacks"]["generation_grid_absolute"]


def test_no_new_sensor_or_actuator_was_fitted():
    freeze = json.loads((ROOT / "configs/v3_4_0r/PRE_ANALYSIS_FREEZE.json").read_text())
    frozen = freeze["inherited_frozen"]
    assert frozen["sensor_w"] == "results/v3_4_0/cache/sensor_w.npy"
    assert frozen["actuator"] == "results/v3_3_5a/cache/v_p0.npy"
    assert frozen["layer"] == 14 and frozen["depth"] == 3
    assert len(frozen["rho_family"]) == 7


def test_q_target_was_not_changed_after_v340():
    freeze = json.loads((ROOT / "configs/v3_4_0r/PRE_ANALYSIS_FREEZE.json").read_text())
    assert freeze["budget"]["q_target_rms"] == 0.03
    assert freeze["frozen_q_target_rms"] == 0.03


def test_invalid_q025_final_is_quarantined():
    freeze = json.loads((ROOT / "configs/v3_4_0r/PRE_ANALYSIS_FREEZE.json").read_text())
    assert freeze["invalid_run"]["use_in_confirmatory_inference"] is False
    assert not (ROOT / "results/v3_4_0r/raw/final_D_final_r_harmful.csv").exists()
    assert (ROOT / freeze["invalid_run"]["preserved_at"]).exists()
