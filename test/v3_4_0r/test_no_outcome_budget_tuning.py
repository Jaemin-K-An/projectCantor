import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_budget_was_frozen_before_the_final():
    freeze = json.loads((ROOT / "configs/v3_4_0r/PRE_ANALYSIS_FREEZE.json").read_text())
    assert freeze["D_final_r_touched"] is False
    assert "D_final_r_harmful" in freeze["frozen_before"]


def test_frozen_eta_matches_the_calibration_output():
    freeze = json.loads((ROOT / "configs/v3_4_0r/PRE_ANALYSIS_FREEZE.json").read_text())
    cal = json.loads((ROOT / "results/v3_4_0r/tables/budget_calibration.json").read_text())
    assert freeze["budget"]["eta_per_arm"] == cal["eta_per_arm"]
    assert freeze["budget"]["q_target_rms"] == cal["q_target_selected"]


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


def test_budget_choice_is_disclosed_as_weaker_than_v340():
    freeze = json.loads((ROOT / "configs/v3_4_0r/PRE_ANALYSIS_FREEZE.json").read_text())
    note = freeze["budget"]["note"].lower()
    assert "budget validity" in note and "not efficacy" in note
    assert freeze["budget"]["q_target_rms"] < 0.03
