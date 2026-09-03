import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_historical_constants_are_exact():
    cfg = json.loads((ROOT / "configs/v3_4_0r/controller.json").read_text())
    old = json.loads((ROOT / "configs/v3_4_0/PRE_ANALYSIS_FREEZE.json").read_text())
    assert cfg["frozen_from_v340"]["W"] == old["geometry"]["W"] == 2.2805212277347544
    assert cfg["NEW_IN_V340R"]["budget_calibration"]["q_target_rms"] == 0.03
    assert cfg["NEW_IN_V340R"]["hard_q_cap"]["q_cap"] == 0.05


def test_transport_and_window_gates_are_exact():
    cfg = json.loads((ROOT / "configs/v3_4_0r/controller.json").read_text())
    assert cfg["SENSOR_TRANSFER_GATE"]["gate"] == {
        "auroc_min": 0.70,
        "auroc_ci_lower_min": 0.60,
        "balanced_accuracy_at_zero_min": 0.65,
    }
    assert cfg["EXTERNAL_WINDOW_GATE"]["coverage_min"] == 0.90


def test_external_dataset_is_revision_and_byte_pinned():
    cfg = json.loads((ROOT / "configs/v3_4_0r/external_dataset.json").read_text())
    assert len(cfg["revision"]) == 40
    assert len(cfg["file_sha256"]) == 64
    assert cfg["selected_before_model_output"] is True
    assert cfg["selection_used_benchmark_outcomes"] is False
