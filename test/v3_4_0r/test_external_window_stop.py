import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_transport_passes_all_preregistered_thresholds():
    row = json.loads((ROOT / "results/v3_4_0r/tables/sensor_transfer.json").read_text())
    assert row["transport_verdict"] == "ST1_PASS"
    assert row["auroc"] >= 0.70
    assert row["balanced_accuracy_at_zero"] >= 0.65
    assert row["auroc_ci95"][0] > 0.60
    assert row["sensor_scope"] == "SENS2_REFUSAL_SENSOR_ONLY"


def test_frozen_window_fails_without_recalibration():
    row = json.loads((ROOT / "results/v3_4_0r/tables/external_window.json").read_text())
    old = json.loads((ROOT / "configs/v3_4_0/PRE_ANALYSIS_FREEZE.json").read_text())
    assert row["W"] == old["geometry"]["W"] == 2.2805212277347544
    assert row["W_recalibrated"] is False
    assert row["coverage"] == pytest.approx(130 / 150)
    assert row["coverage"] < row["coverage_min"] == 0.90
    assert row["verdict"] == "ST3_WINDOW_SHIFT"


def test_final_stage_rejects_nonfrozen_manifest():
    import sys
    sys.path.insert(0, str(ROOT / "scripts/v3_4_0r"))
    from _common import require_confirmatory_freeze

    with pytest.raises(RuntimeError, match="requires PRE_ANALYSIS_FROZEN"):
        require_confirmatory_freeze()


def test_no_canonical_final_or_utility_output_exists():
    assert not (ROOT / "results/v3_4_0r/raw/final_D_final_r_harmful.csv").exists()
    assert not (ROOT / "results/v3_4_0r/raw/utility_D_final_r_benign.csv").exists()


def test_resume_audit_records_invalid_work_without_using_it():
    row = json.loads((ROOT / "results/v3_4_0r/tables/POST_GATE_INVALIDATION.json").read_text())
    assert row["status"] == "INVALIDATED_NOT_CONFIRMATORY"
    assert row["invalid_final"]["use_in_claims"] is False
    assert row["no_final_statistics_computed"] is True
