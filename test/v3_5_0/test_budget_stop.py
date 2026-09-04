import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_q03_is_provably_unattainable_for_every_arm():
    row = json.loads((ROOT / "results/v3_5_0/tables/budget_calibration.json").read_text())
    assert row["verdict"] == "BUD2_MISMATCH"
    assert row["selection_targets"] == [.03]
    assert row["clipping_is_selection_gate"] is False
    assert row["eta_per_arm"] is None
    assert all(v["maximum_attainable_q_rms"] < .03 for v in row["per_arm"].values())


def test_conformal_calibration_is_label_and_endpoint_free():
    row = json.loads((ROOT / "results/v3_5_0/tables/risk_window_calibration.json").read_text())
    assert row["n"] == 300 and row["alpha"] == .05
    assert row["order_index_one_based"] == 286
    assert row["uses_labels"] is False
    assert row["uses_generated_endpoints"] is False
    assert row["final_data_used"] is False
    assert row["retuning_allowed"] is False


def test_certificate_has_zero_implementation_violations():
    row = json.loads((ROOT / "results/v3_5_0/tables/certificate_validation.json").read_text())
    assert row["verdict"] == "GEO1_ONE_SIDED_CANTOR_CERTIFICATE_VALID"
    assert row["total_violations"] == 0


def test_realised_attack_coordinate_movements_are_exact():
    frame = pd.read_csv(ROOT / "results/v3_5_0/raw/attacked_budget_states.csv")
    attack_w = frame[frame.family == "ATTACK_W"]
    attack_v = frame[(frame.family == "ATTACK_V") & (frame.epsilon > 0)]
    assert np.max(np.abs(attack_w.delta_d + attack_w.epsilon)) < 1e-12
    assert np.max(np.abs(attack_v.delta_d + .3102297257950366 * attack_v.epsilon)) < 1e-12
    assert np.max(np.abs(frame.delta_x) - frame.epsilon) < 1e-12
