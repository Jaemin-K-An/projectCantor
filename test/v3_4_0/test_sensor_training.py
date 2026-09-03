import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from cantor_guard_v340.behavioral_sensor import SensorGate, fit_sensor, sensor_metrics

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/v3_4_0"


def test_sensor_is_l2_regularized_and_rejects_bad_C(rng):
    H, y = rng.normal(size=(60, 12)), (rng.random(60) > 0.5).astype(int)
    fit_sensor(H, y, C=0.1)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            fit_sensor(H, y, C=bad)


def test_training_requires_both_behaviour_classes(rng):
    H = rng.normal(size=(30, 5))
    with pytest.raises(ValueError):
        fit_sensor(H, np.ones(30, dtype=int), C=0.1)


def test_labels_come_from_output_behaviour_not_prompt_kind():
    """The section 8 shortcut -- harmful vs harmless PROMPT -- must not be the target."""
    train = pd.read_csv(RESULTS / "raw" / "clean_D_sensor_train.csv")
    assert train.kind.nunique() == 1, "sensor training set must be one prompt kind"
    assert set(train.y_safe.unique()) == {0, 1}, "both behaviours must occur within that kind"
    assert 0.15 < train.y_safe.mean() < 0.85


def test_sensor_trained_on_clean_states_only():
    summary = json.loads((RESULTS / "tables" / "clean_collection_summary.json").read_text())
    assert summary["intervention"] == "none"


def test_train_tune_confirm_are_disjoint():
    blocks = json.loads((ROOT / "configs/v3_4_0/splits.json").read_text())["blocks"]
    a, b, c = (set(blocks[k]) for k in ("D_sensor_train", "D_sensor_tune", "D_sensor_confirm"))
    assert not (a & b) and not (a & c) and not (b & c)


def test_C_was_selected_on_tune_not_confirm():
    fit = json.loads((RESULTS / "tables" / "sensor_confirm.json").read_text())
    assert fit["selected_on"] == "D_sensor_tune"
    best = max(fit["sweep"], key=lambda r: r["tune_auroc"])
    assert best["C"] == pytest.approx(fit["C_selected"])


def test_training_accuracy_is_never_the_evidence():
    fit = json.loads((RESULTS / "tables" / "sensor_confirm.json").read_text())
    assert fit["gate"]["checks"], "the gate must be evaluated on held-out confirm"
    assert fit["confirm"]["n"] == 85


def test_gate_thresholds_match_the_frozen_config():
    cfg = json.loads((ROOT / "configs/v3_4_0/sensor.json").read_text())["GATE"]
    fit = json.loads((RESULTS / "tables" / "sensor_confirm.json").read_text())
    assert fit["gate"]["thresholds"]["auroc_ci_lower_min"] == cfg["auroc_ci_lower_min"]
    assert fit["gate"]["thresholds"]["balanced_accuracy_min"] == cfg["balanced_accuracy_at_zero_min"]


def test_gate_actually_rejects_a_useless_sensor(rng):
    H, y = rng.normal(size=(200, 20)), (rng.random(200) > 0.5).astype(int)
    s = fit_sensor(H, y, C=0.01)
    m = sensor_metrics(s, rng.normal(size=(200, 20)), (rng.random(200) > 0.5).astype(int), y_train=y)
    gate = SensorGate().evaluate(m, {"auroc_ci95": [0.35, 0.62]})
    assert not gate["passed"]
