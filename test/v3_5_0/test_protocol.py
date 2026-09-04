import hashlib
import inspect
import json
import pathlib

from cantor_guard_v340 import p0_generation

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_exact_frozen_hashes():
    assert sha("results/v3_4_0/cache/sensor_w.npy") == "f16942ce8c6f89d2eaee2679da4778156450cd44fe1b9ac3529f3434f402f1fe"
    assert sha("results/v3_3_5a/cache/v_p0.npy") == "c22957e2fe05e9fa3bc158853dbb5c88965b62a98c2aefd63f11fa73d480172a"


def test_frozen_constants_and_architecture():
    cfg = load("configs/v3_5_0/controller.json")
    assert cfg["q_target_rms"] == .03
    assert cfg["q_cap"] == .05
    assert cfg["depth"] == 3
    assert cfg["rho_family"] == [.25, .28, .30, 1/3, .36, .40, .44]
    assert cfg["safe_side_action"] == 0
    assert cfg["outside_risk_action"] == 1
    assert cfg["budget_uses_outputs_or_labels"] is False


def test_fresh_split_sizes_and_no_overlap():
    cfg = load("configs/v3_5_0/splits.json")
    assert cfg["sizes"] == {"D_risk_window_cal": 300, "D_budget_v350": 300,
                             "D_attack_dev_v350": 50, "D_final_v350_harmful": 200,
                             "D_final_v350_benign": 80}
    ids = [x for block in cfg["blocks"].values() for x in block]
    assert len(ids) == len(set(ids)) == 930
    audit = load("results/v3_5_0/tables/historical_leakage_audit.json")
    assert audit["passed"] and audit["historical_exact_overlap"] == 0


def test_attack_precedes_controller_at_p0_and_changes_logits_path():
    src = inspect.getsource(p0_generation.p0_attack_then_control)
    assert src.index("row = row + delta") < src.index("controller.policy_record(attacked)")
    assert "hf[arange, last_idx, :] = row" in src
    assert "if tr[\"forward\"] == 0" in src


def test_final_runner_contains_true_baseline_and_refuses_without_freeze():
    src = (ROOT / "scripts/v3_5_0/run_final_harmful.py").read_text()
    assert "require_freeze()" in src
    assert '"ATTACK_ONLY"' in (ROOT / "scripts/v3_5_0/_common.py").read_text()
    assert '"LINEAR"' in (ROOT / "scripts/v3_5_0/_common.py").read_text()


def test_no_final_outputs_exist_after_budget_hard_stop():
    freeze = load("configs/v3_5_0/PRE_ANALYSIS_FREEZE.json")
    assert freeze["status"] == "NOT_FROZEN_BUDGET_IMPOSSIBLE"
    assert freeze["D_final_v350_touched"] is False
    assert not (ROOT / "results/v3_5_0/raw/final_D_final_v350_harmful.csv").exists()
    assert not (ROOT / "results/v3_5_0/raw/utility_D_final_v350_benign.csv").exists()
