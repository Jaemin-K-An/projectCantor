import importlib
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/v3_4_0"
sys.path.insert(0, str(ROOT / "scripts/v3_4_0"))


@pytest.fixture()
def classifier(monkeypatch):
    module = importlib.import_module("final_claim_check_v340")
    importlib.reload(module)
    return module


def run_with(module, monkeypatch, tables):
    monkeypatch.setattr(module, "_maybe", lambda name: tables.get(name))
    captured = {}
    monkeypatch.setattr(module, "write_json", lambda path, payload: captured.update(payload))
    module.main()
    return captured


def _full(**overrides):
    tables = {
        "sensor_confirm.json": {"gate": {"passed": True}},
        "semantic_evaluator_validation.json": {"verdict": "SEM3_PROXY_ONLY"},
        "actuator_validation.json": {"verdict": "ACT1_CAUSAL_ACTUATOR_REPLICATED"},
        "sensor_window_and_coupling.json": {"controllability": {"verdict": "COUP1_CONTROLLABLE"}},
        "certificate_validation.json": {"verdict": "CERT1_CANTOR_SENSOR_CERTIFICATE_VALID"},
        "final_budget_audit.json": {"all_matched": True},
        "utility.json": {"verdict": "U1_PASS"},
        "generation_analysis.json": {"by_family": {"ATTACK_V": {
            "max_t": {"contrasts": [{"a": 1}]}, "all_favour_cantor": False,
            "any_favours_other": False, "all_within_sesoi": True}}},
    }
    tables.update(overrides)
    return tables


def test_sensor_failure_dominates_everything(classifier, monkeypatch):
    out = run_with(classifier, monkeypatch, _full(**{"sensor_confirm.json": {"gate": {"passed": False}}}))
    assert out["SENSOR"] == "SENS3_SENSOR_NOT_GENERALIZABLE"
    assert out["OVERALL"] == "E_LINEAR_BEHAVIORAL_SENSOR_NOT_SUPPORTED"


def test_weak_coupling_gives_verdict_f(classifier, monkeypatch):
    out = run_with(classifier, monkeypatch, _full(**{
        "sensor_window_and_coupling.json": {"controllability": {"verdict": "COUP2_WEAK_SENSOR_ACTUATOR_COUPLING"}}}))
    assert out["OVERALL"] == "F_SENSOR_VALID_ACTUATOR_COUPLING_TOO_WEAK"


def test_semantic_claim_is_blocked_without_a_valid_evaluator(classifier, monkeypatch):
    out = run_with(classifier, monkeypatch, _full())
    assert out["SEMANTIC"] == "SEM3_PROXY_ONLY"
    assert out["semantic_claim_allowed"] is False
    assert out["SENSOR"] == "SENS2_REFUSAL_SENSOR_ONLY"
    assert out["GENERATION"] != "GEN1_CANTOR_SEMANTIC_GAIN"


def test_cantor_win_without_evaluator_is_refusal_only(classifier, monkeypatch):
    out = run_with(classifier, monkeypatch, _full(**{"generation_analysis.json": {"by_family": {
        "ATTACK_V": {"max_t": {"contrasts": [{"a": 1}]}, "all_favour_cantor": True,
                     "any_favours_other": False, "all_within_sesoi": False}}}}))
    assert out["GENERATION"] == "GEN4_REFUSAL_ONLY_RESULT"


def test_other_rho_better_is_reported_not_hidden(classifier, monkeypatch):
    out = run_with(classifier, monkeypatch, _full(**{"generation_analysis.json": {"by_family": {
        "ATTACK_V": {"max_t": {"contrasts": [{"a": 1}]}, "all_favour_cantor": False,
                     "any_favours_other": True, "all_within_sesoi": False}}}}))
    assert out["GENERATION"] == "GEN3_OTHER_RHO_BETTER"


def test_budget_mismatch_blocks_the_strong_verdict(classifier, monkeypatch):
    out = run_with(classifier, monkeypatch, _full(**{"final_budget_audit.json": {"all_matched": False}}))
    assert out["architecture_complete"] is False
    assert out["OVERALL"] == "G_INCONCLUSIVE"


def test_utility_failure_blocks_the_strong_verdict(classifier, monkeypatch):
    out = run_with(classifier, monkeypatch, _full(**{"utility.json": {"verdict": "U2_FAIL"}}))
    assert out["OVERALL"] == "G_INCONCLUSIVE"


def test_architecture_supported_when_every_gate_passes(classifier, monkeypatch):
    out = run_with(classifier, monkeypatch, _full())
    assert out["OVERALL"] == "A_SENSOR_ACTUATOR_CANTOR_CONTROLLER_SUPPORTED"
    assert out["architecture_complete"] is True


def test_llm_is_never_called_fractal(classifier, monkeypatch):
    out = run_with(classifier, monkeypatch, _full())
    assert out["llm_is_not_claimed_to_be_fractal"] is True
    assert "conditional" in json.dumps(out).lower()
