import json
import pathlib
import re


HEX16 = re.compile(r"^[0-9a-f]{16}$")


def _strings(value):
    if isinstance(value, dict):
        for nested in value.values():
            yield from _strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _strings(nested)
    elif isinstance(value, str):
        yield value


def test_all_v335c_blocks_are_pairwise_disjoint_and_fresh():
    current = json.loads(pathlib.Path("configs/v3_3_5c/splits.json").read_text())
    ids = [pid for block in current["blocks"].values() for pid in block]
    assert len(ids) == len(set(ids))
    prior = set()
    for path in pathlib.Path("configs").rglob("*.json"):
        if "v3_3_5c" in path.parts:
            continue
        try:
            prior |= {value for value in _strings(json.loads(path.read_text())) if HEX16.fullmatch(value)}
        except json.JSONDecodeError:
            pass
    assert not (set(ids) & prior)
    assert current["hash_overlap_with_all_prior"] == 0


def test_fresh_final_and_benign_sizes_meet_protocol():
    current = json.loads(pathlib.Path("configs/v3_3_5c/splits.json").read_text())
    assert len(current["blocks"]["D_final_P0_335c"]) >= 80
    assert len(current["blocks"]["D_benign_P0_335c"]) >= 50


def test_failed_boundary_keeps_final_untouched_and_protocol_unfrozen():
    freeze = json.loads(pathlib.Path("configs/v3_3_5c/PRE_ANALYSIS_FREEZE.json").read_text())
    controller = json.loads(pathlib.Path("configs/v3_3_5c/controller.json").read_text())
    attack = json.loads(pathlib.Path("configs/v3_3_5c/attack_grid.json").read_text())
    assert freeze["status"] == "NOT_FROZEN_BEHAVIORAL_GATE_FAILED"
    assert freeze["D_final_touched"] is False
    assert controller["tau"] is None
    assert controller["eta_by_rho"] is None
    assert attack["common_absolute_epsilon_grid"] is None
    assert not pathlib.Path("results/v3_3_5c/raw/final_p0_cantor.csv").exists()
    assert not pathlib.Path("results/v3_3_5c/private/final_p0_cantor_completions.csv").exists()
