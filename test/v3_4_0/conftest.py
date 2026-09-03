import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))

RESULTS = ROOT / "results/v3_4_0"
CONFIG = ROOT / "configs/v3_4_0"


def _json(path):
    return json.loads(pathlib.Path(path).read_text())


@pytest.fixture(scope="session")
def sensor():
    from cantor_guard_v340.sensor_distance import SensorHyperplane

    fit = _json(RESULTS / "tables" / "sensor_confirm.json")
    return SensorHyperplane(np.load(RESULTS / "cache" / "sensor_w.npy"), float(fit["b"]))


@pytest.fixture(scope="session")
def actuator():
    from cantor_guard_v340.actuator import Actuator

    cfg = _json(CONFIG / "actuator.json")
    return Actuator(np.load(ROOT / cfg["direction_file"]).astype(float).reshape(-1), int(cfg["safe_sign"]))


@pytest.fixture(scope="session")
def freeze():
    return _json(CONFIG / "PRE_ANALYSIS_FREEZE.json")


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(20260903)
