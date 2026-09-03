import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
sys.path.insert(0, str(ROOT / "scripts/v3_4_0r"))

# Earlier version suites use the same top-level helper names.  Pytest keeps
# them in sys.modules across collection, so force V3.4.0R to import its own
# helpers after putting this version first on sys.path.
for _name in ("_common", "_stats", "patch_claim_classifier"):
    sys.modules.pop(_name, None)

RESULTS = ROOT / "results/v3_4_0r"
CONFIG = ROOT / "configs/v3_4_0r"


@pytest.fixture(scope="session")
def freeze():
    return json.loads((CONFIG / "PRE_ANALYSIS_FREEZE.json").read_text())


@pytest.fixture(scope="session")
def sensor():
    from _common import frozen_sensor

    return frozen_sensor()


@pytest.fixture(scope="session")
def actuator():
    from _common import frozen_actuator

    return frozen_actuator()


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(34000)
