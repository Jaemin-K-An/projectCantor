import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path("scripts/v3_3_5c").resolve()))
from fit_p0_controller_budgets import eta_for_target


def test_eta_hits_rms_target_mechanically():
    actions = np.array([0, 1 / 7, 4 / 7, 1.0])
    target = 0.1
    eta = eta_for_target(actions, target)
    assert np.sqrt(np.mean((eta * actions) ** 2)) == target


def test_one_target_is_shared_by_every_rho_in_config():
    import json
    config = json.loads(pathlib.Path("configs/v3_3_5c/controller.json").read_text())
    assert isinstance(config["q_target"], (float, type(None)))
    assert "q_target_by_rho" not in config
    assert config["budget_tolerance_relative"] == 0.03
