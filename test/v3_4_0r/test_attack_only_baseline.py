import json
import pathlib

import numpy as np

from cantor_guard_v340.attack import attack_w
from cantor_guard_v340.p0_generation import p0_attack_then_control

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_attack_only_arm_is_declared(freeze):
    assert "ATTACK_ONLY" in freeze["arms"]
    assert "CLEAN" in freeze["arms"]


def test_attack_only_applies_no_correction(sensor, actuator, rng):
    """The baseline V3.4.0 lacked: attacked state, controller absent."""
    from cantor_guard_v340r.controllers import CappedCantorController

    h = rng.normal(size=(20, sensor.w.size)) * 6
    delta = attack_w(sensor, 1.5, sign=-1)
    attacked = h + delta[None, :]
    # with no controller the state is exactly the attacked state
    assert np.allclose(np.atleast_1d(sensor.distance(attacked)),
                       np.atleast_1d(sensor.distance(h)) - 1.5)
    ctrl = CappedCantorController(sensor=sensor, actuator=actuator, W=2.2805,
                                  rho=1 / 3, eta=0.04, q_cap=0.05)
    # and a controller genuinely moves it, so the two arms are distinguishable
    assert not np.allclose(ctrl.correct(attacked).h_corrected, attacked)


def test_efficacy_contrast_is_against_the_baseline_not_other_rho():
    stats = json.loads((ROOT / "configs/v3_4_0r/statistics.json").read_text())
    contrasts = [tuple(c) for c in stats["controller_efficacy_contrasts"]]
    assert ("1/3", "ATTACK_ONLY") in contrasts
    assert ("LINEAR", "ATTACK_ONLY") in contrasts


def test_inertness_cannot_be_claimed_from_rho_similarity_alone():
    from patch_claim_classifier import controller_verdict

    # rho arms identical to each other says nothing; only the baseline contrast counts
    assert controller_verdict(interval_lo=None, interval_hi=None,
                              efficacy_sesoi=0.03) == "CTRL3_INCONCLUSIVE"
