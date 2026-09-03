import numpy as np
import pytest

from cantor_guard_v340.cantor_geometry import classify, partition


def test_higher_leaf_index_means_higher_risk(sensor, actuator, freeze):
    """r = 1/2 - d/(2W): d decreasing (riskier) drives r up, so index rises.

    V3.4.0's failure scan tested new_index < base_index, which is backwards.
    """
    from cantor_guard_v340r.controllers import CappedCantorController

    W = float(freeze["inherited_frozen"]["W"])
    c = CappedCantorController(sensor=sensor, actuator=actuator, W=W, rho=1 / 3,
                               eta=0.0, q_cap=0.05)
    base = sensor.project_to_hyperplane(np.ones(sensor.w.size))
    # offsets chosen so both land in terminal LEAVES (r ~ 0.005 and 0.995),
    # not in the guards that sit just inside them
    safe = np.atleast_2d(base + 0.99 * W * sensor.w_hat)     # large positive d
    risky = np.atleast_2d(base - 0.99 * W * sensor.w_hat)    # large negative d
    r_safe = c.inner.risk_coordinate(np.atleast_1d(sensor.distance(safe)))[0]
    r_risky = c.inner.risk_coordinate(np.atleast_1d(sensor.distance(risky)))[0]
    assert r_risky > r_safe
    cell_safe, cell_risky = classify(r_safe, 1 / 3, 3), classify(r_risky, 1 / 3, 3)
    assert cell_safe.kind == "leaf" and cell_risky.kind == "leaf"
    assert cell_risky.index > cell_safe.index


def test_leaf_indices_are_ordered_by_risk_coordinate():
    leaves, _ = partition(1 / 3, 3)
    assert [c.index for c in leaves] == list(range(8))
    assert all(leaves[i].lo < leaves[i + 1].lo for i in range(7))


def test_classifier_helper_uses_the_corrected_direction():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts/v3_4_0r"))
    from analyse_failure_survival import classify_transition

    assert classify_transition("leaf", 2, "leaf", 5) == "RISKIER_TERMINAL"
    assert classify_transition("leaf", 5, "leaf", 2) == "SAFER_TERMINAL"
    assert classify_transition("leaf", 3, "leaf", 3) == "SAME_TERMINAL"


def test_higher_risk_gets_a_stronger_action(sensor, actuator, freeze):
    from cantor_guard_v340r.controllers import CappedCantorController

    W = float(freeze["inherited_frozen"]["W"])
    c = CappedCantorController(sensor=sensor, actuator=actuator, W=W, rho=1 / 3,
                               eta=1.0, q_cap=1.0,
                               leaf_actions=freeze["inherited_frozen"]["leaf_actions"])
    base = sensor.project_to_hyperplane(np.ones(sensor.w.size))
    safe = np.atleast_2d(base + 0.95 * W * sensor.w_hat)
    risky = np.atleast_2d(base - 0.95 * W * sensor.w_hat)
    assert c.correct(risky).actions[0] > c.correct(safe).actions[0]
