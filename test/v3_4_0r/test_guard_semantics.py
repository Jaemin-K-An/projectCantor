import numpy as np
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts/v3_4_0r"))
from analyse_failure_survival import classify_transition  # noqa: E402


def test_guard_is_not_a_direct_terminal_failure():
    """The guard is deliberately conservative; landing in one is not a failure."""
    assert classify_transition("leaf", 1, "guard", None) == "GUARD_CAPTURE"
    assert classify_transition("leaf", 7, "guard", None) != "RISKIER_TERMINAL"


def test_outside_window_is_its_own_category():
    assert classify_transition("leaf", 3, "outside", None) == "OUTSIDE_CAPTURE"
    assert classify_transition("leaf", 3, "outside", None) != "RISKIER_TERMINAL"


def test_guard_action_is_at_least_the_safer_neighbour(sensor, actuator, freeze):
    from cantor_guard_v340r.controllers import CappedCantorController

    W = float(freeze["inherited_frozen"]["W"])
    c = CappedCantorController(sensor=sensor, actuator=actuator, W=W, rho=1 / 3,
                               eta=1.0, q_cap=1.0,
                               leaf_actions=freeze["inherited_frozen"]["leaf_actions"])
    leaf_actions = np.asarray(freeze["inherited_frozen"]["leaf_actions"], dtype=float)
    for guard in c.inner.guards:
        left = [l for l in c.inner.leaves if l.hi <= guard.lo + 1e-15]
        right = [l for l in c.inner.leaves if l.lo >= guard.hi - 1e-15]
        neighbours = [leaf_actions[max(left, key=lambda x: x.hi).index]] if left else []
        if right:
            neighbours.append(leaf_actions[min(right, key=lambda x: x.lo).index])
        assert c.inner.action_for_cell(guard) == pytest.approx(max(neighbours))


def test_central_guard_holds_the_sensor_boundary(sensor, actuator, freeze):
    from cantor_guard_v340r.controllers import CappedCantorController

    W = float(freeze["inherited_frozen"]["W"])
    for rho in freeze["inherited_frozen"]["rho_family"]:
        c = CappedCantorController(sensor=sensor, actuator=actuator, W=W, rho=float(rho),
                                   eta=0.0, q_cap=0.05)
        on_plane = np.atleast_2d(sensor.project_to_hyperplane(np.ones(sensor.w.size)))
        assert c.correct(on_plane).cell_kind[0] == "guard"
