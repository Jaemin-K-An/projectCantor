import sys

import numpy as np
import pytest

sys.path.insert(0, "llm/src")
from cantor_guard_v335c.p0_cantor_controller import P0CantorSafetyController


def controller(rho=1 / 3, eta=0.14):
    return P0CantorSafetyController(
        v=np.array([1.0, 0.0, 0.0]), tau=0.0, W=3.0, rho=rho,
        eta=eta, safe_sign=1,
    )


def test_controller_consumes_actual_h_and_uses_projection_directly():
    h = np.array([[0.2, 4.0, 0.0], [-0.8, 1.0, 1.0]])
    result = controller().correct(h)
    np.testing.assert_allclose(result.z_observed, h[:, 0])
    assert result.h_corrected.shape == h.shape
    np.testing.assert_allclose(result.delta_h_controller[:, 1:], 0.0)
    assert not hasattr(controller(), "magnitude")


def test_same_tau_and_w_can_be_shared_across_rho():
    family = [controller(rho) for rho in (0.25, 0.28, 0.30, 1 / 3, 0.36, 0.40, 0.44)]
    assert {item.tau for item in family} == {0.0}
    assert {item.W for item in family} == {3.0}
    assert all(np.array_equal(item.leaf_actions, family[0].leaf_actions) for item in family)


def test_guard_action_is_conservative_relative_to_adjacent_leaves():
    item = controller()
    central = item.correct(np.array([0.0, 1.0, 0.0]))
    assert central.cells[0].kind == "guard"
    assert central.actions == pytest.approx(4 / 7)


def test_outside_window_uses_frozen_conservative_fallback():
    item = controller()
    result = item.correct(np.array([-10.0, 1.0, 0.0]))
    assert result.cells[0].kind == "outside"
    assert result.actions == 1.0
    assert result.q_ctrl == pytest.approx(item.eta)


def test_higher_risk_never_gets_weaker_leaf_action():
    item = controller()
    assert np.all(np.diff(item.leaf_actions) >= 0)


def test_certificate_is_independent_of_eta():
    from cantor_guard_v335c.cantor_geometry import epsilon_z
    a, b = controller(eta=0.01), controller(eta=0.9)
    assert epsilon_z(a.rho, a.W) == epsilon_z(b.rho, b.W)
