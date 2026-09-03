from types import SimpleNamespace
import sys

import numpy as np
import torch

sys.path.insert(0, "llm/src")
from cantor_guard_v335c.p0_attack_generation import P0AttackTrace, p0_attack_then_control
from cantor_guard_v335c.p0_cantor_controller import P0CantorSafetyController


def _bundle():
    block = torch.nn.Identity()
    model = SimpleNamespace(model=SimpleNamespace(layers=[block]))
    return SimpleNamespace(model=model, device="cpu", repo="fake", block=block)


def test_attack_occurs_before_controller_and_corrected_residual_is_returned():
    bundle = _bundle()
    h = torch.tensor([[[1.0, 2.0]]])
    controller = P0CantorSafetyController(
        v=[1, 0], tau=1.0, W=2.0, rho=1 / 3, eta=0.2, safe_sign=1,
    )
    trace = P0AttackTrace()
    with p0_attack_then_control(
        bundle, layer=0, v=[1, 0], last_idx=torch.tensor([0]),
        controller=controller, attack_epsilon=0.5, unsafe_sign=-1, trace=trace,
    ):
        corrected = bundle.block(h)
        decode = bundle.block(h)
    assert trace.z_clean[0] == 1.0
    assert trace.z_attacked[0] == 0.5
    expected = controller.correct(np.array([[0.5, 2.0]])).h_corrected
    np.testing.assert_allclose(corrected[0, 0].numpy(), expected[0], atol=1e-6)
    np.testing.assert_allclose(decode.numpy(), h.numpy())


def test_no_p0_intervention_leaks_into_g1_or_later():
    bundle = _bundle()
    h = torch.tensor([[[1.0, 2.0]]])
    trace = P0AttackTrace()
    with p0_attack_then_control(
        bundle, layer=0, v=[1, 0], last_idx=torch.tensor([0]),
        normalized_u=-0.2, trace=trace,
    ):
        first = bundle.block(h)
        second = bundle.block(h)
        third = bundle.block(h)
    assert not torch.allclose(first, h)
    assert torch.allclose(second, h) and torch.allclose(third, h)
    assert trace.phases == ["P0", "G1", "G2"]
