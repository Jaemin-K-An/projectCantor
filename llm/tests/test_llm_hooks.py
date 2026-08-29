"""LLM-side correctness (harness §49 items 9-18). Requires a downloaded model.

These are the tests that make the intervention claims checkable:
  * zero intervention must be a bit-exact no-op
  * the hook must fire exactly once per (token, layer) under KV caching
  * v_ref must be unit norm and the threat coordinate monotone
  * the controller must always push the margin toward safety
  * tracked tables must never carry raw model text
"""
import numpy as np, pytest, torch

from cantor_guard.models import load_model, chat_prompt, decoder_layers, MODEL_REGISTRY
from cantor_guard.generation import controlled
from cantor_guard.probes import RefusalDirections
from cantor_guard.control_baselines import make_controller
from cantor_guard.threat_coordinate import threat_from_margin
from cantor_guard.io import assert_no_raw_completions

KEY = "qwen2.5-0.5b-instruct"


@pytest.fixture(scope="module")
def bundle():
    try:
        return load_model(KEY)
    except Exception as e:                                   # pragma: no cover
        pytest.skip(f"model unavailable: {e}")


@pytest.fixture(scope="module")
def dirs(bundle):
    rng = np.random.default_rng(0)
    layers = list(range(bundle.n_layers))
    v = rng.normal(size=(len(layers), bundle.d_model)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return RefusalDirections(layers=layers, v=v,
                             tau=np.zeros(len(layers), np.float32),
                             sigma=np.ones(len(layers), np.float32), method="dummy")


def _logits(bundle, text):
    enc = bundle.tokenizer(text, return_tensors="pt").to(bundle.device)
    with torch.no_grad():
        return bundle.model(**enc).logits.detach().float().cpu()


def test_v_ref_unit_norm(dirs):
    assert np.allclose(np.linalg.norm(dirs.v, axis=1), 1.0, atol=1e-6)


def test_zero_intervention_is_bit_identical(bundle, dirs):
    """controller=None and attack_eps=0 must leave the model untouched."""
    txt = chat_prompt(bundle, "Explain gravity in one sentence.")
    base = _logits(bundle, txt)
    layers = [4, 8, 12]
    with controlled(bundle, dirs=dirs, layers=layers, controller=None, attack_eps=0.0):
        hooked = _logits(bundle, txt)
    assert torch.equal(base, hooked), "hook altered the model with no intervention"


def test_zero_gain_controller_is_identity(bundle, dirs):
    """eta = 0 must also be an exact no-op, so 'no effect' is reachable."""
    txt = chat_prompt(bundle, "Name three primary colours.")
    base = _logits(bundle, txt)
    c = make_controller("L7_cantor", n=5, B_total=1.0, eta=0.0, gamma=1.0,
                        harm_gate=False)
    with controlled(bundle, dirs=dirs, layers=[6], controller=c, attack_eps=0.0):
        hooked = _logits(bundle, txt)
    assert torch.allclose(base, hooked, atol=0, rtol=0)


def test_nonzero_intervention_changes_output(bundle, dirs):
    """Sanity: a real intervention must actually change the logits."""
    txt = chat_prompt(bundle, "Name three primary colours.")
    base = _logits(bundle, txt)
    c = make_controller("L1_constant", n=5, B_total=1.0, eta=5.0, gamma=1.0,
                        harm_gate=False)
    with controlled(bundle, dirs=dirs, layers=[6], controller=c):
        hooked = _logits(bundle, txt)
    assert not torch.equal(base, hooked)


def test_hook_fires_once_per_token_per_layer(bundle, dirs):
    """Under KV caching the hook must see each token exactly once."""
    tok = bundle.tokenizer
    enc = tok(chat_prompt(bundle, "Count to three."), return_tensors="pt").to(bundle.device)
    n_prompt = enc["input_ids"].shape[1]
    n_new = 12
    layers = [3, 9]
    counter: dict = {}
    with controlled(bundle, dirs=dirs, layers=layers, controller=None,
                    call_counter=counter):
        with torch.no_grad():
            out = bundle.model.generate(**enc, max_new_tokens=n_new, do_sample=False,
                                        pad_token_id=tok.pad_token_id)
    produced = out.shape[1] - n_prompt
    for l in layers:
        # prefill sees n_prompt positions, then one position per generated token
        assert counter[l] == n_prompt + produced, (l, counter[l], n_prompt, produced)


def test_threat_coordinate_monotone_in_margin():
    m = np.linspace(-8, 8, 2001)
    for gamma in (0.5, 1.0, 3.0):
        r = threat_from_margin(m, gamma)
        assert (np.diff(r) < 0).all()                 # safer margin -> lower threat
        assert np.isclose(threat_from_margin(np.array([0.0]), gamma)[0], 0.5)
        assert r.min() > 0 and r.max() < 1


def test_controller_always_pushes_toward_safety():
    """c(r) >= 0 for every margin and every family: the intervention can never
    move the state toward the unsafe side."""
    m = np.linspace(-10, 10, 5001)
    for fam in ["L1_constant", "L2_central", "L3_periodic", "L5_shuffled",
                "L6_center_anchored", "L7_cantor"]:
        c = make_controller(fam, n=5, B_total=1.0, eta=1.0, gamma=1.0, seed=2)
        assert (c.magnitude(m) >= 0).all()


def test_intervention_norm_equals_magnitude(bundle, dirs):
    """v_ref is unit norm, so ||Delta h|| must equal c(r) exactly."""
    from cantor_guard.hooks import capture_residuals
    txt = chat_prompt(bundle, "Say hello.")
    enc = bundle.tokenizer(txt, return_tensors="pt").to(bundle.device)
    layer = 7
    a, b = {}, {}
    with capture_residuals(bundle, [layer], a):
        with torch.no_grad():
            bundle.model(**enc)
    c = make_controller("L1_constant", n=5, B_total=1.0, eta=2.0, gamma=1.0,
                        harm_gate=False)
    with controlled(bundle, dirs=dirs, layers=[layer], controller=c):
        with capture_residuals(bundle, [layer], b):
            with torch.no_grad():
                bundle.model(**enc)
    # capture runs after the steer hook on the same module, so b = a + c*v
    delta = (b[layer] - a[layer]).numpy()[0]
    i = dirs.layers.index(layer)
    z = float(a[layer].numpy()[0] @ dirs.v[i])
    expected = c.magnitude(np.array([[(z - dirs.tau[i]) / dirs.sigma[i]]]))[0, 0]
    assert np.isclose(np.linalg.norm(delta), expected, atol=1e-3), \
        (np.linalg.norm(delta), expected)


def test_harm_gate_masks_benign(bundle, dirs):
    """With harm_gate on, a benign-flagged item must receive no intervention."""
    txt = chat_prompt(bundle, "What is 2+2?")
    base = _logits(bundle, txt)
    c = make_controller("L1_constant", n=5, B_total=1.0, eta=5.0, gamma=1.0,
                        harm_gate=True)
    with controlled(bundle, dirs=dirs, layers=[6], controller=c,
                    harm_mask=np.array([False])):
        hooked = _logits(bundle, txt)
    assert torch.allclose(base, hooked, atol=1e-5)


def test_tracked_tables_reject_model_text():
    import pandas as pd
    ok = pd.DataFrame({"pid": ["a"], "asr": [0.1], "refusal": [1]})
    assert_no_raw_completions(ok)
    bad = pd.DataFrame({"pid": ["a"], "completion": ["harmful text"]})
    with pytest.raises(ValueError):
        assert_no_raw_completions(bad)
