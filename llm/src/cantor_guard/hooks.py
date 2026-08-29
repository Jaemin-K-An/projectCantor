"""Residual-stream read/write hooks.

CORRECTNESS REQUIREMENTS (harness §25, §49 items 9-10), each covered by a test:

* with the controller disabled the model's logits must be BIT-IDENTICAL to the
  unhooked model (`test_zero_intervention_identity`);
* during KV-cached generation the hook must fire exactly once per (token,
  layer) -- a hook that also fires on cached positions would apply the
  intervention repeatedly to the same residual;
* the write is applied to the BLOCK OUTPUT (the residual stream leaving layer
  l), which is what `v_ref` was estimated from.

We hook `decoder_layers(...)[l]` with a forward hook, so the modification lands
on the residual stream after layer l, before layer l+1 reads it.
"""
from __future__ import annotations
from contextlib import contextmanager
from typing import Callable, Iterable
import torch


def _split(out):
    """Decoder blocks return a Tensor or a tuple whose first element is it."""
    if isinstance(out, tuple):
        return out[0], out[1:]
    return out, None


def _join(h, rest):
    return h if rest is None else (h,) + rest


@contextmanager
def capture_residuals(bundle, layers: Iterable[int], store: dict,
                      *, last_token_only: bool = True):
    """Record the residual stream leaving each layer in `layers` into `store`.

    With `last_token_only` the stored tensor is `[batch, d_model]` taken at the
    final position, which for a left-padded batch is the last real token.
    """
    from .models import decoder_layers
    blocks = decoder_layers(bundle)
    handles = []

    def mk(l):
        def hook(_mod, _inp, out):
            h, _ = _split(out)
            store[l] = (h[:, -1, :] if last_token_only else h).detach().float().cpu()
            return out
        return hook

    try:
        for l in layers:
            handles.append(blocks[l].register_forward_hook(mk(l)))
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def steer_residuals(bundle, layers: Iterable[int],
                    delta_fn: Callable[[int, torch.Tensor], torch.Tensor],
                    *, counter: dict | None = None):
    """Add `delta_fn(layer, h)` to the residual stream leaving each layer.

    `delta_fn` receives the FULL `[batch, seq, d_model]` residual and must
    return a tensor broadcastable to it. During cached generation `seq == 1`
    after the prefill, so the function is called once per generated token per
    layer; `counter` (if given) records those call counts for the hook-firing
    test.
    """
    from .models import decoder_layers
    blocks = decoder_layers(bundle)
    handles = []

    def mk(l):
        def hook(_mod, _inp, out):
            h, rest = _split(out)
            d = delta_fn(l, h)
            if counter is not None:
                counter[l] = counter.get(l, 0) + int(h.shape[1])
            return _join(h + d.to(h.dtype), rest)
        return hook

    try:
        for l in layers:
            handles.append(blocks[l].register_forward_hook(mk(l)))
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def perturb_residuals(bundle, layers: Iterable[int], delta: torch.Tensor,
                      *, scale: float = 1.0):
    """A fixed additive latent ATTACK: h <- h + scale*delta at every layer in
    `layers`. Used for the controlled latent attack (`h <- h - eps*v_ref`)."""
    def fn(_l, h):
        return (scale * delta).to(h.device).view(1, 1, -1).expand_as(h)
    with steer_residuals(bundle, layers, fn):
        yield
