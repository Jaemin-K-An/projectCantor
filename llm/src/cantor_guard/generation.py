"""Controlled generation: latent attack + CantorGuard intervention.

ORDER OF OPERATIONS, made explicit because it is a correctness requirement:
at each hooked layer, for each token position, we apply

    h  ->  h + attack_delta                      (the attack weakens refusal)
    m  =  (<h, v> - tau)/sigma                   (margin READ AFTER the attack)
    h  ->  h + c(r(m)) * v                       (the defence responds)

so the controller sees the ATTACKED state, which is the only setting in which
a state-dependent defence is meaningful. Both steps happen inside a single
forward hook so the order cannot silently change with registration order.

During cached generation the hook fires once per (token, layer): after the
prefill the residual has seq_len 1, so `c` is recomputed for every new token
from that token's own margin.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
import numpy as np
import torch

from .models import chat_prompt, decoder_layers
from .threat_coordinate import threat_from_margin


@dataclass
class RunStats:
    """Latent diagnostics accumulated during a controlled run (harness §33)."""
    margins: list = field(default_factory=list)     # per call: [B, S]
    magnitudes: list = field(default_factory=list)  # per call: [B, S]
    layer: int = -1

    def summarise(self, batch: int) -> list[dict]:
        """Per-batch-element summary of margin and intervention."""
        if not self.margins:
            return [{"m_min": np.nan, "m_mean": np.nan, "frac_m_neg": np.nan,
                     "int_mean": 0.0, "int_max": 0.0, "int_energy": 0.0}
                    for _ in range(batch)]
        M = np.concatenate(self.margins, axis=1)      # [B, T]
        C = np.concatenate(self.magnitudes, axis=1)   # [B, T]
        return [{"m_min": float(M[b].min()), "m_mean": float(M[b].mean()),
                 "frac_m_neg": float((M[b] < 0).mean()),
                 "int_mean": float(C[b].mean()), "int_max": float(C[b].max()),
                 "int_energy": float((C[b] ** 2).sum())} for b in range(batch)]


@contextmanager
def controlled(bundle, *, dirs, layers, controller=None, gamma=1.0,
               harm_mask=None, attack_eps=0.0, attack_layers=None,
               attack_ortho=0.0, attack_seed=0, stats: RunStats | None = None,
               call_counter: dict | None = None):
    """Context manager installing the attack+defence hook on `layers`.

    Parameters
    ----------
    dirs         : RefusalDirections (must cover every layer used here)
    layers       : layers at which the DEFENCE acts
    controller   : CantorGuardController, or None for no defence
    harm_mask    : bool array [B]; the defence is inert where False
    attack_eps   : latent attack strength (h <- h - eps*v)
    attack_layers: layers at which the ATTACK acts (default: same as `layers`)
    """
    blocks = decoder_layers(bundle)
    attack_layers = list(layers if attack_layers is None else attack_layers)
    L2I = {l: i for i, l in enumerate(dirs.layers)}
    rng = np.random.default_rng(attack_seed)

    vt, taut, sigt, atk = {}, {}, {}, {}
    for l in set(list(layers) + attack_layers):
        i = L2I[l]
        v = torch.tensor(dirs.v[i], dtype=torch.float32, device=bundle.device)
        vt[l] = v
        taut[l] = float(dirs.tau[i]); sigt[l] = float(dirs.sigma[i])
        if attack_eps > 0 and l in attack_layers:
            from .attacks import latent_attack_delta
            d = latent_attack_delta(dirs.v[i], attack_eps, rng=rng,
                                    ortho_frac=attack_ortho)
            atk[l] = torch.tensor(d, dtype=torch.float32, device=bundle.device)

    hm = None
    if harm_mask is not None:
        hm = torch.tensor(np.asarray(harm_mask, dtype=bool),
                          device=bundle.device)

    handles = []

    def mk(l):
        defend = (controller is not None) and (l in layers)
        def hook(_m, _i, out):
            h, rest = (out[0], out[1:]) if isinstance(out, tuple) else (out, None)
            hf = h.float()
            if l in atk:                                   # 1. attack
                hf = hf + atk[l].view(1, 1, -1)
            if defend:                                     # 2. read margin
                z = torch.einsum("bsd,d->bs", hf, vt[l])
                m = (z - taut[l]) / sigt[l]
                mnp = m.detach().cpu().numpy()
                c = controller.magnitude(mnp)              # 3. respond
                if hm is not None and controller.harm_gate:
                    c = c * hm.detach().cpu().numpy()[:, None]
                ct = torch.tensor(c, dtype=torch.float32, device=hf.device)
                hf = hf + ct.unsqueeze(-1) * vt[l].view(1, 1, -1)
                if stats is not None and l == (stats.layer if stats.layer >= 0
                                               else list(layers)[0]):
                    stats.margins.append(mnp); stats.magnitudes.append(c)
            if call_counter is not None:
                call_counter[l] = call_counter.get(l, 0) + int(h.shape[1])
            hnew = hf.to(h.dtype)
            return hnew if rest is None else (hnew,) + rest
        return hook

    try:
        for l in sorted(set(list(layers) + attack_layers)):
            handles.append(blocks[l].register_forward_hook(mk(l)))
        yield
    finally:
        for hd in handles:
            hd.remove()


@torch.no_grad()
def generate(bundle, prompts, *, max_new_tokens=64, batch_size=8, **kw):
    """Greedy generation under `controlled(...)`. Returns (texts, stats_rows).

    Greedy (do_sample=False) so that every controller is compared on exactly
    the same decoding path and differences cannot come from sampling noise.
    """
    tok = bundle.tokenizer
    texts, rows = [], []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok([chat_prompt(bundle, p) for p in chunk],
                  return_tensors="pt", padding=True).to(bundle.device)
        kw2 = dict(kw)
        if kw2.get("harm_mask") is not None:
            kw2["harm_mask"] = np.asarray(kw2["harm_mask"])[i:i + len(chunk)]
        st = RunStats()
        with controlled(bundle, stats=st, **kw2):
            out = bundle.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts += tok.batch_decode(gen, skip_special_tokens=True)
        rows += st.summarise(len(chunk))
    return texts, rows
