"""V3.2 generation -- extends the V3.1 hook without modifying it.

`cantor_guard_v31.generation31` is left byte-identical so the frozen V3.1
results stay reproducible. Two things are added here:

  1. MARGIN RECORDING UNDER NO CONTROL. The V3.1 probe only records when a
     controller is active, so the uncontrolled arm carried no margin
     distribution at all and `m_min` was a NaN placeholder. The baseline
     margin distribution is exactly what the calibration shift Delta is
     expressed in, so it needs to be observable.
  2. A separate `gen_seed`, so the layout instance and the generation
     replicate stop sharing one field (V3.1 defect D3).
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
import numpy as np, torch
from cantor_guard.models import decoder_layers
from cantor_guard_v31.generation31 import chat_prompt


@dataclass
class Probe32:
    q: list = field(default_factory=list)
    m: list = field(default_factory=list)
    c: list = field(default_factory=list)

    def stats(self):
        m = (np.concatenate([x.ravel() for x in self.m]) if self.m
             else np.zeros(0))
        if not self.q:
            base = {"C_mean": 0.0, "C_rms": 0.0, "C95": 0.0, "C_max": 0.0}
        else:
            q = np.concatenate([x.ravel() for x in self.q])
            base = {"C_mean": float(q.mean()),
                    "C_rms": float(np.sqrt((q ** 2).mean())),
                    "C95": float(np.quantile(q, .95)), "C_max": float(q.max())}
        if len(m):
            base.update(m_min=float(m.min()), m_mean=float(m.mean()),
                        m_std=float(m.std()), m_q05=float(np.quantile(m, .05)),
                        m_q95=float(np.quantile(m, .95)))
        else:
            base.update(m_min=np.nan, m_mean=np.nan, m_std=np.nan,
                        m_q05=np.nan, m_q95=np.nan)
        return base


@contextmanager
def controlled32(bundle, *, dirs, layers, controller, delta=0.0,
                 attack_eps=0.0, attack_ortho=0.3, attack_seed=0,
                 harm_mask=None, probe: Probe32 | None = None):
    blocks = decoder_layers(bundle)
    L2I = {l: dirs.layers.index(l) for l in layers}
    rng = np.random.default_rng(attack_seed)
    vt, taut, sigt, atk = {}, {}, {}, {}
    for l in layers:
        i = L2I[l]
        vt[l] = torch.tensor(dirs.v[i], dtype=torch.float32, device=bundle.device)
        taut[l], sigt[l] = float(dirs.tau[i]), float(dirs.sigma[i])
        if attack_eps > 0:
            from cantor_guard.attacks import latent_attack_delta
            atk[l] = torch.tensor(
                latent_attack_delta(dirs.v[i], attack_eps, rng=rng,
                                    ortho_frac=attack_ortho),
                dtype=torch.float32, device=bundle.device)
    hm = None if harm_mask is None else torch.tensor(
        np.asarray(harm_mask, bool), device=bundle.device)
    active = controller is not None and controller.family != "T0_none"
    handles = []

    def mk(l):
        def hook(_m, _i, out):
            h, rest = (out[0], out[1:]) if isinstance(out, tuple) else (out, None)
            hf = h.float()
            if l in atk:
                hf = hf + atk[l].view(1, 1, -1)
            # Margins are read whenever a probe is attached, control or not.
            if probe is not None or active:
                z = torch.einsum("bsd,d->bs", hf, vt[l])
                m = ((z - taut[l]) / sigt[l]).detach().cpu().numpy()
            if active:
                c = controller.magnitude(m - delta)          # BOUNDARY ERROR
                if hm is not None and controller.harm_gate:
                    c = c * hm.detach().cpu().numpy()[:, None]
                nrm = torch.linalg.norm(hf, dim=-1).detach().cpu().numpy()
                if controller.max_q is not None:
                    c = np.minimum(c, controller.max_q * nrm)
                if probe is not None:
                    probe.q.append(c / (nrm + 1e-12))
                    probe.m.append(m); probe.c.append(c)
                ct = torch.tensor(c, dtype=torch.float32, device=hf.device)
                hf = hf + ct.unsqueeze(-1) * vt[l].view(1, 1, -1)
            elif probe is not None:
                probe.m.append(m)          # uncontrolled: margins only, q = 0
            hn = hf.to(h.dtype)
            return hn if rest is None else (hn,) + rest
        return hook

    try:
        for l in layers:
            handles.append(blocks[l].register_forward_hook(mk(l)))
        yield
    finally:
        for hd in handles:
            hd.remove()


@torch.no_grad()
def generate32(bundle, prompts, *, max_new_tokens=48, batch_size=12,
               gen_seed: int | None = None, **kw):
    """Greedy decoding, so `gen_seed` only fixes the latent-attack noise draw."""
    tok = bundle.tokenizer
    texts, stats = [], []
    for i in range(0, len(prompts), batch_size):
        ch = prompts[i:i + batch_size]
        enc = tok([chat_prompt(bundle, p) for p in ch], return_tensors="pt",
                  padding=True).to(bundle.device)
        k2 = dict(kw)
        if k2.get("harm_mask") is not None:
            k2["harm_mask"] = np.asarray(k2["harm_mask"])[i:i + len(ch)]
        if gen_seed is not None:
            k2["attack_seed"] = int(gen_seed)
        pr = Probe32()
        with controlled32(bundle, probe=pr, **k2):
            out = bundle.model.generate(**enc, max_new_tokens=max_new_tokens,
                                        do_sample=False,
                                        pad_token_id=tok.pad_token_id)
        texts += tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
        stats.append(pr.stats())
    agg = ({k: float(np.nanmean([s[k] for s in stats])) for k in stats[0]}
           if stats else {})
    return texts, agg
