"""V3.1 controlled generation: attack, boundary error, defence, budget probe.

Order inside a single hook (so registration order cannot change it):
    h <- h + attack_delta                 (epsilon: weakens refusal)
    m <- (<h,v> - tau)/sigma              (margin read AFTER the attack)
    m_hat <- m - Delta                    (the controller's BOUNDARY ERROR)
    h <- h + c(m_hat) * v                 (defence responds to what it believes)

Delta is a calibration error, not an attack: it changes only what the
controller believes about the boundary, never the true state (harness §16).
Every applied update's q = ||dh||/||h|| is recorded so the realised budget can
be matched across controllers.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
import numpy as np
import torch
from cantor_guard.models import chat_prompt, decoder_layers


@dataclass
class Probe:
    q: list = field(default_factory=list)        # ||dh||/||h|| per (batch,token)
    m: list = field(default_factory=list)
    c: list = field(default_factory=list)

    def stats(self):
        if not self.q:
            return {"C_mean": 0.0, "C_rms": 0.0, "C95": 0.0, "C_max": 0.0,
                    "m_min": np.nan}
        q = np.concatenate([x.ravel() for x in self.q])
        m = np.concatenate([x.ravel() for x in self.m])
        return {"C_mean": float(q.mean()), "C_rms": float(np.sqrt((q**2).mean())),
                "C95": float(np.quantile(q, .95)), "C_max": float(q.max()),
                "m_min": float(m.min())}


@contextmanager
def controlled31(bundle, *, dirs, layers, controller, delta=0.0,
                 attack_eps=0.0, attack_ortho=0.3, attack_seed=0,
                 harm_mask=None, probe: Probe | None = None):
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
    handles = []

    def mk(l):
        def hook(_m, _i, out):
            h, rest = (out[0], out[1:]) if isinstance(out, tuple) else (out, None)
            hf = h.float()
            if l in atk:
                hf = hf + atk[l].view(1, 1, -1)
            if controller is not None and controller.family != "T0_none":
                z = torch.einsum("bsd,d->bs", hf, vt[l])
                m = ((z - taut[l]) / sigt[l]).detach().cpu().numpy()
                c = controller.magnitude(m - delta)          # BOUNDARY ERROR
                if hm is not None and controller.harm_gate:
                    c = c * hm.detach().cpu().numpy()[:, None]
                if controller.max_q is not None:
                    nrm = torch.linalg.norm(hf, dim=-1).detach().cpu().numpy()
                    c = np.minimum(c, controller.max_q * nrm)
                ct = torch.tensor(c, dtype=torch.float32, device=hf.device)
                if probe is not None:
                    nrm = torch.linalg.norm(hf, dim=-1).detach().cpu().numpy()
                    probe.q.append(c / (nrm + 1e-12)); probe.m.append(m); probe.c.append(c)
                hf = hf + ct.unsqueeze(-1) * vt[l].view(1, 1, -1)
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
def generate31(bundle, prompts, *, max_new_tokens=48, batch_size=12, **kw):
    tok = bundle.tokenizer
    texts, stats = [], []
    for i in range(0, len(prompts), batch_size):
        ch = prompts[i:i + batch_size]
        enc = tok([chat_prompt(bundle, p) for p in ch], return_tensors="pt",
                  padding=True).to(bundle.device)
        k2 = dict(kw)
        if k2.get("harm_mask") is not None:
            k2["harm_mask"] = np.asarray(k2["harm_mask"])[i:i + len(ch)]
        pr = Probe()
        with controlled31(bundle, probe=pr, **k2):
            out = bundle.model.generate(**enc, max_new_tokens=max_new_tokens,
                                        do_sample=False, pad_token_id=tok.pad_token_id)
        texts += tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
        stats.append(pr.stats())
    agg = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]} if stats else {}
    return texts, agg
