"""V3.3.2 PHASE 4/9 -- genuinely phase-aware generation.

Two calibrations, applied where they belong:

    PREFILL forward  ->  tau_prompt,     sigma_prompt
    DECODE forwards  ->  tau_generation, sigma_generation

`mode` selects the regime so the old behaviour stays reproducible:
    "phase_aware"           the correction
    "legacy_prompt_only"    V3.2: prompt calibration everywhere
    "generation_only"       V3.3.1: generation calibration everywhere
The legacy modes exist for the calibration-mode factor of the mechanism
experiment, not as defaults.
"""
from __future__ import annotations
from contextlib import contextmanager
import numpy as np, torch
from cantor_guard.models import decoder_layers
from cantor_guard_v31.generation31 import chat_prompt
from .phase_state import PhaseState, PREFILL, DECODE

MODES = ("phase_aware", "legacy_prompt_only", "generation_only")


class Probe332:
    def __init__(self):
        self.q, self.m, self.phase = [], [], []

    def stats(self):
        m = np.concatenate([x.ravel() for x in self.m]) if self.m else np.zeros(0)
        if self.q:
            q = np.concatenate([x.ravel() for x in self.q])
            base = {"C_mean": float(q.mean()),
                    "C_rms": float(np.sqrt((q ** 2).mean())),
                    "C95": float(np.quantile(q, .95)), "C_max": float(q.max())}
        else:
            base = {"C_mean": 0.0, "C_rms": 0.0, "C95": 0.0, "C_max": 0.0}
        if len(m):
            base.update(m_mean=float(m.mean()), m_std=float(m.std()),
                        m_min=float(m.min()))
        else:
            base.update(m_mean=np.nan, m_std=np.nan, m_min=np.nan)
        return base


@contextmanager
def phase_controlled(bundle, *, dirs_prompt, dirs_generation, layer, controller,
                     mode="phase_aware", delta=0.0, attack_eps=0.0,
                     attack_ortho=0.3, attack_seed=0, harm_mask=None,
                     probe=None, state=None):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    blocks = decoder_layers(bundle)
    st = state or PhaseState()
    st.reset()
    i_p = dirs_prompt.layers.index(layer)
    i_g = dirs_generation.layers.index(layer)
    v = torch.tensor(dirs_prompt.v[i_p], dtype=torch.float32, device=bundle.device)
    TAU = {PREFILL: float(dirs_prompt.tau[i_p]), DECODE: float(dirs_generation.tau[i_g])}
    SIG = {PREFILL: float(dirs_prompt.sigma[i_p]), DECODE: float(dirs_generation.sigma[i_g])}
    if mode == "legacy_prompt_only":
        TAU[DECODE], SIG[DECODE] = TAU[PREFILL], SIG[PREFILL]
    elif mode == "generation_only":
        TAU[PREFILL], SIG[PREFILL] = TAU[DECODE], SIG[DECODE]

    atk = None
    if attack_eps > 0:
        from cantor_guard.attacks import latent_attack_delta
        rng = np.random.default_rng(attack_seed)
        atk = torch.tensor(latent_attack_delta(dirs_prompt.v[i_p], attack_eps,
                                               rng=rng, ortho_frac=attack_ortho),
                           dtype=torch.float32, device=bundle.device)
    hm = None if harm_mask is None else torch.tensor(np.asarray(harm_mask, bool),
                                                     device=bundle.device)
    active = controller is not None and getattr(controller, "family", "") != "T0_none"
    handles = []

    def hook(_m, inp, out):
        h, rest = (out[0], out[1:]) if isinstance(out, tuple) else (out, None)
        seq_len = h.shape[1]
        has_cache = False
        try:                       # cross-check only; never drives the phase
            kw = _m._v332_last_kwargs if hasattr(_m, "_v332_last_kwargs") else {}
            has_cache = bool(kw.get("past_key_value") is not None)
        except Exception:
            pass
        phase = st.observe(seq_len, has_cache or seq_len == 1)
        hf = h.float()
        if atk is not None:
            hf = hf + atk.view(1, 1, -1)
        if probe is not None or active:
            z = torch.einsum("bsd,d->bs", hf, v)
            m = ((z - TAU[phase]) / SIG[phase]).detach().cpu().numpy()
        if active:
            c = controller.magnitude(m - delta)
            if hm is not None and getattr(controller, "harm_gate", False):
                c = c * hm.detach().cpu().numpy()[:, None]
            nrm = torch.linalg.norm(hf, dim=-1).detach().cpu().numpy()
            if getattr(controller, "max_q", None) is not None:
                c = np.minimum(c, controller.max_q * nrm)
            if probe is not None:
                probe.q.append(c / (nrm + 1e-12)); probe.m.append(m)
                probe.phase.append(phase)
            hf = hf + torch.tensor(c, dtype=torch.float32,
                                   device=hf.device).unsqueeze(-1) * v.view(1, 1, -1)
        elif probe is not None:
            probe.m.append(m); probe.phase.append(phase)
        hn = hf.to(h.dtype)
        return hn if rest is None else (hn,) + rest

    try:
        handles.append(blocks[layer].register_forward_hook(hook))
        yield st
    finally:
        for hd in handles:
            hd.remove()


@torch.no_grad()
def generate332(bundle, prompts, *, dirs_prompt, dirs_generation, layer,
                controller=None, mode="phase_aware", max_new_tokens=48,
                batch_size=10, record_trace=False, **kw):
    tok = bundle.tokenizer
    texts, stats, traces = [], [], []
    for i in range(0, len(prompts), batch_size):
        ch = prompts[i:i + batch_size]
        enc = tok([chat_prompt(bundle, p) for p in ch], return_tensors="pt",
                  padding=True).to(bundle.device)
        k2 = dict(kw)
        if k2.get("harm_mask") is not None:
            k2["harm_mask"] = np.asarray(k2["harm_mask"])[i:i + len(ch)]
        pr, stt = Probe332(), PhaseState(record_trace=record_trace)
        with phase_controlled(bundle, dirs_prompt=dirs_prompt,
                              dirs_generation=dirs_generation, layer=layer,
                              controller=controller, mode=mode, probe=pr,
                              state=stt, **k2):
            out = bundle.model.generate(**enc, max_new_tokens=max_new_tokens,
                                        do_sample=False,
                                        pad_token_id=tok.pad_token_id)
        texts += tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
        stats.append(pr.stats())
        if record_trace:
            traces.append(stt.consistency())
    agg = ({k: float(np.nanmean([s[k] for s in stats])) for k in stats[0]}
           if stats else {})
    return (texts, agg, traces) if record_trace else (texts, agg)
