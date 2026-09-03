"""V3.3.5b -- generation with a matched-budget TEMPORAL SCHEDULE.

State indexing, fixed and audited:
    forward 0 -> t = 0  (P0, last valid prompt token)
    forward 1 -> t = 1  (G1, first cached decode)
    forward k -> t = k

The perturbation at an active state is Delta h_t = s * q_t * ||h_t|| * v, so
the realised relative magnitude is exactly q_t and every schedule spends the
same trajectory L2 energy. Only P0 is applied at the mask-indexed last valid
prompt position; decode states have a single position.

Budget is NEVER redistributed after EOS: if generation stops early the unused
weight is simply not spent, and the realised B2 is recorded as such.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
import numpy as np, torch
from cantor_guard.models import decoder_layers
from cantor_guard_v31.generation31 import chat_prompt
from cantor_guard_v335a.p0_residual import last_valid_index

__all__ = ["TemporalTrace", "temporal_controlled", "generate_temporal"]


@dataclass
class TemporalTrace:
    forward_index: int = 0
    q_realised: dict = field(default_factory=dict)   # t -> per-item array
    z: dict = field(default_factory=dict)            # t -> per-item projection
    rows: list = field(default_factory=list)

    def reset(self):
        self.forward_index = 0
        self.q_realised.clear(); self.z.clear(); self.rows.clear()


@contextmanager
def temporal_controlled(bundle, *, v, layer, q_by_state, sign, last_idx,
                        trace=None, record=False):
    blocks = decoder_layers(bundle)
    vt = torch.tensor(np.asarray(v, np.float32), device=bundle.device)
    tr = trace or TemporalTrace()
    tr.reset()

    def hook(_m, _i, out):
        h, rest = (out[0], out[1:]) if isinstance(out, tuple) else (out, None)
        t = tr.forward_index
        hf = h.float()
        B = hf.shape[0]
        ar = torch.arange(B, device=hf.device)
        pos = last_idx if t == 0 else torch.full((B,), hf.shape[1] - 1,
                                                 device=hf.device, dtype=torch.long)
        row = hf[ar, pos, :]
        tr.z[t] = (row @ vt).detach().cpu().numpy()
        q = float(q_by_state.get(t, 0.0))
        applied = False
        if q != 0.0:
            nrm = torch.linalg.norm(row, dim=-1, keepdim=True)
            dh = float(sign) * q * nrm * vt.view(1, -1)
            row = row + dh
            tr.q_realised[t] = (torch.linalg.norm(dh, dim=-1)
                                / (nrm.squeeze(-1) + 1e-12)).detach().cpu().numpy()
            hf = hf.clone()
            hf[ar, pos, :] = row
            applied = True
        if record:
            tr.rows.append({"t": t, "seq_len": int(hf.shape[1]),
                            "q_target": q, "applied": applied})
        tr.forward_index += 1
        hn = hf.to(h.dtype)
        return hn if rest is None else (hn,) + rest

    hd = blocks[layer].register_forward_hook(hook)
    try:
        yield tr
    finally:
        hd.remove()


@torch.no_grad()
def generate_temporal(bundle, prompts, *, v, layer, q_by_state, sign=-1.0,
                      max_new_tokens=48, batch_size=10, record=False):
    tok = bundle.tokenizer
    texts, qs, b2r, b1r, traces = [], [], [], [], []
    for i in range(0, len(prompts), batch_size):
        ch = prompts[i:i + batch_size]
        enc = tok([chat_prompt(bundle, p) for p in ch], return_tensors="pt",
                  padding=True).to(bundle.device)
        idx = last_valid_index(enc["attention_mask"])
        tr = TemporalTrace()
        with temporal_controlled(bundle, v=v, layer=layer, q_by_state=q_by_state,
                                 sign=sign, last_idx=idx, trace=tr, record=record):
            out = bundle.model.generate(**enc, max_new_tokens=max_new_tokens,
                                        do_sample=False,
                                        pad_token_id=tok.pad_token_id)
        texts += tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
        if tr.q_realised:
            Q = np.stack([tr.q_realised[t] for t in sorted(tr.q_realised)], 1)
        else:
            Q = np.zeros((len(ch), 1))
        b2r.append(np.sqrt((Q ** 2).sum(1)))
        b1r.append(np.abs(Q).sum(1))
        qs.append(Q.max(1))
        if record:
            traces.append(list(tr.rows))
    return {"texts": texts, "B2_realised": np.concatenate(b2r),
            "B1_realised": np.concatenate(b1r), "q_max": np.concatenate(qs),
            "traces": traces}
