"""V3.3.5a PHASE 1 -- padding-safe P0 extraction and P0-only intervention.

P0 is the residual stream at the selected layer for the LAST NON-PADDING PROMPT
TOKEN, in the prefill forward, BEFORE the first output token is chosen.

    prompt ... -> [P0] -> logits(token 1) -> token 1 -> [G1] -> logits(token 2)
                   ^ forward 0                           ^ forward 1

V3.3.5 dosed G1, which is downstream of the choice of token 1 and therefore
cannot change it. P0 can. That is the whole correction.

PADDING. The historical extraction took h[:, -1, :], which is correct ONLY
because the tokenizer is configured with padding_side="left". That coupling is
implicit and fragile, so this module indexes with the attention mask and the
tests check both orientations explicitly.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
import numpy as np, torch
from cantor_guard.models import decoder_layers
from cantor_guard_v31.generation31 import chat_prompt

__all__ = ["last_valid_index", "last_valid_prompt_residuals", "P0Trace",
           "p0_controlled", "generate_p0_only", "PREFILL", "DECODE"]

PREFILL, DECODE = "PREFILL", "DECODE"


def last_valid_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of each row's final attended token. Correct for either padding side.

    For left padding this is always seq_len-1; for right padding it is
    sum(mask)-1. Computed from the mask so the caller never depends on the
    tokenizer's orientation.
    """
    m = attention_mask.to(torch.int64)
    idx = m.shape[1] - 1 - torch.flip(m, dims=[1]).argmax(dim=1)
    return idx


@torch.no_grad()
def last_valid_prompt_residuals(bundle, prompts, layer: int, *, batch_size=8):
    """P0 residuals, one row per prompt, mask-indexed."""
    tok = bundle.tokenizer
    blocks = decoder_layers(bundle)
    out = np.zeros((len(prompts), bundle.d_model), np.float32)
    for i in range(0, len(prompts), batch_size):
        ch = prompts[i:i + batch_size]
        enc = tok([chat_prompt(bundle, p) for p in ch], return_tensors="pt",
                  padding=True).to(bundle.device)
        idx = last_valid_index(enc["attention_mask"])
        store = {}

        def hook(_m, _i, o):
            h = o[0] if isinstance(o, tuple) else o
            store["h"] = h.float().detach()
            return o

        hd = blocks[layer].register_forward_hook(hook)
        try:
            bundle.model(**enc)
        finally:
            hd.remove()
        h = store["h"]
        out[i:i + len(ch)] = h[torch.arange(len(ch)), idx, :].cpu().numpy()
    return out


@dataclass
class P0Trace:
    forward_index: int = 0
    z_clean: np.ndarray | None = None
    z_attacked: np.ndarray | None = None
    z_corrected: np.ndarray | None = None
    q_p0: np.ndarray | None = None
    rows: list = field(default_factory=list)

    def reset(self):
        self.forward_index = 0
        self.z_clean = self.z_attacked = self.z_corrected = self.q_p0 = None
        self.rows.clear()

    def phase(self) -> str:
        return PREFILL if self.forward_index == 0 else DECODE


@contextmanager
def p0_controlled(bundle, *, v, layer, last_idx, controller=None, dose=0.0,
                  attack_eps=0.0, attack_sign=-1.0, trace=None, record=False):
    """Intervene ONLY at P0 (prefill forward, last valid prompt position).

    Order at P0 is clean -> attack -> controller correction, so the controller
    reacts to the attacked state, exactly as a deployed guard would.
    """
    blocks = decoder_layers(bundle)
    vt = torch.tensor(np.asarray(v, np.float32), device=bundle.device)
    tr = trace or P0Trace()
    tr.reset()

    def hook(_m, _i, out):
        h, rest = (out[0], out[1:]) if isinstance(out, tuple) else (out, None)
        phase = tr.phase()
        hf = h.float()
        acted = attacked = False
        if phase == PREFILL:
            B = hf.shape[0]
            ar = torch.arange(B, device=hf.device)
            row = hf[ar, last_idx, :]
            tr.z_clean = (row @ vt).detach().cpu().numpy()
            if dose != 0.0:
                row = row + float(dose) * vt
            if attack_eps > 0:
                row = row + float(attack_sign) * float(attack_eps) * vt
                attacked = True
            tr.z_attacked = (row @ vt).detach().cpu().numpy()
            if controller is not None:
                hl = row.detach().cpu().numpy()
                dh, mag = controller.intervene(hl, np.asarray(v, float))
                row = row + torch.tensor(dh, dtype=torch.float32, device=hf.device)
                tr.q_p0 = mag / (np.linalg.norm(hl, axis=-1) + 1e-12)
                acted = True
            tr.z_corrected = (row @ vt).detach().cpu().numpy()
            hf = hf.clone()
            hf[ar, last_idx, :] = row
        if record:
            tr.rows.append({"forward_index": tr.forward_index, "phase": phase,
                            "seq_len": int(h.shape[1]),
                            "attack_applied": attacked,
                            "controller_applied": acted})
        tr.forward_index += 1
        hn = hf.to(h.dtype)
        return hn if rest is None else (hn,) + rest

    hd = blocks[layer].register_forward_hook(hook)
    try:
        yield tr
    finally:
        hd.remove()


@torch.no_grad()
def generate_p0_only(bundle, prompts, *, v, layer, controller=None, dose=0.0,
                     attack_eps=0.0, attack_sign=-1.0, max_new_tokens=48,
                     batch_size=10, record=False, want_first_logits=False):
    tok = bundle.tokenizer
    texts, zc, za, zk, qs, traces, first = [], [], [], [], [], [], []
    for i in range(0, len(prompts), batch_size):
        ch = prompts[i:i + batch_size]
        enc = tok([chat_prompt(bundle, p) for p in ch], return_tensors="pt",
                  padding=True).to(bundle.device)
        idx = last_valid_index(enc["attention_mask"])
        tr = P0Trace()
        with p0_controlled(bundle, v=v, layer=layer, last_idx=idx,
                           controller=controller, dose=dose,
                           attack_eps=attack_eps, attack_sign=attack_sign,
                           trace=tr, record=record):
            if want_first_logits:
                lo = bundle.model(**enc).logits[torch.arange(len(ch)), idx, :]
                first.append(lo.float().cpu().numpy())
            tr.reset()
            with p0_controlled(bundle, v=v, layer=layer, last_idx=idx,
                               controller=controller, dose=dose,
                               attack_eps=attack_eps, attack_sign=attack_sign,
                               trace=tr, record=record):
                out = bundle.model.generate(**enc, max_new_tokens=max_new_tokens,
                                            do_sample=False,
                                            pad_token_id=tok.pad_token_id)
        texts += tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
        zc.append(tr.z_clean); za.append(tr.z_attacked); zk.append(tr.z_corrected)
        qs.append(tr.q_p0 if tr.q_p0 is not None else np.zeros(len(ch)))
        if record:
            traces.append(list(tr.rows))
    r = {"texts": texts, "z_clean": np.concatenate(zc),
         "z_attacked": np.concatenate(za), "z_corrected": np.concatenate(zk),
         "q_p0": np.concatenate(qs), "traces": traces}
    if want_first_logits:
        r["first_logits"] = np.concatenate(first, axis=0)
    return r
