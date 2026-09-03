"""V3.3.5 PHASE 4/5/10 -- G1-ONLY dose / attack / controller.

V3.3.4 DEFECT (D) this fixes. generate332 applied the controller on EVERY
forward once active -- prefill and all decode steps. So the boundary was
estimated at one state and the policy acted at many, and the prefill margin was
computed with a different calibration than the controller assumed.

Here exactly one residual state is touched:

    forward 0    PREFILL   nothing
    forward 1    G1        attack, then controller (in that order)
    forward >=2  G2PLUS    nothing

G1 is the FIRST TRUE CACHED DECODE FORWARD after prefill -- the forward whose
input is the first generated token, not the prefill pass that emits it.

ORDER AT G1 (harness section 27): attack first, then the controller classifies
the ATTACKED state and corrects. The controller reacts to the attack; it does
not get to see the clean state.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
import numpy as np, torch
from cantor_guard.models import decoder_layers
from cantor_guard_v31.generation31 import chat_prompt

__all__ = ["G1Trace", "g1_controlled", "generate_g1_only",
           "PREFILL", "G1", "G2PLUS"]

PREFILL, G1, G2PLUS = "PREFILL", "G1", "G2PLUS"


@dataclass
class G1Trace:
    rows: list = field(default_factory=list)
    z_clean: np.ndarray | None = None
    z_attacked: np.ndarray | None = None
    q_g1: np.ndarray | None = None
    forward_index: int = 0

    def reset(self):
        self.rows.clear(); self.forward_index = 0
        self.z_clean = self.z_attacked = self.q_g1 = None

    def phase(self) -> str:
        return PREFILL if self.forward_index == 0 else (
            G1 if self.forward_index == 1 else G2PLUS)


@contextmanager
def g1_controlled(bundle, *, v, layer, controller=None, dose: float = 0.0,
                  attack_eps: float = 0.0, attack_sign: float = -1.0,
                  trace: G1Trace | None = None, record: bool = False):
    blocks = decoder_layers(bundle)
    vt = torch.tensor(np.asarray(v, np.float32), device=bundle.device)
    tr = trace or G1Trace()
    tr.reset()

    def hook(_m, _i, out):
        h, rest = (out[0], out[1:]) if isinstance(out, tuple) else (out, None)
        phase = tr.phase()
        seq_len = h.shape[1]
        hf = h.float()
        applied_attack = applied_ctrl = False
        if phase == G1:
            z0 = torch.einsum("bsd,d->bs", hf, vt)[:, -1]
            tr.z_clean = z0.detach().cpu().numpy()
            if dose != 0.0:
                hf = hf + float(dose) * vt.view(1, 1, -1)
            if attack_eps > 0:
                hf = hf + float(attack_sign) * float(attack_eps) * vt.view(1, 1, -1)
                applied_attack = True
            z1 = torch.einsum("bsd,d->bs", hf, vt)[:, -1]
            tr.z_attacked = z1.detach().cpu().numpy()
            if controller is not None:
                hl = hf[:, -1, :].detach().cpu().numpy()
                dh, mag = controller.intervene(hl, np.asarray(v, float))
                hf = hf.clone()
                hf[:, -1, :] = hf[:, -1, :] + torch.tensor(
                    dh, dtype=torch.float32, device=hf.device)
                nrm = np.linalg.norm(hl, axis=-1) + 1e-12
                tr.q_g1 = mag / nrm
                applied_ctrl = True
        if record:
            tr.rows.append({"forward_index": tr.forward_index, "phase": phase,
                            "seq_len": int(seq_len),
                            "attack_applied": applied_attack,
                            "controller_applied": applied_ctrl})
        tr.forward_index += 1
        hn = hf.to(h.dtype)
        return hn if rest is None else (hn,) + rest

    hd = blocks[layer].register_forward_hook(hook)
    try:
        yield tr
    finally:
        hd.remove()


@torch.no_grad()
def generate_g1_only(bundle, prompts, *, v, layer, controller=None, dose=0.0,
                     attack_eps=0.0, attack_sign=-1.0, max_new_tokens=48,
                     batch_size=10, record=False):
    tok = bundle.tokenizer
    texts, zc, za, qs, traces = [], [], [], [], []
    for i in range(0, len(prompts), batch_size):
        ch = prompts[i:i + batch_size]
        enc = tok([chat_prompt(bundle, p) for p in ch], return_tensors="pt",
                  padding=True).to(bundle.device)
        tr = G1Trace()
        with g1_controlled(bundle, v=v, layer=layer, controller=controller,
                           dose=dose, attack_eps=attack_eps,
                           attack_sign=attack_sign, trace=tr, record=record):
            out = bundle.model.generate(**enc, max_new_tokens=max_new_tokens,
                                        do_sample=False,
                                        pad_token_id=tok.pad_token_id)
        texts += tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
        zc.append(tr.z_clean); za.append(tr.z_attacked)
        qs.append(tr.q_g1 if tr.q_g1 is not None else np.zeros(len(ch)))
        if record:
            traces.append(list(tr.rows))
    return {"texts": texts, "z_clean": np.concatenate(zc),
            "z_attacked": np.concatenate(za), "q_g1": np.concatenate(qs),
            "traces": traces}
