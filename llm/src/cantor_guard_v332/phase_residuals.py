"""V3.3.2 PHASE 5 -- collect residuals at PHASE P and PHASE G1.

Controller is always OFF here: this measures the model, not a controller.
Exactly one G1 state per prompt, so long generations cannot outvote short ones
and a prompt-clustered bootstrap is well defined.
"""
from __future__ import annotations
from contextlib import contextmanager
import numpy as np, torch
from cantor_guard.models import decoder_layers
from cantor_guard_v31.generation31 import chat_prompt
from .phase_state import PhaseState, PREFILL, DECODE

__all__ = ["collect_phase_residuals"]


@torch.no_grad()
def collect_phase_residuals(bundle, prompts, layer: int, *, max_new_tokens: int = 8,
                            batch_size: int = 8, record_trace: bool = False):
    """Return dict of arrays, one row per prompt.

      prompt_last : residual at the last prompt token (PHASE P)
      decode1     : residual at the FIRST decode forward (PHASE G1)
      decode1_4   : mean over decode forwards 1..4
      decode5_8   : mean over decode forwards 5..8
    """
    blocks = decoder_layers(bundle)
    tok = bundle.tokenizer
    d = bundle.d_model
    P = np.zeros((len(prompts), d), np.float32)
    G1 = np.zeros((len(prompts), d), np.float32)
    G14 = np.zeros((len(prompts), d), np.float32)
    G58 = np.zeros((len(prompts), d), np.float32)
    n14 = np.zeros(len(prompts)); n58 = np.zeros(len(prompts))
    traces = []

    for i in range(0, len(prompts), batch_size):
        ch = prompts[i:i + batch_size]
        enc = tok([chat_prompt(bundle, p) for p in ch], return_tensors="pt",
                  padding=True).to(bundle.device)
        st = PhaseState(record_trace=record_trace)
        st.reset()
        store = {}

        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            phase = st.observe(h.shape[1], h.shape[1] == 1)
            hf = h.float().detach()
            if phase == PREFILL:
                store["P"] = hf[:, -1, :].cpu().numpy()
            else:
                k = st.forward_index - 2      # 0-based decode index
                v = hf[:, -1, :].cpu().numpy()
                if k == 0:
                    store["G1"] = v
                if 0 <= k < 4:
                    store["G14"] = store.get("G14", 0.0) + v
                    store["n14"] = store.get("n14", 0) + 1
                if 4 <= k < 8:
                    store["G58"] = store.get("G58", 0.0) + v
                    store["n58"] = store.get("n58", 0) + 1
            return out

        hd = blocks[layer].register_forward_hook(hook)
        try:
            bundle.model.generate(**enc, max_new_tokens=max_new_tokens,
                                  do_sample=False, pad_token_id=tok.pad_token_id)
        finally:
            hd.remove()
        sl = slice(i, i + len(ch))
        P[sl] = store["P"]; G1[sl] = store["G1"]
        if store.get("n14"): G14[sl] = store["G14"] / store["n14"]; n14[sl] = store["n14"]
        if store.get("n58"): G58[sl] = store["G58"] / store["n58"]; n58[sl] = store["n58"]
        if record_trace:
            traces.append(st.consistency())

    return {"prompt_last": P, "decode1": G1, "decode1_4": G14,
            "decode5_8": G58, "n_decode_1_4": n14, "n_decode_5_8": n58,
            "traces": traces}
