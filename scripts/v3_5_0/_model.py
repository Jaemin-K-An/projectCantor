"""Model-side helpers shared by prospective V3.5.0 calibration scripts."""
from __future__ import annotations

import numpy as np
import torch

from cantor_guard.models import decoder_layers
from cantor_guard_v31.generation31 import chat_prompt
from cantor_guard_v335a.p0_residual import last_valid_index


@torch.no_grad()
def clean_residuals(bundle, prompts, *, layer: int = 14, batch_size: int = 8) -> np.ndarray:
    out = []
    for start in range(0, len(prompts), batch_size):
        chunk = list(prompts[start:start + batch_size])
        encoded = bundle.tokenizer([chat_prompt(bundle, p) for p in chunk],
                                   return_tensors="pt", padding=True).to(bundle.device)
        idx = last_valid_index(encoded["attention_mask"])
        store = {}

        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            arange = torch.arange(h.shape[0], device=h.device)
            store["h"] = h.float()[arange, idx, :].detach().cpu().numpy().astype(np.float64)
            return output

        handle = decoder_layers(bundle)[layer].register_forward_hook(hook)
        try:
            bundle.model(**encoded)
        finally:
            handle.remove()
        out.append(store["h"])
    return np.concatenate(out, axis=0)
