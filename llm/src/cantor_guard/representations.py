"""Batched extraction of residual activations at the last prompt token."""
from __future__ import annotations
import numpy as np, torch
from .hooks import capture_residuals
from .models import chat_prompt


@torch.no_grad()
def last_token_residuals(bundle, prompts: list[str], layers: list[int],
                         *, batch_size: int = 8, system: str | None = None,
                         progress: bool = False) -> np.ndarray:
    """Return `[n_prompts, n_layers, d_model]` float32.

    The prompt is rendered with the model's chat template and the generation
    prompt appended, so the captured position is the token the model is about
    to continue from -- the position at which the refusal decision is made.
    """
    tok = bundle.tokenizer
    out = np.zeros((len(prompts), len(layers), bundle.d_model), dtype=np.float32)
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        texts = [chat_prompt(bundle, p, system) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(bundle.device)
        store: dict = {}
        with capture_residuals(bundle, layers, store, last_token_only=True):
            bundle.model(**enc)
        for j, l in enumerate(layers):
            v = store[l].numpy()
            if not np.isfinite(v).all():
                raise RuntimeError(
                    f"non-finite residuals at layer {l}. On MPS this means the "
                    f"SDPA kernel was used with left padding; load the model "
                    f"with attn_implementation='eager' (see models.load_model).")
            out[i:i + len(chunk), j, :] = v
        if progress:
            print(f"  residuals {min(i+batch_size, len(prompts))}/{len(prompts)}",
                  end="\r", flush=True)
    if progress:
        print()
    return out
