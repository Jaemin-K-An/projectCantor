"""Model loading with pinned revisions, and residual-stream access.

Everything is pinned: repo id, revision SHA, dtype, device. `ModelBundle`
records them so every result table can name exactly which weights produced it.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Pinned at V2 freeze time. Both Apache-2.0 and ungated (verified 2026-08-29).
MODEL_REGISTRY: dict[str, dict] = {
    "qwen2.5-0.5b-instruct": {
        "repo": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "family": "qwen2.5", "license": "apache-2.0",
    },
    # SmolLM2-360M is kept in the registry because its NEGATIVE result is
    # reported: it shows no measurable refusal behaviour (0/6 on the harmful
    # smoke set), so it cannot serve as a safety-steering testbed at all.
    "smollm2-360m-instruct": {
        "repo": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "revision": "a10cc1512eabd3dde888204e902eca88bddb4951",
        "family": "smollm2", "license": "apache-2.0",
    },
    "tinyllama-1.1b-chat": {
        "repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        "family": "llama", "license": "apache-2.0",
    },
}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ModelBundle:
    key: str
    repo: str
    revision: str
    family: str
    model: Any
    tokenizer: Any
    device: str
    dtype: str
    n_layers: int
    d_model: int

    def provenance(self) -> dict:
        return {"model_key": self.key, "model_repo": self.repo,
                "model_revision": self.revision, "model_family": self.family,
                "device": self.device, "dtype": self.dtype,
                "n_layers": self.n_layers, "d_model": self.d_model}


def load_model(key: str, *, device: str | None = None,
               dtype: torch.dtype | None = None) -> ModelBundle:
    spec = MODEL_REGISTRY[key]
    device = device or pick_device()
    # float32 on MPS: bfloat16 matmul support is uneven on M-series and the
    # refusal projections here are small differences of large numbers, so the
    # extra precision matters more than the memory at this model size.
    dtype = dtype or (torch.float32 if device in ("mps", "cpu") else torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # required for batched generation
    model = AutoModelForCausalLM.from_pretrained(
        spec["repo"], revision=spec["revision"], torch_dtype=dtype)
    model.to(device).eval()
    model.requires_grad_(False)
    cfg = model.config
    return ModelBundle(key=key, repo=spec["repo"], revision=spec["revision"],
                       family=spec["family"], model=model, tokenizer=tok,
                       device=device, dtype=str(dtype).replace("torch.", ""),
                       n_layers=cfg.num_hidden_layers, d_model=cfg.hidden_size)


def decoder_layers(bundle: ModelBundle):
    """The list of decoder blocks, for hooking the residual stream.

    Llama/Qwen/SmolLM2 all expose `model.model.layers`; anything else raises
    rather than silently hooking the wrong module.
    """
    m = bundle.model
    for path in (("model", "layers"), ("transformer", "h"), ("gpt_neox", "layers")):
        obj = m
        try:
            for p in path:
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise RuntimeError(f"cannot locate decoder layers for {bundle.repo}")


def chat_prompt(bundle: ModelBundle, user_msg: str,
                system: str | None = None) -> str:
    """Render a single-turn user message with the model's own chat template."""
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": user_msg}]
    return bundle.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)
