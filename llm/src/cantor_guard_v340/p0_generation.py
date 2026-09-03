"""Clean P0 residual extraction and generation for sensor training.

The sensor must be trained on CLEAN residual states and the behaviour the
model actually produced from them, so this path applies no intervention at
all.  P0 is the last valid prompt token of the prefill forward -- the state
that exists immediately before token 1 is sampled -- located with the
padding-safe index used since V3.3.5a.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

from cantor_guard.models import decoder_layers
from cantor_guard_v31.generation31 import chat_prompt
from cantor_guard_v335a.p0_residual import last_valid_index


@contextmanager
def capture_p0_residual(bundle, *, layer: int, last_idx, store: dict):
    """Read-only hook: records the P0 row and changes nothing."""
    state = {"forward": 0}

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if state["forward"] == 0:
            arange = torch.arange(h.shape[0], device=h.device)
            row = h.float()[arange, last_idx, :]
            store["h"] = row.detach().cpu().numpy().astype(np.float64)
        state["forward"] += 1
        return output

    handle = decoder_layers(bundle)[layer].register_forward_hook(hook)
    try:
        yield store
    finally:
        handle.remove()


@torch.no_grad()
def clean_p0_and_generate(bundle, prompts, *, layer: int, max_new_tokens: int = 48, batch_size: int = 8):
    """Return clean P0 residuals and the completions they produced."""
    tokenizer = bundle.tokenizer
    residuals: list[np.ndarray] = []
    texts: list[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = list(prompts[start : start + batch_size])
        encoded = tokenizer(
            [chat_prompt(bundle, p) for p in chunk], return_tensors="pt", padding=True
        ).to(bundle.device)
        idx = last_valid_index(encoded["attention_mask"])
        store: dict = {}
        with capture_p0_residual(bundle, layer=layer, last_idx=idx, store=store):
            generated = bundle.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        residuals.append(store["h"])
        texts.extend(
            tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
        )
    return np.concatenate(residuals, axis=0), texts


@contextmanager
def p0_attack_then_control(bundle, *, layer: int, last_idx, attack_delta=None,
                           controller=None, sensor=None, trace: dict | None = None):
    """Order of operations inside the real forward pass, at P0 only.

        clean h -> + attack_delta -> controller observes the ATTACKED state
                -> controller correction -> token-1 logits -> full decode.

    The controller never sees the clean state, and nothing after the prefill
    forward is touched, so no P0 intervention leaks into G1+.
    """
    tr = trace if trace is not None else {}
    tr.setdefault("forward", 0)
    tr.setdefault("phases", [])
    delta = None
    if attack_delta is not None:
        delta = torch.as_tensor(np.asarray(attack_delta, dtype=float), dtype=torch.float32,
                                device=bundle.device)

    def hook(_module, _inputs, output):
        h, rest = (output[0], output[1:]) if isinstance(output, tuple) else (output, None)
        tr["phases"].append("P0" if tr["forward"] == 0 else f"G{tr['forward']}")
        hf = h.float()
        if tr["forward"] == 0:
            arange = torch.arange(hf.shape[0], device=hf.device)
            row = hf[arange, last_idx, :]
            clean = row.detach().cpu().numpy().astype(np.float64)
            tr["h_clean_norm"] = np.linalg.norm(clean, axis=1)
            if sensor is not None:
                tr["d_clean"] = np.atleast_1d(sensor.distance(clean))
            if delta is not None:
                row = row + delta[None, :]
            attacked = row.detach().cpu().numpy().astype(np.float64)
            if sensor is not None:
                tr["d_attacked"] = np.atleast_1d(sensor.distance(attacked))
            tr["q_attack"] = np.linalg.norm(attacked - clean, axis=1) / (tr["h_clean_norm"] + 1e-12)
            if controller is not None:
                tr["records"] = controller.policy_record(attacked)
                result = controller.correct(attacked)
                corrected = np.atleast_2d(result.h_corrected)
                tr["q_ctrl"] = np.atleast_1d(result.q_ctrl)
                row = torch.as_tensor(corrected, dtype=torch.float32, device=hf.device)
            else:
                corrected = attacked
                tr["q_ctrl"] = np.zeros(hf.shape[0], dtype=float)
                tr["records"] = []
            if sensor is not None:
                tr["d_corrected"] = np.atleast_1d(sensor.distance(corrected))
            hf = hf.clone()
            hf[arange, last_idx, :] = row
        tr["forward"] += 1
        cast = hf.to(h.dtype)
        return cast if rest is None else (cast,) + rest

    handle = decoder_layers(bundle)[layer].register_forward_hook(hook)
    try:
        yield tr
    finally:
        handle.remove()


@torch.no_grad()
def generate_defended(bundle, prompts, *, layer: int, attack_delta=None, controller=None,
                      sensor=None, max_new_tokens: int = 48, batch_size: int = 8,
                      record_first_logits: bool = False):
    """Run the full attack -> defend -> generate pipeline and collect the trace."""
    tokenizer = bundle.tokenizer
    out = {k: [] for k in ("texts", "h_clean_norm", "d_clean", "d_attacked", "d_corrected",
                           "q_attack", "q_ctrl", "records")}
    if record_first_logits:
        out["first_logits"] = []
    for start in range(0, len(prompts), batch_size):
        chunk = list(prompts[start : start + batch_size])
        encoded = tokenizer([chat_prompt(bundle, p) for p in chunk],
                            return_tensors="pt", padding=True).to(bundle.device)
        idx = last_valid_index(encoded["attention_mask"])
        trace: dict = {}
        with p0_attack_then_control(bundle, layer=layer, last_idx=idx, attack_delta=attack_delta,
                                    controller=controller, sensor=sensor, trace=trace):
            if record_first_logits:
                logits = bundle.model(**encoded).logits
                out["first_logits"].append(
                    logits[torch.arange(logits.shape[0], device=logits.device), idx]
                    .float().detach().cpu().numpy()
                )
                trace.clear()
                trace.update({"forward": 0, "phases": []})
            generated = bundle.model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        out["texts"].extend(
            tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
        )
        for key in ("h_clean_norm", "d_clean", "d_attacked", "d_corrected", "q_attack", "q_ctrl"):
            if key in trace:
                out[key].append(np.asarray(trace[key]))
        out["records"].extend(trace.get("records", []))
    for key in ("h_clean_norm", "d_clean", "d_attacked", "d_corrected", "q_attack", "q_ctrl"):
        out[key] = np.concatenate(out[key]) if out[key] else np.array([])
    if record_first_logits:
        out["first_logits"] = np.concatenate(out["first_logits"], axis=0)
    return out
