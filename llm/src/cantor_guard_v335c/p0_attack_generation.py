"""P0-only model hook: clean -> attack/dose -> controller -> token-1/full decode."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch

from cantor_guard.models import decoder_layers
from cantor_guard_v31.generation31 import chat_prompt
from cantor_guard_v335a.p0_residual import last_valid_index
from .p0_normalized_dose import normalize_direction


@dataclass
class P0AttackTrace:
    forward_index: int = 0
    z_clean: np.ndarray | None = None
    z_attacked: np.ndarray | None = None
    z_corrected: np.ndarray | None = None
    h_clean_norm: np.ndarray | None = None
    q_attack: np.ndarray | None = None
    q_ctrl: np.ndarray | None = None
    controller_records: list[dict] = field(default_factory=list)
    phases: list[str] = field(default_factory=list)


@contextmanager
def p0_attack_then_control(
    bundle,
    *,
    layer: int,
    v,
    last_idx,
    controller=None,
    attack_epsilon: float = 0.0,
    unsafe_sign: int = -1,
    normalized_u: float | None = None,
    trace: P0AttackTrace | None = None,
):
    if unsafe_sign not in (-1, 1):
        raise ValueError("unsafe_sign must be +/-1")
    if attack_epsilon < 0:
        raise ValueError("attack_epsilon must be nonnegative")
    if normalized_u is not None and attack_epsilon:
        raise ValueError("normalized dose and absolute attack are separate protocols")
    direction = normalize_direction(v)
    vt = torch.as_tensor(direction, dtype=torch.float32, device=bundle.device)
    tr = trace or P0AttackTrace()
    handle = None

    def hook(_module, _inputs, output):
        h, rest = (output[0], output[1:]) if isinstance(output, tuple) else (output, None)
        phase = "P0" if tr.forward_index == 0 else f"G{tr.forward_index}"
        tr.phases.append(phase)
        hf = h.float()
        if tr.forward_index == 0:
            arange = torch.arange(hf.shape[0], device=hf.device)
            row = hf[arange, last_idx, :]
            clean_norm = torch.linalg.norm(row, dim=-1)
            tr.h_clean_norm = clean_norm.detach().cpu().numpy()
            tr.z_clean = (row @ vt).detach().cpu().numpy()
            if normalized_u is not None:
                delta_attack = float(normalized_u) * clean_norm[:, None] * vt[None, :]
            else:
                delta_attack = float(unsafe_sign) * float(attack_epsilon) * vt[None, :]
            row = row + delta_attack
            tr.q_attack = (
                torch.linalg.norm(delta_attack, dim=-1) / (clean_norm + 1e-12)
            ).detach().cpu().numpy()
            tr.z_attacked = (row @ vt).detach().cpu().numpy()
            if controller is not None:
                controlled = controller.correct(row.detach().cpu().numpy())
                row = torch.as_tensor(controlled.h_corrected, dtype=torch.float32, device=hf.device)
                tr.q_ctrl = np.asarray(controlled.q_ctrl, dtype=float).reshape(-1)
                tr.controller_records = controller.policy_record(
                    (row - torch.as_tensor(controlled.delta_h_controller, dtype=torch.float32, device=hf.device)).detach().cpu().numpy()
                )
            else:
                tr.q_ctrl = np.zeros(hf.shape[0], dtype=float)
            tr.z_corrected = (row @ vt).detach().cpu().numpy()
            hf = hf.clone()
            hf[arange, last_idx, :] = row
        tr.forward_index += 1
        cast = hf.to(h.dtype)
        return cast if rest is None else (cast,) + rest

    handle = decoder_layers(bundle)[layer].register_forward_hook(hook)
    try:
        yield tr
    finally:
        if handle is not None:
            handle.remove()


def _first_logits(bundle, encoded, idx, **hook_args):
    trace = P0AttackTrace()
    with p0_attack_then_control(bundle, last_idx=idx, trace=trace, **hook_args):
        logits = bundle.model(**encoded).logits
    selected = logits[torch.arange(logits.shape[0], device=logits.device), idx]
    return selected.float().detach().cpu().numpy(), trace


@torch.no_grad()
def generate_attacked_p0(
    bundle,
    prompts,
    *,
    layer: int,
    v,
    controller=None,
    attack_epsilon: float = 0.0,
    unsafe_sign: int = -1,
    normalized_u: float | None = None,
    max_new_tokens: int = 48,
    batch_size: int = 8,
    record_first_logits: bool = False,
):
    tokenizer = bundle.tokenizer
    output = {
        "texts": [], "z_clean": [], "z_attacked": [], "z_corrected": [],
        "h_clean_norm": [], "q_attack": [], "q_ctrl": [], "controller_records": [],
        "generation_phases": [],
    }
    if record_first_logits:
        output.update({"first_logits_clean": [], "first_logits_attacked": [], "first_logits_corrected": []})
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        encoded = tokenizer([chat_prompt(bundle, prompt) for prompt in chunk], return_tensors="pt", padding=True).to(bundle.device)
        idx = last_valid_index(encoded["attention_mask"])
        common = {"bundle": bundle, "layer": layer, "v": v, "unsafe_sign": unsafe_sign}
        if record_first_logits:
            clean, _ = _first_logits(bundle, encoded, idx, **common)
            attacked, _ = _first_logits(
                bundle, encoded, idx, controller=None, attack_epsilon=attack_epsilon,
                normalized_u=normalized_u, **common
            )
            corrected, _ = _first_logits(
                bundle, encoded, idx, controller=controller, attack_epsilon=attack_epsilon,
                normalized_u=normalized_u, **common
            )
            output["first_logits_clean"].append(clean)
            output["first_logits_attacked"].append(attacked)
            output["first_logits_corrected"].append(corrected)
        trace = P0AttackTrace()
        with p0_attack_then_control(
            bundle, layer=layer, v=v, last_idx=idx, controller=controller,
            attack_epsilon=attack_epsilon, unsafe_sign=unsafe_sign,
            normalized_u=normalized_u, trace=trace,
        ):
            generated = bundle.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        output["texts"].extend(
            tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
        )
        for key in ("z_clean", "z_attacked", "z_corrected", "h_clean_norm", "q_attack", "q_ctrl"):
            output[key].append(np.asarray(getattr(trace, key)))
        output["controller_records"].extend(trace.controller_records)
        output["generation_phases"].append(list(trace.phases))
    for key in ("z_clean", "z_attacked", "z_corrected", "h_clean_norm", "q_attack", "q_ctrl"):
        output[key] = np.concatenate(output[key]) if output[key] else np.array([])
    if record_first_logits:
        for key in ("first_logits_clean", "first_logits_attacked", "first_logits_corrected"):
            output[key] = np.concatenate(output[key], axis=0)
    return output
