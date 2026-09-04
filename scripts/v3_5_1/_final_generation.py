"""Utilities for resumable, metadata-complete confirmatory generation."""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from cantor_guard_v32.metrics32 import coherence32, is_refusal32


def load_partial(raw_path: pathlib.Path, private_path: pathlib.Path):
    if raw_path.exists() != private_path.exists():
        raise RuntimeError("raw/private partial-generation files are inconsistent")
    if not raw_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    raw, private = pd.read_csv(raw_path), pd.read_csv(private_path)
    if len(raw) != len(private):
        raise RuntimeError("raw/private partial-generation row counts differ")
    return raw, private


def condition_count(frame, **condition):
    if frame.empty:
        return 0
    mask = np.ones(len(frame), dtype=bool)
    for column, value in condition.items():
        if isinstance(value, float):
            mask &= np.isclose(frame[column].astype(float), value, atol=1e-14, rtol=0)
        else:
            mask &= frame[column].astype(str).to_numpy() == str(value)
    return int(mask.sum())


def records_from_generation(prompts, *, arm, family, epsilon, rho, generation,
                            W_R, computational_reuse=False, reused_from=None):
    rows, private = [], []
    policy = generation.get("records") or []
    for i, text in enumerate(generation["texts"]):
        rec = policy[i] if i < len(policy) else {}
        d_clean = float(generation["d_clean"][i])
        d_attacked = float(generation["d_attacked"][i])
        d_corrected = float(generation["d_corrected"][i])
        x_attacked = max(0.0, -d_attacked)
        coherence = float(coherence32(text))
        words = len((text or "").split())
        row = {
            "pid": str(prompts.pid.iloc[i]), "arm": arm, "rho": rho,
            "family": family, "epsilon": float(epsilon),
            "h_clean_norm": float(generation["h_clean_norm"][i]),
            "q_attack": float(generation["q_attack"][i]),
            "d_clean": d_clean, "d_attacked": d_attacked, "d_corrected": d_corrected,
            "risk_eligible": bool(d_attacked < 0),
            "x_clean": max(0.0, -d_clean), "x_attacked": x_attacked,
            "r_R": x_attacked / float(W_R),
            "cell_type": rec.get("cell_kind", "none"),
            "cell_index": rec.get("cell_index"),
            "policy_status": rec.get("status", "ATTACK_ONLY" if arm == "ATTACK_ONLY" else "NO_CONTROLLER"),
            "action": float(rec.get("action", 0.0)),
            "q_raw": float(rec.get("q_raw", 0.0)),
            "q_ctrl": float(generation["q_ctrl"][i]),
            "clipped": bool(rec.get("clipped", False)),
            "outside_risk_window": bool(rec.get("outside_risk_window", False)),
            "post_control_expected_sensor_displacement": float(rec.get("delta_d_expected", 0.0)),
            "refusal_proxy": int(is_refusal32(text)),
            "coherence": coherence, "output_length": words,
            "degeneration": int(coherence < .95 or words == 0),
            "computational_reuse": bool(computational_reuse),
            "reused_from": reused_from,
        }
        rows.append(row)
        private.append({**row, "prompt": str(prompts.prompt.iloc[i]), "completion": text})
    return pd.DataFrame(rows), pd.DataFrame(private)


def append_and_checkpoint(raw, private, new_raw, new_private, raw_path, private_path):
    raw = pd.concat([raw, new_raw], ignore_index=True)
    private = pd.concat([private, new_private], ignore_index=True)
    if raw.duplicated(["pid", "family", "epsilon", "arm"]).any():
        raise RuntimeError("duplicate factorial cells detected during checkpoint")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path, index=False)
    private.to_csv(private_path, index=False)
    return raw, private
