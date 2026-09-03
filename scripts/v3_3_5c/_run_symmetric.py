from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "llm/src")
from cantor_guard.io import seed_everything  # noqa: E402
from cantor_guard.models import load_model  # noqa: E402
from cantor_guard_v32.metrics32 import coherence32, is_refusal32, safe_score32  # noqa: E402
from cantor_guard_v335c.p0_attack_generation import generate_attacked_p0  # noqa: E402

from _common import RESULTS, behavioral_protocol, load_direction


def run_symmetric(split: str, doses: list[float], *, batch_size: int = 8) -> pd.DataFrame:
    protocol = behavioral_protocol()
    seed_everything(int(protocol["generation_seed"]))
    prompts = pd.read_csv(RESULTS / "cache" / f"{split}.csv")
    bundle = load_model(protocol["model"])
    direction = load_direction(protocol)
    rows: list[dict] = []
    private: list[dict] = []
    for dose in doses:
        generated = generate_attacked_p0(
            bundle,
            prompts.prompt.tolist(),
            layer=int(protocol["layer"]),
            v=direction,
            normalized_u=float(dose),
            max_new_tokens=int(protocol["max_new_tokens"]),
            batch_size=batch_size,
        )
        for i, text in enumerate(generated["texts"]):
            coherence = coherence32(text)
            refusal = int(is_refusal32(text))
            words = len((text or "").split())
            row = {
                "pid": prompts.pid.iloc[i],
                "u": float(dose),
                "h_clean_norm": float(generated["h_clean_norm"][i]),
                "z_clean": float(generated["z_clean"][i]),
                "z_after": float(generated["z_attacked"][i]),
                "relative_norm_realised": float(generated["q_attack"][i]),
                "refusal_proxy": refusal,
                "semantic_harmful_compliance": None,
                "coherence": float(coherence),
                "safe_score32": float(safe_score32(text)),
                "output_length": words,
                "degeneration": int(coherence < float(protocol["coherence_gate"]) or words == 0),
            }
            rows.append(row)
            private.append({**row, "completion": text})
        frame = pd.DataFrame(rows)
        current = frame[frame.u == dose]
        print(
            f"u={dose:+.2f} refusal={current.refusal_proxy.mean():.3f} "
            f"coherence={current.coherence.mean():.3f} "
            f"degenerate={current.degeneration.mean():.3f} "
            f"q_med={current.relative_norm_realised.median():.3f}",
            flush=True,
        )
    result = pd.DataFrame(rows)
    raw_path = RESULTS / "raw" / f"symmetric_{split}.csv"
    private_path = RESULTS / "private" / f"symmetric_{split}_completions.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(raw_path, index=False)
    pd.DataFrame(private).to_csv(private_path, index=False)
    return result
