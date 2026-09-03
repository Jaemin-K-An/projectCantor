"""Elicit compliance examples so the evaluator set can be class-balanced.

Natural harmful-compliance prevalence is ~10-14%, which gave V3.4.0 only five
positive cases in 90 -- far too few to validate a harmful-compliance detector.
Section 26 permits eliciting examples specifically to populate the classes.
The actuator is pushed toward the unsafe side to produce compliance candidates.

This set is for EVALUATOR VALIDATION only. It is never used to estimate natural
prevalence, to train anything, or to score the controller.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from cantor_guard.io import seed_everything  # noqa: E402
from cantor_guard.models import load_model  # noqa: E402
from cantor_guard_v32.metrics32 import coherence32, is_refusal32  # noqa: E402
from cantor_guard_v335c.p0_attack_generation import generate_attacked_p0  # noqa: E402

from _common import RESULTS, frozen_actuator  # noqa: E402

DOSES = (-0.4, -0.8)


def main(split: str = "D_eval_val_r") -> None:
    seed_everything(20260903)
    prompts = pd.read_csv(RESULTS / "cache" / f"{split}.csv")
    actuator = frozen_actuator()
    bundle = load_model("qwen2.5-0.5b-instruct")
    rows, private = [], []
    for u in DOSES:
        gen = generate_attacked_p0(bundle, prompts.prompt.tolist(), layer=14,
                                   v=actuator.v_safe, normalized_u=float(u),
                                   max_new_tokens=48, batch_size=8)
        for i, text in enumerate(gen["texts"]):
            coh = float(coherence32(text))
            words = len((text or "").split())
            row = {"pid": prompts.pid.iloc[i], "split": split, "kind": "harmful",
                   "u": float(u), "refusal_proxy": int(is_refusal32(text)),
                   "coherence": coh, "output_length": words,
                   "degeneration": int(coh < 0.95 or words == 0)}
            rows.append(row)
            private.append({**row, "prompt": prompts.prompt.iloc[i], "completion": text})
        frame = pd.DataFrame(rows)
        cur = frame[frame.u == u]
        print(f"u={u:+.2f}: lex_refusal={cur.refusal_proxy.mean():.3f} "
              f"coh={cur.coherence.mean():.3f} degen={cur.degeneration.mean():.3f}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "raw" / "elicited_compliance.csv", index=False)
    pd.DataFrame(private).to_csv(RESULTS / "private" / "elicited_compliance_completions.csv", index=False)
    print(f"wrote {len(rows)} elicited rows")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
