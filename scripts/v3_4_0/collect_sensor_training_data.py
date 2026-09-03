"""Phase 2 -- clean P0 residuals and clean completions for every split.

No intervention is applied anywhere in this script.  Residuals are saved as
float64 arrays keyed by prompt id; completions go to the gitignored private
directory and only hashes/scalars are tracked.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "llm/src"))
from cantor_guard.io import seed_everything  # noqa: E402
from cantor_guard.models import load_model  # noqa: E402
from cantor_guard_v32.metrics32 import coherence32, is_refusal32, safe_score32  # noqa: E402
from cantor_guard_v340.p0_generation import clean_p0_and_generate  # noqa: E402

from _common import RESULTS, write_json  # noqa: E402

MODEL = "qwen2.5-0.5b-instruct"
LAYER = 14
MAX_NEW_TOKENS = 48
SEED = 20260903
SPLITS = [
    "D_eval_val_harmful", "D_eval_val_benign",
    "D_sensor_train", "D_sensor_tune", "D_sensor_confirm",
    "D_actuator_validate", "D_window_cal", "D_controller_budget",
    "D_attack_dev", "D_final_harmful", "D_final_benign",
]


def main(splits=None) -> None:
    targets = list(splits or SPLITS)
    seed_everything(SEED)
    bundle = load_model(MODEL)
    summary = {}
    for split in targets:
        frame = pd.read_csv(RESULTS / "cache" / f"{split}.csv")
        residual, texts = clean_p0_and_generate(
            bundle, frame.prompt.tolist(), layer=LAYER, max_new_tokens=MAX_NEW_TOKENS
        )
        rows = []
        private = []
        for i, text in enumerate(texts):
            coherence = float(coherence32(text))
            words = len((text or "").split())
            row = {
                "pid": frame.pid.iloc[i],
                "split": split,
                "kind": frame.kind.iloc[i],
                "h_norm": float(np.linalg.norm(residual[i])),
                "refusal_proxy": int(is_refusal32(text)),
                "safe_score32": float(safe_score32(text)),
                "coherence": coherence,
                "output_length": words,
                "degeneration": int(coherence < 0.95 or words == 0),
            }
            rows.append(row)
            private.append({**row, "prompt": frame.prompt.iloc[i], "completion": text})
        table = pd.DataFrame(rows)
        (RESULTS / "raw").mkdir(parents=True, exist_ok=True)
        (RESULTS / "private").mkdir(parents=True, exist_ok=True)
        table.to_csv(RESULTS / "raw" / f"clean_{split}.csv", index=False)
        pd.DataFrame(private).to_csv(RESULTS / "private" / f"clean_{split}_completions.csv", index=False)
        np.save(RESULTS / "cache" / f"h_{split}.npy", residual)
        summary[split] = {
            "n": int(len(table)),
            "refusal_rate": float(table.refusal_proxy.mean()),
            "mean_coherence": float(table.coherence.mean()),
            "degeneration_rate": float(table.degeneration.mean()),
            "mean_h_norm": float(table.h_norm.mean()),
            "d_model": int(residual.shape[1]),
        }
        print(
            f"{split}: n={len(table)} refusal={table.refusal_proxy.mean():.3f} "
            f"coh={table.coherence.mean():.3f} degen={table.degeneration.mean():.3f} "
            f"|h|={table.h_norm.mean():.2f}",
            flush=True,
        )
    write_json(RESULTS / "tables" / "clean_collection_summary.json", {
        "model": MODEL, "layer": LAYER, "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED, "intervention": "none", "splits": summary,
    })


if __name__ == "__main__":
    main(sys.argv[1:] or None)
