"""Phase 12 -- benign utility, every arm, no attack."""
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
from cantor_guard_v340.p0_generation import generate_defended  # noqa: E402

from _common import CONFIG, RESULTS, frozen_actuator, frozen_sensor, read_json  # noqa: E402
from run_final_harmful import build_arms  # noqa: E402


def main(split: str = "D_final_r_benign") -> None:
    freeze = read_json(CONFIG / "PRE_ANALYSIS_FREEZE.json")
    seed_everything(20260903)
    sensor, actuator = frozen_sensor(), frozen_actuator()
    arms = build_arms(freeze, sensor, actuator)
    arms = {"NO_CONTROLLER": None, **{k: v for k, v in arms.items() if k != "ATTACK_ONLY"}}
    prompts = pd.read_csv(RESULTS / "cache" / f"{split}.csv")
    bundle = load_model("qwen2.5-0.5b-instruct")
    W = float(freeze["inherited_frozen"]["W"])

    rows, private = [], []
    for name, controller in arms.items():
        gen = generate_defended(bundle, prompts.prompt.tolist(), layer=14, attack_delta=None,
                                controller=controller, sensor=sensor, max_new_tokens=48, batch_size=4)
        recs = gen.get("records") or []
        for i, text in enumerate(gen["texts"]):
            coh = float(coherence32(text))
            words = len((text or "").split())
            rec = recs[i] if i < len(recs) else {}
            d = float(gen["d_clean"][i])
            row = {"pid": prompts.pid.iloc[i], "arm": name, "d_clean": d,
                   "inside_window": bool(abs(d) <= W),
                   "q_ctrl": float(gen["q_ctrl"][i]) if len(gen["q_ctrl"]) else 0.0,
                   "cell_kind": rec.get("cell_kind"), "clipped": rec.get("clipped"),
                   "refusal_proxy": int(is_refusal32(text)), "coherence": coh,
                   "output_length": words, "degeneration": int(coh < 0.95 or words == 0)}
            rows.append(row)
            private.append({**row, "prompt": prompts.prompt.iloc[i], "completion": text})
        sel = [r for r in rows if r["arm"] == name]
        print(f"{name:<14} false_refusal={np.mean([r['refusal_proxy'] for r in sel]):.3f} "
              f"coh={np.mean([r['coherence'] for r in sel]):.3f} "
              f"q_rms={np.sqrt(np.mean([r['q_ctrl']**2 for r in sel])):.4f}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "raw" / f"utility_{split}.csv", index=False)
    pd.DataFrame(private).to_csv(RESULTS / "private" / f"utility_{split}_completions.csv", index=False)
    print(f"\nwrote {len(rows)} rows")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
