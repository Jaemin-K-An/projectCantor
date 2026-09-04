"""Phase 13: fresh benign utility experiment with exact safe-side zero action."""
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
from cantor_guard_v350.p0_generation import generate_defended  # noqa: E402
from _common import RESULTS, build_arms, frozen_actuator, frozen_sensor, require_freeze  # noqa: E402


def main() -> None:
    freeze = require_freeze(); output = RESULTS / "raw/utility_D_final_v350_benign.csv"
    if output.exists(): raise SystemExit("final benign output already exists; regeneration forbidden")
    prompts = pd.read_csv(RESULTS / "cache/D_final_v350_benign.csv")
    if len(prompts) < 80: raise SystemExit("final benign requires n>=80")
    seed_everything(20260904); sensor, actuator = frozen_sensor(), frozen_actuator()
    acting = build_arms(freeze, sensor, actuator)
    arms = {"NO_CONTROLLER": None, **{k: v for k, v in acting.items() if k != "ATTACK_ONLY"}}
    bundle = load_model(freeze["sensor"]["model"]); rows, private = [], []
    for arm, controller in arms.items():
        gen = generate_defended(bundle, prompts.prompt.tolist(), layer=14, controller=controller, sensor=sensor,
                                max_new_tokens=freeze["generation"]["max_new_tokens"], batch_size=freeze["generation"]["batch_size"])
        recs = gen.get("records") or []
        for i, text in enumerate(gen["texts"]):
            rec = recs[i] if i < len(recs) else {}; d = float(gen["d_clean"][i]); coh = float(coherence32(text)); words = len((text or "").split())
            row = {"pid": prompts.pid.iloc[i], "arm": arm, "d_clean": d, "x_risk": max(0.0, -d),
                   "safe_side": d >= 0, "q_ctrl": float(gen["q_ctrl"][i]),
                   "cell_kind": rec.get("cell_kind"), "clipped": rec.get("clipped"),
                   "refusal_proxy": int(is_refusal32(text)), "coherence": coh,
                   "output_length": words, "degeneration": int(coh < .95 or words == 0)}
            rows.append(row); private.append({**row, "prompt": prompts.prompt.iloc[i], "completion": text})
    pd.DataFrame(rows).to_csv(output, index=False)
    pd.DataFrame(private).to_csv(RESULTS / "private/utility_D_final_v350_benign_completions.csv", index=False)


if __name__ == "__main__": main()
