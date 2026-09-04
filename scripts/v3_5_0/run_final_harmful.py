"""Phase 12: fresh attack-only versus one-sided Cantor final experiment."""
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
from cantor_guard_v340.attack import attack_v, attack_w  # noqa: E402
from cantor_guard_v350.p0_generation import generate_defended  # noqa: E402
from _common import RESULTS, build_arms, frozen_actuator, frozen_sensor, require_freeze  # noqa: E402


def main() -> None:
    freeze = require_freeze()
    output = RESULTS / "raw/final_D_final_v350_harmful.csv"
    if output.exists(): raise SystemExit("final harmful output already exists; regeneration forbidden")
    prompts = pd.read_csv(RESULTS / "cache/D_final_v350_harmful.csv")
    if len(prompts) < 150: raise SystemExit("final harmful requires n>=150")
    seed_everything(20260904); sensor, actuator = frozen_sensor(), frozen_actuator()
    arms = build_arms(freeze, sensor, actuator); bundle = load_model(freeze["sensor"]["model"])
    rows, private = [], []

    def record(arm, family, eps, gen):
        recs = gen.get("records") or []
        for i, text in enumerate(gen["texts"]):
            rec = recs[i] if i < len(recs) else {}; coh = float(coherence32(text)); words = len((text or "").split())
            dc, da = float(gen["d_clean"][i]), float(gen["d_attacked"][i])
            row = {"pid": prompts.pid.iloc[i], "arm": arm, "family": family, "epsilon": eps,
                   "d_clean": dc, "d_attacked": da, "d_corrected": float(gen["d_corrected"][i]),
                   "x_clean": max(0.0, -dc), "x_attacked": max(0.0, -da),
                   "delta_d": da - dc, "delta_x": max(0.0, -da) - max(0.0, -dc),
                   "delta_r_R": (max(0.0, -da) - max(0.0, -dc)) / float(freeze["risk_window"]["W_R"]),
                   "q_ctrl": float(gen["q_ctrl"][i]), "cell_kind": rec.get("cell_kind"),
                   "cell_index": rec.get("cell_index"), "action": rec.get("action"),
                   "clipped": rec.get("clipped"), "outside_risk_window": rec.get("outside_risk_window"),
                   "refusal_proxy": int(is_refusal32(text)), "coherence": coh,
                   "output_length": words, "degeneration": int(coh < .95 or words == 0)}
            rows.append(row); private.append({**row, "prompt": prompts.prompt.iloc[i], "completion": text})

    gen = generate_defended(bundle, prompts.prompt.tolist(), layer=14, controller=None, sensor=sensor,
                            max_new_tokens=freeze["generation"]["max_new_tokens"], batch_size=freeze["generation"]["batch_size"])
    record("CLEAN", "NONE", 0.0, gen)
    for family in ("ATTACK_W", "ATTACK_V"):
        for eps in [e for e in freeze["attacks"]["grid_absolute"] if e > 0]:
            delta = attack_w(sensor, eps, sign=-1) if family == "ATTACK_W" else attack_v(actuator, eps)
            for arm, controller in arms.items():
                gen = generate_defended(bundle, prompts.prompt.tolist(), layer=14, attack_delta=delta,
                                        controller=controller, sensor=sensor,
                                        max_new_tokens=freeze["generation"]["max_new_tokens"],
                                        batch_size=freeze["generation"]["batch_size"])
                record(arm, family, eps, gen)
            print(f"{family} epsilon={eps:.9f} complete", flush=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    pd.DataFrame(private).to_csv(RESULTS / "private/final_D_final_v350_harmful_completions.csv", index=False)


if __name__ == "__main__": main()
