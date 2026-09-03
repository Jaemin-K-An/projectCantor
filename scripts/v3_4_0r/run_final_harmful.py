"""Phase 11 -- confirmatory run with the baselines V3.4.0 was missing.

Arms per (attack family, epsilon): ATTACK_ONLY (no controller), LINEAR
(sensor-actuator without the recursive partition), and the seven rho
controllers.  Plus one CLEAN arm.  Without ATTACK_ONLY nothing distinguishes
"the controller helps" from "the controller does nothing", and without LINEAR
nothing distinguishes the architecture from the Cantor partition.
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
from cantor_guard_v340.attack import attack_v, attack_w  # noqa: E402
from cantor_guard_v340.p0_generation import generate_defended  # noqa: E402
from cantor_guard_v340r.controllers import CappedCantorController, LinearThresholdController  # noqa: E402

from _common import CONFIG, RESULTS, frozen_actuator, frozen_sensor, read_json  # noqa: E402


def build_arms(freeze, sensor, actuator):
    frozen = freeze["inherited_frozen"]
    W = float(frozen["W"])
    q_cap = float(freeze["hard_q_cap"]["q_cap"])
    etas = freeze["budget"]["eta_per_arm"]
    arms = {"ATTACK_ONLY": None,
            "LINEAR": LinearThresholdController(sensor=sensor, actuator=actuator, W=W,
                                                eta=float(etas["LINEAR"]), q_cap=q_cap)}
    for rho in frozen["rho_family"]:
        key = "1/3" if abs(float(rho) - 1 / 3) < 1e-12 else f"{float(rho):.2f}"
        arms[key] = CappedCantorController(sensor=sensor, actuator=actuator, W=W, rho=float(rho),
                                           eta=float(etas[key]), q_cap=q_cap,
                                           leaf_actions=frozen["leaf_actions"])
    return arms


def main(split: str = "D_final_r_harmful") -> None:
    freeze = read_json(CONFIG / "PRE_ANALYSIS_FREEZE.json")
    seed_everything(20260903)
    sensor, actuator = frozen_sensor(), frozen_actuator()
    arms = build_arms(freeze, sensor, actuator)
    grid = [e for e in freeze["attacks"]["grid_absolute"] if e > 0]
    prompts = pd.read_csv(RESULTS / "cache" / f"{split}.csv")
    bundle = load_model("qwen2.5-0.5b-instruct")

    rows, private = [], []

    def record(arm, family, eps, gen):
        recs = gen.get("records") or []
        for i, text in enumerate(gen["texts"]):
            coh = float(coherence32(text))
            words = len((text or "").split())
            rec = recs[i] if i < len(recs) else {}
            row = {"pid": prompts.pid.iloc[i], "arm": arm, "family": family, "epsilon": float(eps),
                   "d_clean": float(gen["d_clean"][i]) if len(gen["d_clean"]) else np.nan,
                   "d_attacked": float(gen["d_attacked"][i]) if len(gen["d_attacked"]) else np.nan,
                   "d_corrected": float(gen["d_corrected"][i]) if len(gen["d_corrected"]) else np.nan,
                   "q_ctrl": float(gen["q_ctrl"][i]) if len(gen["q_ctrl"]) else 0.0,
                   "cell_kind": rec.get("cell_kind"), "cell_index": rec.get("cell_index"),
                   "action": rec.get("action"), "clipped": rec.get("clipped"),
                   "refusal_proxy": int(is_refusal32(text)), "coherence": coh,
                   "output_length": words, "degeneration": int(coh < 0.95 or words == 0)}
            rows.append(row)
            private.append({**row, "prompt": prompts.prompt.iloc[i], "completion": text})

    gen = generate_defended(bundle, prompts.prompt.tolist(), layer=14, attack_delta=None,
                            controller=None, sensor=sensor, max_new_tokens=48, batch_size=4)
    record("CLEAN", "NONE", 0.0, gen)
    print(f"CLEAN lex_refusal={np.mean([is_refusal32(t) for t in gen['texts']]):.3f}", flush=True)

    for family in ("ATTACK_W", "ATTACK_V"):
        for eps in grid:
            delta = attack_w(sensor, eps, sign=-1) if family == "ATTACK_W" else attack_v(actuator, eps)
            for arm, controller in arms.items():
                gen = generate_defended(bundle, prompts.prompt.tolist(), layer=14,
                                        attack_delta=delta, controller=controller,
                                        sensor=sensor, max_new_tokens=48, batch_size=4)
                record(arm, family, eps, gen)
            print(f"{family} eps={eps:.4f} done ({len(arms)} arms)", flush=True)

    pd.DataFrame(rows).to_csv(RESULTS / "raw" / f"final_{split}.csv", index=False)
    pd.DataFrame(private).to_csv(RESULTS / "private" / f"final_{split}_completions.csv", index=False)
    print(f"\nwrote {len(rows)} rows")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
