"""Phase 15 -- the confirmatory run on untouched D_final_harmful.

Every controller sees the SAME prompts, the SAME absolute attack magnitudes and
the SAME decoding; only rho differs.  The attack is injected first, the
controller observes the attacked residual, and its correction is what produces
token 1.  Nothing here is retuned: all parameters come from the frozen
PRE_ANALYSIS_FREEZE.json.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "llm/src"))
from cantor_guard.io import seed_everything  # noqa: E402
from cantor_guard.models import load_model  # noqa: E402
from cantor_guard_v32.metrics32 import coherence32, is_refusal32  # noqa: E402
from cantor_guard_v340.actuator import Actuator  # noqa: E402
from cantor_guard_v340.attack import attack_v, attack_w  # noqa: E402
from cantor_guard_v340.p0_generation import generate_defended  # noqa: E402
from cantor_guard_v340.sensor_actuator_controller import SensorActuatorCantorController  # noqa: E402
from cantor_guard_v340.sensor_distance import SensorHyperplane  # noqa: E402

from _common import CONFIG, RESULTS, read_json, rho_key  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
# Behaviour labels are attached afterwards, in one batched pass by
# score_semantic.py, so decoding is not blocked on the labeller.


def main(split: str = "D_final_harmful") -> None:
    freeze = read_json(CONFIG / "PRE_ANALYSIS_FREEZE.json")
    seed_everything(20260903)
    sensor = SensorHyperplane(np.load(RESULTS / "cache" / "sensor_w.npy"), float(freeze["sensor"]["b"]))
    actuator = Actuator(np.load(ROOT / freeze["actuator"]["file"]).astype(float).reshape(-1),
                        int(freeze["actuator"]["safe_sign"]))
    W = float(freeze["geometry"]["W"])
    leaf_actions = freeze["geometry"]["leaf_actions"]
    grid = list(freeze["attacks"]["generation_grid_absolute"])
    prompts = pd.read_csv(RESULTS / "cache" / f"{split}.csv")

    bundle = load_model("qwen2.5-0.5b-instruct")

    rows, private = [], []
    for rho in freeze["geometry"]["rho_family"]:
        key = rho_key(rho)
        controller = SensorActuatorCantorController(
            sensor=sensor, actuator=actuator, W=W, rho=float(rho),
            eta=float(freeze["budget"]["eta_per_rho"][key]), leaf_actions=leaf_actions,
        )
        conditions = [("NONE", 0.0)] + [(f, e) for f in ("ATTACK_W", "ATTACK_V") for e in grid if e > 0]
        for family, eps in conditions:
            if family == "NONE":
                delta = None
            elif family == "ATTACK_W":
                delta = attack_w(sensor, eps, sign=-1)
            else:
                delta = attack_v(actuator, eps)
            gen = generate_defended(bundle, prompts.prompt.tolist(), layer=14,
                                    attack_delta=delta, controller=controller,
                                    sensor=sensor, max_new_tokens=48, batch_size=8)
            records = gen["records"]
            for i, text in enumerate(gen["texts"]):
                coh = float(coherence32(text))
                rec = records[i] if i < len(records) else {}
                row = {
                    "pid": prompts.pid.iloc[i], "rho": float(rho), "rho_key": key,
                    "family": family, "epsilon": float(eps),
                    "d_clean": float(gen["d_clean"][i]),
                    "d_attacked": float(gen["d_attacked"][i]),
                    "d_corrected": float(gen["d_corrected"][i]),
                    "q_ctrl": float(gen["q_ctrl"][i]),
                    "cell_kind": rec.get("cell_kind"), "cell_index": rec.get("cell_index"),
                    "action": rec.get("action"), "outside_window": rec.get("outside_window"),
                    "refusal_proxy": int(is_refusal32(text)),
                    "coherence": coh, "output_length": len((text or "").split()),
                    "degeneration": int(coh < 0.95 or len((text or "").split()) == 0),
                }
                rows.append(row)
                private.append({**row, "prompt": prompts.prompt.iloc[i], "completion": text})
            print(f"rho={key:<5} {family:<9} eps={eps:.4f} "
                  f"lex_refusal={np.mean([is_refusal32(t) for t in gen['texts']]):.3f} "
                  f"q_rms={np.sqrt(np.mean(gen['q_ctrl']**2)):.4f}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "raw" / f"final_{split}.csv", index=False)
    pd.DataFrame(private).to_csv(RESULTS / "private" / f"final_{split}_completions.csv", index=False)
    print(f"\nwrote {len(rows)} rows; run score_semantic.py to attach behaviour labels")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
