"""Phase 18 (generation) -- run every controller on untouched benign prompts."""
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
from cantor_guard_v340.p0_generation import generate_defended  # noqa: E402
from cantor_guard_v340.sensor_actuator_controller import SensorActuatorCantorController  # noqa: E402
from cantor_guard_v340.sensor_distance import SensorHyperplane  # noqa: E402

from _common import CONFIG, RESULTS, read_json, rho_key  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main(split: str = "D_final_benign") -> None:
    freeze = read_json(CONFIG / "PRE_ANALYSIS_FREEZE.json")
    seed_everything(20260903)
    sensor = SensorHyperplane(np.load(RESULTS / "cache" / "sensor_w.npy"), float(freeze["sensor"]["b"]))
    actuator = Actuator(np.load(ROOT / freeze["actuator"]["file"]).astype(float).reshape(-1),
                        int(freeze["actuator"]["safe_sign"]))
    W = float(freeze["geometry"]["W"])
    prompts = pd.read_csv(RESULTS / "cache" / f"{split}.csv")
    bundle = load_model("qwen2.5-0.5b-instruct")

    rows, private = [], []
    # "NO_CONTROLLER" is the reference arm the utility gate is stated against.
    arms = [("NO_CONTROLLER", None)] + [
        (rho_key(r), SensorActuatorCantorController(
            sensor=sensor, actuator=actuator, W=W, rho=float(r),
            eta=float(freeze["budget"]["eta_per_rho"][rho_key(r)]),
            leaf_actions=freeze["geometry"]["leaf_actions"]))
        for r in freeze["geometry"]["rho_family"]
    ]
    for name, controller in arms:
        gen = generate_defended(bundle, prompts.prompt.tolist(), layer=14, attack_delta=None,
                                controller=controller, sensor=sensor, max_new_tokens=48, batch_size=8)
        records = gen["records"]
        for i, text in enumerate(gen["texts"]):
            coh = float(coherence32(text))
            words = len((text or "").split())
            rec = records[i] if i < len(records) else {}
            row = {
                "pid": prompts.pid.iloc[i], "arm": name,
                "d_clean": float(gen["d_clean"][i]),
                "q_ctrl": float(gen["q_ctrl"][i]) if len(gen["q_ctrl"]) else 0.0,
                "cell_kind": rec.get("cell_kind"), "outside_window": rec.get("outside_window"),
                "refusal_proxy": int(is_refusal32(text)),
                "coherence": coh, "output_length": words,
                "degeneration": int(coh < 0.95 or words == 0),
            }
            rows.append(row)
            private.append({**row, "prompt": prompts.prompt.iloc[i], "completion": text})
        print(f"{name:<14} false_refusal={np.mean([r['refusal_proxy'] for r in rows if r['arm']==name]):.3f} "
              f"coh={np.mean([r['coherence'] for r in rows if r['arm']==name]):.3f}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "raw" / f"utility_{split}.csv", index=False)
    pd.DataFrame(private).to_csv(RESULTS / "private" / f"utility_{split}_completions.csv", index=False)
    print(f"\nwrote {len(rows)} rows; run score_semantic.py utility_{split} then analyse_utility.py")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
