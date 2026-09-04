"""Run frozen no-attack benign utility generation for all acting arms."""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from cantor_guard.io import seed_everything  # noqa: E402
from cantor_guard.models import load_model  # noqa: E402
from cantor_guard_v351.p0_generation import generate_defended  # noqa: E402
from _common import (RESULTS, build_arms, frozen_actuator, frozen_sensor,
                     require_freeze, sha256, write_json)  # noqa: E402
from _final_generation import (append_and_checkpoint, condition_count, load_partial,
                               records_from_generation)  # noqa: E402


def rho_for(arm):
    if not arm.startswith("CANTOR_"):
        return None
    value = arm.removeprefix("CANTOR_")
    return 1/3 if value == "1/3" else float(value)


def main() -> None:
    freeze = require_freeze()
    complete = RESULTS / "tables/final_benign_generation.json"
    if complete.exists():
        raise SystemExit("final benign generation already complete; regeneration forbidden")
    raw_path = RESULTS / "raw/utility_D_final_v351_benign.csv"
    private_path = RESULTS / "private/utility_D_final_v351_benign_completions.csv"
    prompts = pd.read_csv(RESULTS / "cache/D_final_v351_benign.csv")
    if len(prompts) != 80:
        raise SystemExit("final benign set must contain exactly 80 inherited prompts")
    raw, private = load_partial(raw_path, private_path)
    seed_everything(int(freeze["generation"]["seed"]))
    sensor, actuator = frozen_sensor(), frozen_actuator()
    arms = {"NO_CONTROLLER": None, **build_arms(freeze, sensor, actuator, include_attack_only=False)}
    bundle = load_model(freeze["sensor"]["model"])
    for arm, controller in arms.items():
        count = condition_count(raw, family="NONE", epsilon=0.0, arm=arm)
        if count == len(prompts):
            continue
        if count:
            raise RuntimeError(f"partial benign arm is not resumable: {arm} n={count}")
        generation = generate_defended(
            bundle, prompts.prompt.astype(str).tolist(), layer=14,
            controller=controller, sensor=sensor,
            max_new_tokens=int(freeze["generation"]["max_new_tokens"]),
            batch_size=int(freeze["generation"]["batch_size"]),
        )
        new_raw, new_private = records_from_generation(
            prompts, arm=arm, family="NONE", epsilon=0.0,
            rho=rho_for(arm), generation=generation,
            W_R=float(freeze["risk_window"]["W_R"]),
        )
        raw, private = append_and_checkpoint(raw, private, new_raw, new_private, raw_path, private_path)
        print(f"benign arm={arm} complete rows={len(raw)}/{len(arms)*len(prompts)}", flush=True)
    expected = len(arms) * len(prompts)
    if len(raw) != expected:
        raise RuntimeError(f"incomplete benign factorial: {len(raw)} != {expected}")
    write_json(complete, {
        "split": "D_final_v351_benign", "n_prompts": len(prompts),
        "arms": list(arms), "expected_rows": expected, "observed_rows": len(raw),
        "decoding": "greedy deterministic", "max_new_tokens": freeze["generation"]["max_new_tokens"],
        "raw_pre_scoring_sha256": sha256(raw_path),
        "private_pre_scoring_sha256": sha256(private_path),
        "status": "FINAL_BENIGN_GENERATION_COMPLETE",
    })


if __name__ == "__main__":
    main()
