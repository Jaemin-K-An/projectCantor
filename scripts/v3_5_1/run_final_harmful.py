"""Run the frozen 2-family x 14-epsilon x 9-arm harmful factorial."""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from cantor_guard.io import seed_everything  # noqa: E402
from cantor_guard.models import load_model  # noqa: E402
from cantor_guard_v340.attack import attack_v, attack_w  # noqa: E402
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
    complete = RESULTS / "tables/final_harmful_generation.json"
    if complete.exists():
        raise SystemExit("final harmful factorial already complete; regeneration forbidden")
    raw_path = RESULTS / "raw/final_D_final_v351_harmful.csv"
    private_path = RESULTS / "private/final_D_final_v351_harmful_completions.csv"
    prompts = pd.read_csv(RESULTS / "cache/D_final_v351_harmful.csv")
    if len(prompts) != 200:
        raise SystemExit("final harmful set must contain exactly 200 inherited prompts")
    raw, private = load_partial(raw_path, private_path)
    seed_everything(int(freeze["generation"]["seed"]))
    sensor, actuator = frozen_sensor(), frozen_actuator()
    arms = build_arms(freeze, sensor, actuator, include_attack_only=True)
    grid = list(map(float, freeze["attacks"]["grid_absolute"]))
    bundle = load_model(freeze["sensor"]["model"])

    for family in ("ATTACK_V", "ATTACK_W"):
        for epsilon in grid:
            delta = attack_v(actuator, epsilon) if family == "ATTACK_V" else attack_w(sensor, epsilon, sign=-1)
            for arm, controller in arms.items():
                count = condition_count(raw, family=family, epsilon=epsilon, arm=arm)
                if count == len(prompts):
                    continue
                if count:
                    raise RuntimeError(f"partial condition is not resumable: {family} {epsilon} {arm} n={count}")

                # At epsilon=0 the two attack operators are exactly identical.
                # Reuse the already generated ATTACK_V block, changing metadata only.
                if family == "ATTACK_W" and epsilon == 0.0:
                    source_mask = ((raw.family == "ATTACK_V") & (raw.arm == arm) &
                                   raw.epsilon.astype(float).eq(0.0))
                    source_private_mask = ((private.family == "ATTACK_V") & (private.arm == arm) &
                                           private.epsilon.astype(float).eq(0.0))
                    if int(source_mask.sum()) != len(prompts) or int(source_private_mask.sum()) != len(prompts):
                        raise RuntimeError("zero-epsilon ATTACK_V source block missing")
                    new_raw = raw.loc[source_mask].copy()
                    new_private = private.loc[source_private_mask].copy()
                    for frame in (new_raw, new_private):
                        frame["family"] = "ATTACK_W"
                        frame["computational_reuse"] = True
                        frame["reused_from"] = "ATTACK_V@0"
                else:
                    generation = generate_defended(
                        bundle, prompts.prompt.astype(str).tolist(), layer=14,
                        attack_delta=delta, controller=controller, sensor=sensor,
                        max_new_tokens=int(freeze["generation"]["max_new_tokens"]),
                        batch_size=int(freeze["generation"]["batch_size"]),
                    )
                    new_raw, new_private = records_from_generation(
                        prompts, arm=arm, family=family, epsilon=epsilon,
                        rho=rho_for(arm), generation=generation,
                        W_R=float(freeze["risk_window"]["W_R"]),
                    )
                raw, private = append_and_checkpoint(
                    raw, private, new_raw, new_private, raw_path, private_path)
                print(f"{family} epsilon={epsilon:.12g} arm={arm} complete "
                      f"rows={len(raw)}/{2*len(grid)*len(arms)*len(prompts)}", flush=True)

    expected = 2 * len(grid) * len(arms) * len(prompts)
    if len(raw) != expected or len(private) != expected:
        raise RuntimeError(f"incomplete harmful factorial: {len(raw)} != {expected}")
    write_json(complete, {
        "split": "D_final_v351_harmful", "n_prompts": len(prompts),
        "families": ["ATTACK_V", "ATTACK_W"], "grid": grid,
        "arms": list(arms), "expected_rows": expected, "observed_rows": len(raw),
        "zero_epsilon_cross_family_reuse": True,
        "decoding": "greedy deterministic", "max_new_tokens": freeze["generation"]["max_new_tokens"],
        "raw_pre_scoring_sha256": sha256(raw_path),
        "private_pre_scoring_sha256": sha256(private_path),
        "status": "FINAL_HARMFUL_GENERATION_COMPLETE",
    })


if __name__ == "__main__":
    main()
