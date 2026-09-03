"""Construct an outcome-independent common absolute epsilon grid and run attack DEV."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "llm/src")
from cantor_guard.models import load_model  # noqa: E402
from cantor_guard_v32.metrics32 import coherence32, is_refusal32  # noqa: E402
from cantor_guard_v335c.cantor_geometry import epsilon_z  # noqa: E402
from cantor_guard_v335c.p0_attack_generation import generate_attacked_p0  # noqa: E402

from _common import CONFIG, RESULTS, behavioral_protocol, load_direction, read_json, rho_key, write_json
from _controllers import make_controller


def common_grid(rhos, W: float) -> list[float]:
    cert = {rho_key(rho): float(epsilon_z(rho, W)) for rho in rhos}
    minimum, maximum = min(cert.values()), max(cert.values())
    cantor = cert["1/3"]
    primary = [cert[rho_key(rho)] for rho in (0.30, 0.36, 0.40)]
    values = [0.0, 0.5 * minimum, 0.9 * minimum, 0.99 * cantor, cantor, 1.01 * cantor, 1.10 * cantor, 1.25 * maximum, 1.50 * maximum]
    for value in primary:
        values += [0.99 * value, 1.01 * value]
    return sorted({round(float(value), 12) for value in values})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    if args.summarize_only:
        frame = pd.read_csv(RESULTS / "raw/p0_attack_dev.csv")
        summary = []
        for rho, group in frame.groupby("rho"):
            q = group.q_ctrl.to_numpy(float)
            summary.append({
                "rho": float(rho), "q_rms": float(np.sqrt(np.mean(q**2))),
                "q_mean": float(q.mean()), "q_p95": float(np.quantile(q, 0.95)),
                "q_max": float(q.max()), "guard_frequency": float(np.mean(group.cell_kind == "guard")),
                "leaf_frequency": float(np.mean(group.cell_kind == "leaf")),
                "outside_window_frequency": float(group.outside_window.mean()),
                "coherence": float(group.coherence.mean()),
                "refusal_proxy": float(group.refusal_proxy.mean()),
            })
        write_json(RESULTS / "tables/p0_attack_dev_summary.json", {
            "n_rows": len(frame), "n_prompts": int(frame.pid.nunique()),
            "n_rho": int(frame.rho.nunique()), "n_epsilon": int(frame.epsilon.nunique()),
            "grid_changed_from_outcomes": False, "rho": summary,
        })
        print(pd.DataFrame(summary).to_string(index=False))
        return
    protocol = behavioral_protocol()
    controller_cfg = read_json(CONFIG / "controller.json")
    if not controller_cfg.get("eta_by_rho"):
        raise SystemExit("STOP: controller budgets are not frozen")
    rhos = [float(x) for x in controller_cfg["rho_family"]]
    W = float(controller_cfg["W"])
    grid = common_grid(rhos, W)
    attack_cfg = read_json(CONFIG / "attack_grid.json")
    attack_cfg["common_absolute_epsilon_grid"] = grid
    attack_cfg["certificate_by_rho"] = {rho_key(rho): float(epsilon_z(rho, W)) for rho in rhos}
    attack_cfg["selected_without_outcomes"] = True
    write_json(CONFIG / "attack_grid.json", attack_cfg)
    print("common absolute epsilon grid:", grid)
    if args.build_only:
        return
    prompts = pd.read_csv(RESULTS / "cache/D_attack_dev_335c.csv")
    direction = load_direction(protocol)
    bundle = load_model(protocol["model"])
    rows = []
    private = []
    for rho in rhos:
        controller = make_controller(rho, direction)
        for epsilon in grid:
            generated = generate_attacked_p0(
                bundle, prompts.prompt.tolist(), layer=int(protocol["layer"]), v=direction,
                controller=controller, attack_epsilon=epsilon,
                unsafe_sign=int(attack_cfg["unsafe_sign"]),
                max_new_tokens=int(protocol["max_new_tokens"]), batch_size=args.batch,
            )
            for i, text in enumerate(generated["texts"]):
                record = generated["controller_records"][i]
                row = {
                    "pid": prompts.pid.iloc[i], "rho": rho, "epsilon": epsilon,
                    "epsilon_over_certificate": epsilon / float(epsilon_z(rho, W)),
                    "refusal_proxy": int(is_refusal32(text)),
                    "semantic_harmful_compliance": None,
                    "coherence": coherence32(text), "output_length": len(text.split()),
                    "z_clean": generated["z_clean"][i], "z_attacked": generated["z_attacked"][i],
                    "z_corrected": generated["z_corrected"][i], "q_ctrl": generated["q_ctrl"][i],
                    **{key: record[key] for key in ("cell_kind", "cell_index", "cell_level", "action", "outside_window")},
                }
                rows.append(row)
                private.append({**row, "completion": text})
            print(f"rho={rho_key(rho)} eps={epsilon:.5g} refusal={np.mean([r['refusal_proxy'] for r in rows[-len(prompts):]]):.3f}", flush=True)
    (RESULTS / "raw").mkdir(parents=True, exist_ok=True)
    (RESULTS / "private").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "raw/p0_attack_dev.csv", index=False)
    pd.DataFrame(private).to_csv(RESULTS / "private/p0_attack_dev_completions.csv", index=False)


if __name__ == "__main__":
    main()
