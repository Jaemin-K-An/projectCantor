"""Run untouched D_final through attacked, actual P0 rho-family controllers."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.special import logsumexp

import sys
sys.path.insert(0, "llm/src")
from cantor_guard.models import load_model  # noqa: E402
from cantor_guard_v32.metrics32 import coherence32, is_refusal32, safe_score32  # noqa: E402
from cantor_guard_v335c.p0_attack_generation import generate_attacked_p0  # noqa: E402

from _common import CONFIG, RESULTS, behavioral_protocol, load_direction, read_json, rho_key
from _controllers import make_controller
from freeze_v335c import verify_freeze


def logit_metrics(clean, attacked, corrected, *, top_k: int = 10) -> list[dict]:
    clean = np.asarray(clean, dtype=np.float64)
    attacked = np.asarray(attacked, dtype=np.float64)
    corrected = np.asarray(corrected, dtype=np.float64)
    logp = clean - logsumexp(clean, axis=1, keepdims=True)
    p = np.exp(logp)
    logq_attack = attacked - logsumexp(attacked, axis=1, keepdims=True)
    logq_corrected = corrected - logsumexp(corrected, axis=1, keepdims=True)
    clean_top = np.argmax(clean, axis=1)
    attacked_top = np.argmax(attacked, axis=1)
    corrected_top = np.argmax(corrected, axis=1)
    records = []
    for i in range(clean.shape[0]):
        ctop = set(np.argpartition(clean[i], -top_k)[-top_k:])
        atop = set(np.argpartition(attacked[i], -top_k)[-top_k:])
        rtop = set(np.argpartition(corrected[i], -top_k)[-top_k:])
        records.append({
            "kl_clean_attacked": float(np.sum(p[i] * (logp[i] - logq_attack[i]))),
            "kl_clean_corrected": float(np.sum(p[i] * (logp[i] - logq_corrected[i]))),
            "attacked_top1_flip": int(attacked_top[i] != clean_top[i]),
            "corrected_top1_flip": int(corrected_top[i] != clean_top[i]),
            "attacked_topk_overlap": len(ctop & atop) / top_k,
            "corrected_topk_overlap": len(ctop & rtop) / top_k,
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    freeze = read_json(CONFIG / "PRE_ANALYSIS_FREEZE.json")
    verify_freeze(freeze)
    if (RESULTS / "raw/final_p0_cantor.csv").exists():
        raise SystemExit("STOP: final output already exists; no automatic rerun/retuning")
    protocol = behavioral_protocol()
    controller_cfg = read_json(CONFIG / "controller.json")
    attack_cfg = read_json(CONFIG / "attack_grid.json")
    prompts = pd.read_csv(RESULTS / "cache/D_final_P0_335c.csv")
    direction = load_direction(protocol)
    bundle = load_model(protocol["model"])
    rows, private = [], []
    for rho in [float(x) for x in controller_cfg["rho_family"]]:
        controller = make_controller(rho, direction)
        for epsilon in [float(x) for x in attack_cfg["common_absolute_epsilon_grid"]]:
            generated = generate_attacked_p0(
                bundle, prompts.prompt.tolist(), layer=int(protocol["layer"]), v=direction,
                controller=controller, attack_epsilon=epsilon,
                unsafe_sign=int(attack_cfg["unsafe_sign"]),
                max_new_tokens=int(protocol["max_new_tokens"]), batch_size=args.batch,
                record_first_logits=True,
            )
            mechanisms = logit_metrics(
                generated["first_logits_clean"], generated["first_logits_attacked"],
                generated["first_logits_corrected"],
            )
            for i, text in enumerate(generated["texts"]):
                policy = generated["controller_records"][i]
                row = {
                    "pid": prompts.pid.iloc[i], "rho": rho, "epsilon": epsilon,
                    "refusal_proxy": int(is_refusal32(text)),
                    "safe_score32": safe_score32(text),
                    "semantic_harmful_compliance": None,
                    "coherence": coherence32(text), "output_length": len(text.split()),
                    "degeneration": int(coherence32(text) < float(protocol["coherence_gate"]) or len(text.split()) == 0),
                    "z_clean": generated["z_clean"][i], "z_attacked": generated["z_attacked"][i],
                    "z_corrected": generated["z_corrected"][i], "q_ctrl": generated["q_ctrl"][i],
                    **{key: policy[key] for key in ("cell_kind", "cell_index", "cell_level", "action", "outside_window")},
                    **mechanisms[i],
                }
                rows.append(row)
                private.append({**row, "completion": text})
            print(f"rho={rho_key(rho)} epsilon={epsilon:.6g} n={len(prompts)}", flush=True)
    (RESULTS / "raw").mkdir(parents=True, exist_ok=True)
    (RESULTS / "private").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "raw/final_p0_cantor.csv", index=False)
    pd.DataFrame(private).to_csv(RESULTS / "private/final_p0_cantor_completions.csv", index=False)


if __name__ == "__main__":
    main()
