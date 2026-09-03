"""Phase 4 -- the deployment state distribution, with no generation at all.

Only residuals are needed: the attack is a fixed vector added to h, so every
attacked sensor distance and Cantor cell follows arithmetically. Nothing is
decoded, so no output, label or endpoint can influence the budget.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from cantor_guard.io import seed_everything  # noqa: E402
from cantor_guard.models import load_model  # noqa: E402
from cantor_guard_v31.generation31 import chat_prompt  # noqa: E402
from cantor_guard_v335a.p0_residual import last_valid_index  # noqa: E402
from cantor_guard.models import decoder_layers  # noqa: E402

from _common import RESULTS, frozen_sensor, write_json  # noqa: E402

SPLIT = "D_budget_attacked_r"


@torch.no_grad()
def clean_residuals(bundle, prompts, *, layer: int, batch_size: int = 8) -> np.ndarray:
    out = []
    for start in range(0, len(prompts), batch_size):
        chunk = list(prompts[start : start + batch_size])
        enc = bundle.tokenizer([chat_prompt(bundle, p) for p in chunk],
                               return_tensors="pt", padding=True).to(bundle.device)
        idx = last_valid_index(enc["attention_mask"])
        store = {}

        def hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            arange = torch.arange(h.shape[0], device=h.device)
            store["h"] = h.float()[arange, idx, :].detach().cpu().numpy().astype(np.float64)
            return output

        handle = decoder_layers(bundle)[layer].register_forward_hook(hook)
        try:
            bundle.model(**enc)
        finally:
            handle.remove()
        out.append(store["h"])
    return np.concatenate(out, axis=0)


def main() -> None:
    seed_everything(20260903)
    prompts = pd.read_csv(RESULTS / "cache" / f"{SPLIT}.csv")
    bundle = load_model("qwen2.5-0.5b-instruct")
    H = clean_residuals(bundle, prompts.prompt.tolist(), layer=14)
    np.save(RESULTS / "cache" / f"h_{SPLIT}.npy", H)
    sensor = frozen_sensor()
    d = np.atleast_1d(sensor.distance(H))
    write_json(RESULTS / "tables" / "budget_states_collected.json", {
        "split": SPLIT, "n": int(H.shape[0]), "d_model": int(H.shape[1]),
        "generation_performed": False, "labels_consulted": False,
        "mean_d": float(d.mean()), "sd_d": float(d.std(ddof=1)),
        "mean_h_norm": float(np.linalg.norm(H, axis=1).mean()),
    })
    print(f"{SPLIT}: n={H.shape[0]} d_model={H.shape[1]} mean_d={d.mean():+.3f} "
          f"sd={d.std(ddof=1):.3f} |h|={np.linalg.norm(H,axis=1).mean():.2f}")
    print("no generation, no labels")


if __name__ == "__main__":
    main()
