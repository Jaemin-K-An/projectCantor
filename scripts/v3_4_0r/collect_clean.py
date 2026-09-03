"""Clean P0 residuals + completions for a V3.4.0R split. No intervention."""
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
from cantor_guard_v340.p0_generation import clean_p0_and_generate  # noqa: E402

from _common import RESULTS, frozen_sensor, write_json  # noqa: E402


def main(splits) -> None:
    seed_everything(20260903)
    bundle = load_model("qwen2.5-0.5b-instruct")
    sensor = frozen_sensor()
    summary = {}
    for split in splits:
        frame = pd.read_csv(RESULTS / "cache" / f"{split}.csv")
        H, texts = clean_p0_and_generate(bundle, frame.prompt.tolist(), layer=14,
                                         max_new_tokens=48, batch_size=8)
        d = np.atleast_1d(sensor.distance(H))
        rows, private = [], []
        for i, text in enumerate(texts):
            coh = float(coherence32(text))
            words = len((text or "").split())
            row = {"pid": frame.pid.iloc[i], "split": split, "kind": frame.kind.iloc[i],
                   "h_norm": float(np.linalg.norm(H[i])), "d_clean": float(d[i]),
                   "refusal_proxy": int(is_refusal32(text)), "coherence": coh,
                   "output_length": words,
                   "degeneration": int(coh < 0.95 or words == 0)}
            rows.append(row)
            private.append({**row, "prompt": frame.prompt.iloc[i], "completion": text})
        table = pd.DataFrame(rows)
        table.to_csv(RESULTS / "raw" / f"clean_{split}.csv", index=False)
        pd.DataFrame(private).to_csv(RESULTS / "private" / f"clean_{split}_completions.csv", index=False)
        np.save(RESULTS / "cache" / f"h_{split}.npy", H)
        summary[split] = {"n": int(len(table)), "mean_d": float(d.mean()), "sd_d": float(d.std(ddof=1)),
                          "refusal_proxy_rate": float(table.refusal_proxy.mean()),
                          "mean_coherence": float(table.coherence.mean()),
                          "degeneration_rate": float(table.degeneration.mean())}
        print(f"{split}: n={len(table)} mean_d={d.mean():+.3f} sd={d.std(ddof=1):.3f} "
              f"lex_refusal={table.refusal_proxy.mean():.3f} coh={table.coherence.mean():.3f}", flush=True)
    write_json(RESULTS / "tables" / f"clean_collection_{'_'.join(splits)[:60]}.json",
               {"intervention": "none", "layer": 14, "splits": summary})


if __name__ == "__main__":
    main(sys.argv[1:])
