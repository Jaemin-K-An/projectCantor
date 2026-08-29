"""V3 result I/O — keeps V3 outputs in results/v3/, never in the V2 namespace."""
from __future__ import annotations
import json
from pathlib import Path
from cantor_guard.io import ROOT, provenance, assert_no_raw_completions

V3_RAW = ROOT / "results" / "v3" / "raw"
V3_TAB = ROOT / "results" / "v3" / "tables"
V3_PRIVATE = ROOT / "results" / "v3" / "private"
V3_CACHE = ROOT / "results" / "v3" / "cache"
FIG3 = ROOT / "figures" / "v3"
for _p in (V3_RAW, V3_TAB, V3_PRIVATE, V3_CACHE, FIG3):
    _p.mkdir(parents=True, exist_ok=True)


def write_v3(df, name: str, *, raw: bool = True, meta: dict | None = None) -> Path:
    assert_no_raw_completions(df)
    out = (V3_RAW if raw else V3_TAB) / name
    df.to_csv(out, index=False)
    with open(str(out) + ".meta.json", "w") as f:
        json.dump({**provenance(), **(meta or {})}, f, indent=2)
    print(f"[v3-io] wrote {len(df)} rows -> {out}")
    return out
