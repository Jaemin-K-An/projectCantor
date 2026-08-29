"""V3.1 result I/O — keeps V3.1 outputs out of the V2/V3 namespaces."""
from __future__ import annotations
import json
from pathlib import Path
from cantor_guard.io import ROOT, provenance, assert_no_raw_completions
V31_RAW = ROOT/"results"/"v3_1"/"raw"; V31_TAB = ROOT/"results"/"v3_1"/"tables"
V31_PRIVATE = ROOT/"results"/"v3_1"/"private"; V31_CACHE = ROOT/"results"/"v3_1"/"cache"
FIG31 = ROOT/"figures"/"v3_1"
for _p in (V31_RAW, V31_TAB, V31_PRIVATE, V31_CACHE, FIG31): _p.mkdir(parents=True, exist_ok=True)
def write_v31(df, name, *, raw=True, meta=None):
    assert_no_raw_completions(df)
    out = (V31_RAW if raw else V31_TAB)/name
    df.to_csv(out, index=False)
    json.dump({**provenance(), **(meta or {})}, open(str(out)+".meta.json","w"), indent=2)
    print(f"[v31-io] wrote {len(df)} rows -> {out}")
    return out
