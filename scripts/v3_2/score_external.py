"""V3.2 PHASE 12 -- score the SAME completions with the independent evaluator.

Run after the final test. Reads the persisted completions, scores them with the
third-party rejection classifier, and merges the result into the tracked table
as `safe_ext`. No completion text is written to the tracked table.
"""
import argparse, sys, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
from cantor_guard_v32.evaluators import (ExternalRefusalEvaluator,
                                         external_safe_score, agreement)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
a = ap.parse_args()

RAW = pathlib.Path("results/v3_2/raw") / f"v32_final_{a.model}.csv"
TXT = pathlib.Path("results/v3_2/private") / f"v32_completions_{a.model}.csv"
df, tx = pd.read_csv(RAW), pd.read_csv(TXT)
tx["text"] = tx["text"].fillna("").astype(str)
KEY = ["family", "layout_seed", "attack", "delta", "eps", "pid"]
print(f"{len(df)} scored rows, {len(tx)} completions")

ev = ExternalRefusalEvaluator()
print(f"scoring with {ev.spec['repo']} @ {ev.revision[:12]} ...", flush=True)
tx["safe_ext"] = external_safe_score(list(tx.text), ev)

# Deduplicate defensively: the runner appends and may have resumed.
tx = tx.drop_duplicates(subset=KEY, keep="last")
before = len(df)
df = df.drop_duplicates(subset=KEY, keep="last").merge(
    tx[KEY + ["safe_ext"]], on=KEY, how="left")
print(f"merged: {before} -> {len(df)} rows, "
      f"{df.safe_ext.isna().sum()} unscored")
ok = df.dropna(subset=["safe_ext"])
print("agreement lex32 vs ext:",
      {k: (round(v, 4) if isinstance(v, float) else v)
       for k, v in agreement(ok.safe_lex32.values, ok.safe_ext.values).items()})
print("\nmean safety by family:")
print(df.groupby("family")[["safe_lex32", "safe_ext"]].mean()
        .sort_values("safe_lex32", ascending=False).round(4).to_string())
df.to_csv(RAW, index=False)
print(f"\nwrote {RAW}")
