"""Benchmark loading with pinned dataset revisions, and LEAKAGE-SAFE splits.

Sources (all ungated, revisions pinned at freeze time):
  JailbreakBench/JBB-Behaviors   harmful + benign behaviours (Chao et al. 2024)
  natolambert/xstest-v2-copy     XSTest over-refusal probes (Röttger et al. 2024)
  tatsu-lab/alpaca               benign instructions for utility

Files are fetched individually with `hf_hub_download` at a pinned revision
rather than through `datasets.load_dataset`, so the exact bytes are
reproducible and no dataset script is executed.

SPLITTING (harness §29): prompts are NEVER split by raw string. JBB carries a
`Behavior` and a `Category` per row; splits are made on a GOAL GROUP derived
from the behaviour label, so paraphrases of the same underlying goal cannot
straddle the calibration/dev/test boundary. The split and its seed are saved.
"""
from __future__ import annotations
import hashlib, re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

DATASET_REGISTRY = {
    "jbb": {"repo": "JailbreakBench/JBB-Behaviors",
            "revision": "886acc352a31533ffbcf4ef22c744658688086fc",
            "harmful": "data/harmful-behaviors.csv",
            "benign": "data/benign-behaviors.csv"},
    "xstest": {"repo": "natolambert/xstest-v2-copy",
               "revision": "b71afe2a6d10e5a6254ea8bcb006c48b095a15d5",
               "file": "data/prompts-00000-of-00001.parquet"},
    "alpaca": {"repo": "tatsu-lab/alpaca",
               "revision": "dce01c9b08f87459cf36a430d809084718273017",
               "file": "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"},
}


def _get(repo, revision, filename):
    return hf_hub_download(repo_id=repo, filename=filename, revision=revision,
                           repo_type="dataset")


def prompt_id(text: str) -> str:
    """Stable short id for a prompt: the first 16 hex chars of its SHA-256.

    Tracked result tables carry this, never the harmful prompt text itself."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def _goal_group(label: str) -> str:
    """Normalised goal key: lowercase, punctuation stripped, stopwords removed.

    Rows sharing a key are treated as the same underlying goal and are kept on
    the same side of every split."""
    t = re.sub(r"[^a-z0-9 ]", " ", str(label).lower())
    stop = {"a", "an", "the", "for", "of", "to", "how", "and", "or", "in", "on",
            "with", "that", "this", "write", "give", "explain", "provide"}
    toks = [w for w in t.split() if w and w not in stop]
    return " ".join(sorted(set(toks))[:6]) or t.strip()


def load_jbb() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(harmful, benign) JBB behaviours with `prompt`, `goal_group`, `pid`."""
    spec = DATASET_REGISTRY["jbb"]
    out = []
    for key in ("harmful", "benign"):
        df = pd.read_csv(_get(spec["repo"], spec["revision"], spec[key]))
        col = "Goal" if "Goal" in df.columns else df.columns[1]
        beh = "Behavior" if "Behavior" in df.columns else col
        d = pd.DataFrame({
            "prompt": df[col].astype(str),
            "behavior": df[beh].astype(str),
            "category": df["Category"].astype(str) if "Category" in df else "unknown",
            "source": df["Source"].astype(str) if "Source" in df else "jbb",
        })
        d["goal_group"] = d["behavior"].map(_goal_group)
        d["pid"] = d["prompt"].map(prompt_id)
        d["benchmark"] = "jbb_" + key
        d["is_harmful"] = (key == "harmful")
        out.append(d.reset_index(drop=True))
    return out[0], out[1]


def load_xstest() -> pd.DataFrame:
    """XSTest: `safe` prompts that look unsafe (over-refusal probes) and
    genuinely `unsafe` contrast prompts."""
    spec = DATASET_REGISTRY["xstest"]
    df = pd.read_parquet(_get(spec["repo"], spec["revision"], spec["file"]))
    pc = "prompt" if "prompt" in df.columns else df.columns[0]
    tc = "type" if "type" in df.columns else None
    d = pd.DataFrame({"prompt": df[pc].astype(str)})
    d["category"] = df[tc].astype(str) if tc else "unknown"
    # XSTest 'contrast_*' types are the genuinely unsafe ones
    d["is_harmful"] = d["category"].str.startswith("contrast")
    d["behavior"] = d["category"]
    d["goal_group"] = d["category"].map(_goal_group)
    d["pid"] = d["prompt"].map(prompt_id)
    d["benchmark"] = "xstest"
    d["source"] = "xstest"
    return d.reset_index(drop=True)


def load_alpaca(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """A fixed benign instruction sample for utility measurement."""
    spec = DATASET_REGISTRY["alpaca"]
    df = pd.read_parquet(_get(spec["repo"], spec["revision"], spec["file"]))
    df = df[df["input"].astype(str).str.len() == 0]          # single-turn only
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    d = pd.DataFrame({"prompt": df.iloc[idx]["instruction"].astype(str).values})
    d["behavior"] = "alpaca"; d["category"] = "benign_utility"
    d["goal_group"] = ["alpaca_%d" % i for i in range(len(d))]
    d["pid"] = d["prompt"].map(prompt_id)
    d["benchmark"] = "alpaca"; d["is_harmful"] = False; d["source"] = "alpaca"
    return d.reset_index(drop=True)


@dataclass
class Split:
    calibration: pd.DataFrame
    dev: pd.DataFrame
    test: pd.DataFrame
    seed: int
    def summary(self) -> dict:
        return {"n_calibration": len(self.calibration), "n_dev": len(self.dev),
                "n_test": len(self.test), "split_seed": self.seed,
                "group_overlap": self.overlap()}
    def overlap(self) -> int:
        a = set(self.calibration.goal_group); b = set(self.dev.goal_group)
        c = set(self.test.goal_group)
        return len(a & b) + len(a & c) + len(b & c)


def grouped_split(df: pd.DataFrame, *, fracs=(0.35, 0.30, 0.35),
                  seed: int = 20260829) -> Split:
    """Split by `goal_group`, never by row, so no goal appears in two splits."""
    groups = sorted(df.goal_group.unique())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(groups))
    g = [groups[i] for i in perm]
    n1 = int(round(fracs[0] * len(g))); n2 = n1 + int(round(fracs[1] * len(g)))
    sets = (set(g[:n1]), set(g[n1:n2]), set(g[n2:]))
    parts = [df[df.goal_group.isin(s)].reset_index(drop=True) for s in sets]
    sp = Split(*parts, seed=seed)
    assert sp.overlap() == 0, "goal groups leaked across splits"
    return sp
