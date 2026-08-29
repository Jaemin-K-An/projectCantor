"""Deterministic seeding, provenance, and result I/O.

V1 used Julia's `hash()` for per-condition seeds. Python's `hash()` is salted
per process (PYTHONHASHSEED), so it is NOT reproducible across runs — the
harness explicitly forbids it (§46). Everything here derives seeds from
SHA-256 instead.
"""
from __future__ import annotations
import hashlib, json, os, platform, subprocess, sys, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
V2_RAW = ROOT / "results" / "v2" / "raw"
V2_TAB = ROOT / "results" / "v2" / "tables"
V2_PRIVATE = ROOT / "results" / "v2" / "private"      # gitignored; harmful text only
V2_CACHE = ROOT / "results" / "v2" / "cache"          # gitignored; activations
FIG = ROOT / "figures" / "v2"
for _p in (V2_RAW, V2_TAB, V2_PRIVATE, V2_CACHE, FIG):
    _p.mkdir(parents=True, exist_ok=True)


def stable_seed(*parts: Any, bits: int = 31) -> int:
    """A reproducible seed from any tuple of values, via SHA-256.

    Deterministic across processes, machines and Python versions.
    """
    s = "\x1f".join(repr(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "big") % (1 << bits)


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU + MPS/CUDA) from one integer."""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_info() -> dict:
    try:
        c = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        d = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                           capture_output=True, text=True, timeout=10)
        return {"git_commit": c.stdout.strip() or "nogit",
                "git_dirty": bool(d.stdout.strip())}
    except Exception:
        return {"git_commit": "nogit", "git_dirty": None}


def provenance(**extra: Any) -> dict:
    """The metadata block attached to every V2 result table (§47)."""
    import torch
    p = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device_mps": bool(getattr(torch.backends, "mps", None) and
                           torch.backends.mps.is_available()),
        "device_cuda": torch.cuda.is_available(),
        **git_info(),
    }
    try:
        import transformers
        p["transformers"] = transformers.__version__
    except Exception:
        pass
    p.update({k: (v if isinstance(v, (int, float, str, bool)) else str(v))
              for k, v in extra.items()})
    return p


def write_table(df, name: str, *, raw: bool = True, meta: dict | None = None,
                overwrite: bool = True) -> Path:
    """Write a result table plus a `.meta.json` provenance sidecar."""
    out = (V2_RAW if raw else V2_TAB) / name
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass overwrite=True deliberately")
    df.to_csv(out, index=False)
    with open(str(out) + ".meta.json", "w") as f:
        json.dump({**provenance(), **(meta or {})}, f, indent=2)
    print(f"[io] wrote {len(df)} rows -> {out}")
    return out


# --------------------------------------------------------------------------
# harmful-text hygiene (§26, §45)
# --------------------------------------------------------------------------

_TRACKED_FORBIDDEN = ("completion", "response", "generation", "output_text", "text")


def assert_no_raw_completions(df, *, allow: tuple[str, ...] = ()) -> None:
    """Guard: a table destined for the repository must not carry model text.

    Raw completions to harmful prompts stay in `results/v2/private/`, which is
    gitignored. Tracked tables carry prompt IDs, hashes and scalar scores only.
    """
    bad = [c for c in df.columns
           if any(k in c.lower() for k in _TRACKED_FORBIDDEN) and c not in allow]
    if bad:
        raise ValueError(
            f"refusing to write model text to a tracked table: columns {bad}. "
            f"Write them to results/v2/private/ instead.")
