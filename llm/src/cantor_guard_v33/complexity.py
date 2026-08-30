"""V3.3 PHASE 2/3 -- five separate complexity axes under one canonical codec.

The harness (sections 4, 8, 10, 29) is emphatic about three things, and this
module enforces all three:

  * the five axes are NEVER summed into one number called "complexity"
  * we do not claim to measure Kolmogorov complexity; we measure DESCRIPTION
    LENGTH UNDER A FIXED CODING SCHEME, identical for every family
  * SYMBOLIC representation size and MATERIALISED evaluation size are
    different quantities and are reported separately

The axes:
  C1 explicit parameter count
  C2 recursive/procedural generator description bits  (canonical codec)
  C3 serialized instance bytes (raw and gzip -- reported, never decisive)
  C4 construction / extension cost
  C5 certification obligations  (in certificates.py)

A warning that the results bear out: a seeded shuffle is ALSO short to
describe -- algorithm + seed + n. Any claim that Cantor is special because it
compresses must survive comparison against seeded procedural controls
(harness STOP B), and description length alone does not distinguish them.
"""
from __future__ import annotations
import gzip, json, math, pathlib
from dataclasses import dataclass, asdict
import numpy as np

SPEC = json.loads((pathlib.Path(__file__).resolve().parents[3]
                   / "configs/v3_3/encoding.json").read_text())
B = SPEC["bits"]
OP = SPEC["opcodes"]
EL = SPEC["energy_laws"]

__all__ = ["ControllerDesc", "describe", "FAMILIES", "canonical_bits",
           "materialised_words", "SPEC"]


@dataclass
class ControllerDesc:
    family: str
    model: str                 # D1 explicit / D2 procedural+seed / D3 recursive
    n: int
    n_components: int          # materialised gap count (2^n - 1 for Cantor)
    canonical_bits: int        # C2 -- the primary description metric
    explicit_params: int       # C1
    ast_nodes: int             # generator syntax-tree size
    storage_words_symbolic: int   # what a point query must hold resident
    materialised_words: int       # what an explicit evaluator must hold
    exact_scale_transfer: bool    # does an exact cross-scale identity hold?
    serialized_bytes: int = 0
    gzip_bytes: int = 0

    def as_row(self) -> dict:
        return asdict(self)


def canonical_bits(fields: list[str], n_components: int = 0,
                   n_weights: int = 0) -> int:
    """Bit length under the frozen codec. Same accounting for every family."""
    total = 0
    for f in fields:
        if f.startswith("per_component"):
            total += n_components * 3 * B["float"]
        elif f.startswith("per_weight"):
            total += n_weights * B["float"]
        else:
            kind = f.split(":")[0]
            total += B[kind]
    return total


def materialised_words(n_components: int) -> int:
    """(a, b, e) per component -- what an explicit layout must hold."""
    return 3 * n_components


def _serialize(payload: dict) -> tuple[int, int]:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return len(raw), len(gzip.compress(raw, mtime=0))


def describe(family: str, n: int, *, E0: float = 1.0, seed: int | None = None,
             ifs=None, weights=None) -> ControllerDesc:
    """Build the canonical description record for one controller.

    `ifs` is an IFSSpec for recursive families. Cantor is the (2, 1/3) case and
    gets NO special treatment here -- it is encoded with the same RECURSIVE_IFS
    opcode as every other recursive family.
    """
    from .general_recursive import IFSSpec

    if family in ("cantor_recursive", "recursive_non_cantor"):
        spec = ifs if ifs is not None else IFSSpec(2, 1.0 / 3.0)
        nc = spec.n_components(n)
        bits = canonical_bits(SPEC["fields"]["RECURSIVE_IFS"])
        payload = {"op": OP["RECURSIVE_IFS"], "n": n, "E0": E0,
                   "law": EL["PER_LEVEL_E0_OVER_COUNT"], "b": spec.b,
                   "rho": spec.rho}
        d = ControllerDesc(family, "D3", n, nc, bits, explicit_params=6,
                           ast_nodes=9, storage_words_symbolic=4,
                           materialised_words=materialised_words(nc),
                           exact_scale_transfer=True)

    elif family == "shuffled_seeded":
        nc = (1 << n) - 1
        bits = canonical_bits(SPEC["fields"]["PROCEDURAL_SEEDED"])
        payload = {"op": OP["PROCEDURAL_SEEDED"], "n": n, "E0": E0,
                   "law": EL["PER_LEVEL_E0_OVER_COUNT"], "alg": 1,
                   "seed": int(seed)}
        d = ControllerDesc(family, "D2", n, nc, bits, explicit_params=5,
                           ast_nodes=8, storage_words_symbolic=materialised_words(nc),
                           materialised_words=materialised_words(nc),
                           exact_scale_transfer=False)

    elif family == "center_anchored_seeded":
        nc = (1 << n) - 1
        bits = canonical_bits(SPEC["fields"]["PROCEDURAL_SEEDED"])
        payload = {"op": OP["PROCEDURAL_SEEDED"], "n": n, "E0": E0,
                   "law": EL["PER_LEVEL_E0_OVER_COUNT"], "alg": 2,
                   "seed": int(seed)}
        d = ControllerDesc(family, "D2", n, nc, bits, explicit_params=5,
                           ast_nodes=8, storage_words_symbolic=materialised_words(nc),
                           materialised_words=materialised_words(nc),
                           exact_scale_transfer=False)

    elif family == "periodic_procedural":
        nc = (1 << n) - 1
        bits = canonical_bits(SPEC["fields"]["PROCEDURAL_PERIODIC"])
        payload = {"op": OP["PROCEDURAL_PERIODIC"], "n": n, "E0": E0,
                   "law": EL["PER_LEVEL_E0_OVER_COUNT"]}
        # A periodic layout IS closed-form addressable: index = floor(r/period).
        d = ControllerDesc(family, "D2", n, nc, bits, explicit_params=4,
                           ast_nodes=6, storage_words_symbolic=4,
                           materialised_words=materialised_words(nc),
                           exact_scale_transfer=False)

    elif family == "shuffled_explicit":
        nc = (1 << n) - 1
        bits = canonical_bits(SPEC["fields"]["EXPLICIT_LIST"], n_components=nc)
        payload = {"op": OP["EXPLICIT_LIST"], "n": n, "E0": E0, "nc": nc,
                   "comp": [[0.0, 0.0, 0.0]] * min(nc, 4096)}
        d = ControllerDesc(family, "D1", n, nc, bits,
                           explicit_params=3 * nc, ast_nodes=3 * nc + 4,
                           storage_words_symbolic=materialised_words(nc),
                           materialised_words=materialised_words(nc),
                           exact_scale_transfer=False)

    elif family == "learned_minimax_explicit":
        nc = (1 << n) - 1
        nw = 8
        bits = canonical_bits(SPEC["fields"]["EXPLICIT_WEIGHTS"], n_weights=nw)
        payload = {"op": OP["EXPLICIT_WEIGHTS"], "n": n, "E0": E0, "nw": nw,
                   "w": [0.0] * nw}
        d = ControllerDesc(family, "D1", n, nc, bits, explicit_params=nw + 3,
                           ast_nodes=nw + 6,
                           storage_words_symbolic=materialised_words(nc),
                           materialised_words=materialised_words(nc),
                           exact_scale_transfer=False)
    else:
        raise ValueError(f"unknown family {family}")

    d.serialized_bytes, d.gzip_bytes = _serialize(payload)
    return d


FAMILIES = ["cantor_recursive", "recursive_non_cantor", "periodic_procedural",
            "shuffled_seeded", "center_anchored_seeded", "shuffled_explicit",
            "learned_minimax_explicit"]
