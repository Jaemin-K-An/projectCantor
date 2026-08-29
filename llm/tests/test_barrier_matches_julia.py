"""Cross-language check: the Python and Julia barrier must agree to 1e-12.

Without this, "the same controller was used in the synthetic study and in the
LLM study" would be an assertion rather than a fact. The Julia reference is
produced by scripts/v2/export_barrier_reference.jl.

Randomised families are excluded: their layouts depend on each language's RNG,
so only the DETERMINISTIC families can be compared pointwise. Their agreement
is what pins the shared field/potential arithmetic.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from cantor_guard.cantor_barrier import build_layout

REF = Path(__file__).resolve().parents[2] / "results" / "v2" / "raw" / "barrier_reference.csv"
FAM = {"B1_constant": "L1_constant", "B2_central": "L2_central",
       "B3_periodic": "L3_periodic", "B7_cantor": "L7_cantor"}


@pytest.mark.skipif(not REF.exists(),
                    reason="run scripts/v2/export_barrier_reference.jl first")
def test_python_matches_julia():
    ref = pd.read_csv(REF)
    for (fam, n), g in ref.groupby(["family", "n"]):
        L = build_layout(FAM[fam], int(n), float(g.E0.iloc[0]))
        r = g.r.to_numpy()
        assert np.allclose(L.field(r), g.field.to_numpy(), atol=1e-12), f"{fam} n={n} field"
        assert np.allclose(L.potential(r), g.potential.to_numpy(), atol=1e-12), f"{fam} n={n} V"
