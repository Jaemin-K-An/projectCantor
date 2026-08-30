"""V3.3 -- exact scale transfer for EVERY control, periodic included.

The post-hoc fairness correction showed periodic matches Cantor on all three
preregistered structural metrics. So the question becomes: is there ANY
property that separates the recursive families from periodic? Scale transfer is
the candidate, and it must be measured on periodic too rather than assumed.

E_scale = sup_r |u_{n+1}(T_i(r)) - alpha * u_n(r)| / max(|alpha*u_n(r)|, 1)
"""
import sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
from cantor_guard.cantor_barrier import (cantor_gap_list, BarrierLayout,
                                         build_layout)
from cantor_guard_v33.general_recursive import IFSSpec, CANTOR, SymbolicIFS
from cantor_guard_v33.symbolic_cantor import N_GAPS

E0, SEED = 1.0, 20260831
NON_CANTOR = IFSSpec(2, 0.28)
r = np.linspace(1e-9, 1 - 1e-9, 4000)
rows = []

for n in range(2, 14):
    # recursive families: the identity is structural
    for name, spec in (("cantor_recursive", CANTOR),
                       ("recursive_non_cantor", NON_CANTOR)):
        hi, lo = SymbolicIFS(spec, n, E0), SymbolicIFS(spec, n - 1, E0)
        worst = 0.0
        for c in (0.0, (spec.b - 1) * spec.stride):
            lhs = hi.field(c + spec.rho * r)
            rhs = spec.alpha_field * lo.field(r)
            worst = max(worst, float((np.abs(lhs - rhs)
                                      / np.maximum(np.abs(rhs), 1.0)).max()))
        rows.append({"n": n, "family": name, "E_scale_rel": worst,
                     "new_parameters": 0, "zero_shot": True})

    # non-recursive controls: same affine maps, no identity expected
    for name, fam in (("periodic_procedural", "L3_periodic"),
                      ("shuffled_seeded", "L5_shuffled"),
                      ("center_anchored_seeded", "L6_center_anchored")):
        Ln = build_layout(fam, n, E0, seed=SEED)
        Lm = build_layout(fam, n - 1, E0, seed=SEED)
        lhs = Ln.field(r / 3.0)
        rhs = 1.5 * Lm.field(r)
        rows.append({"n": n, "family": name,
                     "E_scale_rel": float((np.abs(lhs - rhs)
                                           / np.maximum(np.abs(rhs), 1.0)).max()),
                     "new_parameters": 3 * N_GAPS(n), "zero_shot": False})

df = pd.DataFrame(rows)
out = pathlib.Path("results/v3_3/tables/bench_scale_transfer_all.csv")
df.to_csv(out, index=False)
summ = df.groupby("family").agg(max_E_scale_rel=("E_scale_rel", "max"),
                                median_E_scale_rel=("E_scale_rel", "median"),
                                new_params_at_max_n=("new_parameters", "max"),
                                zero_shot=("zero_shot", "all"))
print(summ.to_string())
exact = summ[summ.max_E_scale_rel < 1e-5].index.tolist()
print(f"\nfamilies with EXACT scale transfer: {sorted(exact)}")
pathlib.Path("results/v3_3/tables/scale_transfer_summary.json").write_text(
    json.dumps({"exact_scale_transfer": sorted(exact),
                "summary": summ.reset_index().to_dict("records")}, indent=2))
