"""V3.3.2 PHASE 17 -- System B Pareto on real coordinates.

Zero regression is trivially achievable by abstaining always, so regression
alone cannot rank rho. The pair (regression, abstention) can: a controller that
never refines is safe and useless, exactly the trade-off the guard theory is
about.
"""
import sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
from cantor_guard_v332.absolute_guard import rho_abs_star, G_n, RHO_CANTOR
from cantor_guard_v331.guard_geometry import RHO_CANTOR as RC

TAB = pathlib.Path("results/v3_3_2/tables")
CAL = json.loads((TAB / "phase_calibration_qwen2.5-0.5b-instruct.json").read_text())
DELTA = CAL["U_EST"]["delta_abs_quantiles"]["q50"]
df = pd.read_csv("results/v3_3_2/raw/systemB_real_coordinate.csv")
s = df.groupby(["depth", "rho"]).agg(regression=("regression", "mean"),
                                     abstention=("abstention", "mean")).reset_index()

print(f"delta_abs = {DELTA:.5f}\n")
out = {}
for n, g in s.groupby("depth"):
    g = g.sort_values("rho")
    # Pareto: minimise both. A point is dominated if another is <= on both and
    # < on one.
    front = []
    for _, r in g.iterrows():
        dom = ((g.regression <= r.regression) & (g.abstention <= r.abstention) &
               ((g.regression < r.regression) | (g.abstention < r.abstention)))
        if not dom.any():
            front.append(float(r.rho))
    zero = g[g.regression <= 1e-12]
    best_zero = float(zero.rho.max()) if len(zero) else None
    pred = rho_abs_star(n, DELTA)
    out[int(n)] = {"pareto_front_rhos": front,
                   "cantor_on_front": bool(any(abs(x - RC) < 1e-9 for x in front)),
                   "max_rho_zero_regression": best_zero,
                   "abstention_at_that_rho": (float(zero[zero.rho == best_zero]
                                                    .abstention.iloc[0])
                                              if best_zero else None),
                   "abstention_at_cantor": float(
                       g[np.isclose(g.rho, RC)].abstention.iloc[0]),
                   "theory_rho_abs_star": pred}
    print(f"=== depth {n} ===   theory rho_abs* = "
          f"{'INFEASIBLE' if pred is None else f'{pred:.5f}'}")
    print(g.assign(on_front=g.rho.isin(front)).round(5).to_string(index=False))
    print(f"  Pareto front: {[round(x,4) for x in front]}   "
          f"Cantor on front: {out[int(n)]['cantor_on_front']}")
    if best_zero is not None:
        print(f"  largest rho with zero regression = {best_zero:.4f} "
              f"(abstention {out[int(n)]['abstention_at_that_rho']:.4f}); "
              f"Cantor abstention {out[int(n)]['abstention_at_cantor']:.4f}")

# Does the theory's prediction bracket the empirical zero-regression boundary?
print("\n=== does theory predict the empirical boundary? ===")
brack = {}
for n, g in s.groupby("depth"):
    g = g.sort_values("rho")
    zero = g[g.regression <= 1e-12]
    pred = rho_abs_star(n, DELTA)
    if not len(zero) or pred is None:
        brack[int(n)] = {"bracketed": None, "note": "no zero-regression rho or infeasible"}
        print(f"  n={n}: theory INFEASIBLE; empirical zero-regression up to "
              f"{zero.rho.max() if len(zero) else None} (degenerate, abstention "
              f"{g[g.rho==zero.rho.max()].abstention.iloc[0]:.3f})" if len(zero) else
              f"  n={n}: no zero-regression rho")
        continue
    lo = float(zero.rho.max())
    above = g[g.rho > lo]
    hi = float(above.rho.min()) if len(above) else 0.5
    ok = lo <= pred <= hi
    brack[int(n)] = {"empirical_bracket": [lo, hi], "theory": pred, "bracketed": ok}
    print(f"  n={n}: empirical boundary in ({lo:.4f}, {hi:.4f}];  "
          f"theory {pred:.5f};  bracketed = {ok}")

(TAB / "systemB_pareto.json").write_text(json.dumps(
    {"delta_abs": DELTA, "per_depth": out, "bracketing": brack}, indent=2))
print(f"\nwrote {TAB}/systemB_pareto.json")
