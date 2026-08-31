"""V3.3.3 -- gap 2. Report EVERY quantile, for BOTH boundary definitions.

V3.3.2 leaned on q50 of U_EST_mid. Here q50/q75/q90/q95 are reported for both
U_EST_mid (projection midpoint) and U_EST_beh (behavioural boundary), and the
full feasible interval is given at each, with INFEASIBLE stated plainly.

rho_abs_star(n, delta) is "the largest contraction ratio meeting an absolute
guard requirement delta at depth n". It is NOT the LLM optimum, the safest
controller, or the behavioural optimum.
"""
import sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
from cantor_guard_v332.absolute_guard import (feasible_interval, rho_abs_star,
                                              G_n_max, rho_guard_max, RHO_CANTOR)

TAB = pathlib.Path("results/v3_3_3/tables")
V332 = json.loads(pathlib.Path("results/v3_3_2/tables/"
                               "phase_calibration_qwen2.5-0.5b-instruct.json").read_text())
BEH = json.loads((TAB / "behavioral_boundary.json").read_text())
SRC = {"U_EST_mid": V332["U_EST"]["delta_abs_quantiles"],
       "U_EST_beh": BEH["U_EST_beh_quantiles"]}
QS = ("q50", "q75", "q90", "q95")

rows = []
for name, q in SRC.items():
    if q is None:
        print(f"{name}: unavailable"); continue
    print(f"\n=== {name} ===   " + json.dumps({k: round(v, 5) for k, v in q.items()}))
    for n in (2, 3, 5):
        print(f"  n={n}  G_n,max={G_n_max(n):.6f} at rho={rho_guard_max(n):.4f}"
              f"   G_n(1/3)={3.0**-n:.6f}")
        for p in QS:
            d = q[p]
            iv = feasible_interval(n, d)
            if iv is None:
                print(f"    {p} delta={d:.5f} -> INFEASIBLE")
                rows.append({"uncertainty": name, "n": n, "quantile": p,
                             "delta": d, "feasible": False, "rho_left": None,
                             "rho_right": None, "cantor_feasible": False})
            else:
                cf = iv[0] <= RHO_CANTOR <= iv[1]
                print(f"    {p} delta={d:.5f} -> [{iv[0]:.4f}, {iv[1]:.4f}]"
                      f"  rho_abs*={iv[1]:.4f}  cantor_feasible={cf}")
                rows.append({"uncertainty": name, "n": n, "quantile": p,
                             "delta": d, "feasible": True, "rho_left": iv[0],
                             "rho_right": iv[1], "cantor_feasible": bool(cf)})
df = pd.DataFrame(rows); df.to_csv(TAB / "quantile_sensitivity.csv", index=False)

print("\n=== is 'rho ~ 0.46' robust to the quantile choice? (n=3) ===")
for name in SRC:
    sub = df[(df.uncertainty == name) & (df.n == 3) & df.feasible]
    if not len(sub):
        print(f"  {name}: infeasible at every quantile"); continue
    lo, hi = sub.rho_right.min(), sub.rho_right.max()
    print(f"  {name}: rho_abs* spans [{lo:.4f}, {hi:.4f}] across q50..q95"
          f"  (n feasible = {len(sub)}/4)")
    ok = sub.loc[sub.cantor_feasible, "quantile"].tolist()
    print(f"    Cantor 1/3 inside the feasible interval at: {ok}")
(TAB / "quantile_summary.json").write_text(df.to_json(orient="records", indent=2))
print(f"\nwrote {TAB}/quantile_sensitivity.csv")
