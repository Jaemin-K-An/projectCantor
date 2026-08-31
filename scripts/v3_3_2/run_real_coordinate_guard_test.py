"""V3.3.2 PHASE 11/14 -- System B guard geometry on REAL residual coordinates.

The synthetic sweep put the guard controller on a UNIFORM state distribution.
This puts it on the distribution the LLM actually produces: the threat
coordinate r measured at PHASE G1 on the untouched D_final, with the corrected
two-class calibration.

Perturbations use the MEASURED estimator uncertainty delta_abs -- an absolute
width in r-units, identical for every rho. Nothing here is rho-normalised.

This is SYSTEM B (discrete guard/refine geometry). It does NOT license any
conclusion about SYSTEM A (the smooth barrier's effect on generated text);
that is measured separately and reported separately.
"""
import argparse, sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd, torch
from cantor_guard.models import load_model
from cantor_guard.io import seed_everything
from cantor_guard_v332.phase_residuals import collect_phase_residuals
from cantor_guard_v332.calibration import threat_coordinate
from cantor_guard_v331.hierarchical_guard import GuardController, GUARD, LEAF
from cantor_guard_v332.absolute_guard import (rho_abs_star, G_n, RHO_CANTOR,
                                              rho_guard_max)
from cantor_guard_v32.cluster_stats import cluster_bootstrap_by_goal

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--n-pert", type=int, default=64)
ap.add_argument("--seed", type=int, default=20260901)
a = ap.parse_args()
seed_everything(a.seed)

TAB = pathlib.Path("results/v3_3_2/tables")
CAL = json.loads((TAB / f"phase_calibration_{a.model}.json").read_text())
FS = json.loads(pathlib.Path("configs/v3_3_2/final_split.json").read_text())
RHOS = [0.20, 0.24, 0.28, RHO_CANTOR, 0.36, 0.40, 0.44]
DEPTHS = [2, 3, 5]
DELTA = CAL["U_EST"]["delta_abs_quantiles"]["q50"]
GAMMA = CAL["gamma"]
LAYER = CAL["layer"]
print(f"delta_abs (q50, measured, rho-independent) = {DELTA:.5f}")

pr = pd.read_csv("results/v3_3_2/cache/d_final_prompts.csv")
prompts, pids = list(pr.prompt), list(pr.pid)
b = load_model(a.model)
z = np.load(f"results/v3_3_2/cache/{a.model}_v332_phase.npz")
v = z["v"]
print(f"collecting D_final G1 residuals ({len(prompts)} untouched goals)", flush=True)
R = collect_phase_residuals(b, prompts, LAYER, batch_size=a.batch)
cg = CAL["calibrations"]["G1"]
r_true = threat_coordinate(R["decode1"] @ v, cg["tau"], cg["sigma"], GAMMA)
print(f"  r distribution: mean={r_true.mean():.4f} sd={r_true.std():.4f} "
      f"[{r_true.min():.4f}, {r_true.max():.4f}]")

rng = np.random.default_rng(a.seed)
# worst-case magnitude, random sign: the guard bound is a worst-case statement
signs = rng.choice([-1.0, 1.0], size=(len(r_true), a.n_pert))
rows = []
for n in DEPTHS:
    for rho in RHOS:
        C = GuardController(rho, n)
        k_t, _, a_t = C.classify(r_true)
        reg = np.zeros((len(r_true), a.n_pert), bool)
        gact = np.zeros_like(reg)
        for j in range(a.n_pert):
            r_obs = np.clip(r_true + DELTA * signs[:, j], 0.0, 1.0)
            k_o, _, a_o = C.classify(r_obs)
            reg[:, j] = (k_t == LEAF) & (k_o == LEAF) & (a_t != a_o)
            gact[:, j] = (k_o == GUARD)
        for i, pid in enumerate(pids):
            rows.append({"pid": pid, "rho": rho, "depth": n,
                         "is_cantor": abs(rho - RHO_CANTOR) < 1e-9,
                         "regression": float(reg[i].mean()),
                         "abstention": float(gact[i].mean()),
                         "guard_width": float(G_n(rho, n)),
                         "guard_covers_delta": bool(G_n(rho, n) >= DELTA)})
df = pd.DataFrame(rows)
df.to_csv("results/v3_3_2/raw/systemB_real_coordinate.csv", index=False)

print("\n=== System B on REAL D_final coordinates (per-goal means) ===")
summ = df.groupby(["depth", "rho"]).agg(
    regression=("regression", "mean"), abstention=("abstention", "mean"),
    guard=("guard_width", "first"), covers=("guard_covers_delta", "first"))
print(summ.round(5).to_string())

# goal-clustered comparisons against Cantor, per depth
print("\n=== goal-clustered CIs vs Cantor (regression) ===")
comp = []
for n in DEPTHS:
    g = df[df.depth == n]
    piv = g.pivot_table(index="pid", columns="rho", values="regression")
    for rho in RHOS:
        if abs(rho - RHO_CANTOR) < 1e-9:
            continue
        m = piv[[RHO_CANTOR, rho]].dropna().reset_index().rename(
            columns={RHO_CANTOR: "score_a", rho: "score_b"})
        st = cluster_bootstrap_by_goal(m, "score_a", "score_b", n_boot=20000, seed=7)
        sig = st["ci_lo"] > 0 or st["ci_hi"] < 0
        comp.append({"depth": n, "rho": rho, "mean_diff": st["mean_diff"],
                     "ci_lo": st["ci_lo"], "ci_hi": st["ci_hi"], "significant": sig})
        print(f"  n={n} cantor-vs-{rho:.4f}: d={st['mean_diff']:+.5f} "
              f"[{st['ci_lo']:+.5f},{st['ci_hi']:+.5f}] sig={sig}")
pd.DataFrame(comp).to_csv(TAB / "systemB_cluster_comparisons.csv", index=False)

emp = {}
for n in DEPTHS:
    s = summ.loc[n]
    zero = s[s.regression <= 1e-12]
    emp[n] = {"argmin_regression_rho": float(s.regression.idxmin()),
              "max_rho_with_zero_regression": (float(zero.index.max())
                                               if len(zero) else None),
              "regression_spread": float(s.regression.max() - s.regression.min()),
              "abstention_spread": float(s.abstention.max() - s.abstention.min()),
              "theory_rho_abs_star": rho_abs_star(n, DELTA)}
    print(f"\n  n={n}: largest rho with ZERO regression = "
          f"{emp[n]['max_rho_with_zero_regression']}   "
          f"theory rho_abs* = {emp[n]['theory_rho_abs_star']}")
json.dump({"delta_abs_used": DELTA, "per_depth": emp,
           "any_significant": bool(pd.DataFrame(comp).significant.any())},
          open(TAB / "systemB_summary.json", "w"), indent=2)
print(f"\nwrote results/v3_3_2/raw/systemB_real_coordinate.csv")
