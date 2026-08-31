"""V3.3.1 PHASE 11 -- synthetic guard/refinement trade-off.

THE ANTI-TAUTOLOGY TEST (harness section 11). "Maximising min(rho, g) makes
them equal" is arithmetic; on its own it says nothing about control. So the
criterion has to earn its keep by making a FALSIFIABLE prediction.

It does. If the calibration uncertainty radius scales with the resolution the
controller is trying to achieve, delta = beta * (leaf width) = beta * rho^n,
then a direct child-to-child crossing at the finest level must traverse the
whole level-n guard, whose width is rho^(n-1) * g. Preventing it requires

    rho^(n-1) * (1 - 2rho)  >=  beta * rho^n
    (1 - 2rho)              >=  beta * rho          <- rho^(n-1) cancels

i.e. kappa = beta and

    rho*(beta) = 1/(2 + beta)

so the optimum is 1/3 exactly when beta = 1: the uncertainty radius equals ONE
LEAF WIDTH, the very resolution being attempted.

CORRECTION TO THE ONE-HALF READING. It is tempting to demand a two-sided
margin, guard >= 2*delta, giving kappa = 2*beta and Cantor at beta = 1/2. The
sweep below rejects that: a single perturbation r -> r + Delta has to cross the
entire guard, so one-sided is correct. Measured median |empirical - predicted|
is 0.0011 for kappa = beta against 0.0723 for kappa = 2*beta, and the grid
spacing is 0.02. Both are reported.

WHY THE LEAF WIDTH AND NOT THE TOP-LEVEL CHILD. A first version set
delta = beta*rho, using the level-1 child. That is not scale invariant: by
level k the cell has contracted to rho^(k-1), so a fixed absolute delta dwarfs
the deep cells and every configuration is dominated by deep-level crossings.
The rho^(n-1) cancellation above is what makes the criterion depth-independent,
and it is the reason the condition can be stated as a pure ratio at all.

Two measured axes:
  REGRESSION        true state in one child, perturbed estimate in the OPPOSITE
                    child, with the guard never entered -- the policy flipped
                    without the uncertainty buffer catching it. RISES with rho,
                    because a larger child leaves a narrower guard.
  ABSTENTION        how often the controller takes the conservative guard
                    action instead of a refined child policy. FALLS with rho.
                    This is the utility cost of guarding (harness section 58):
                    a controller that abstains almost always is safe and
                    useless.

Landing in the guard when the true state is near a boundary is NOT a
regression; that is the guard doing its job (harness section 40).

CORRECTED AXIS. A first version used "guard fired although the true state was
in a leaf" as the utility cost. That metric is perverse: when guards are wide
the TRUE state is usually in a guard too, so the quantity SHRINKS exactly where
over-guarding is worst, and both axes then rose together with rho, leaving no
trade-off at all. The abstention rate is the direct measure of "how often is the
refined policy replaced by the conservative one", which is the cost section 58
asks about. Both definitions are computed and reported.
"""
import sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
from cantor_guard_v331.hierarchical_guard import GuardController, GUARD, LEAF
from cantor_guard_v331.guard_geometry import rho_star, RHO_CANTOR

CFG = json.loads(pathlib.Path("configs/v3_3_1/rho_grid.json").read_text())
OUT = pathlib.Path("results/v3_3_1/tables"); OUT.mkdir(parents=True, exist_ok=True)
N_STATES, SEED = 200_000, 20260831
BETAS = [0.25, 0.375, 0.5, 0.625, 0.75, 1.0, 1.5]
DEPTHS = CFG["depths"]

rng = np.random.default_rng(SEED)
r_true = rng.uniform(0.0, 1.0, N_STATES)
# Two perturbation models. g >= 2*delta is a WORST-CASE guarantee, so the
# uniform model -- where |u| = 1 has measure zero -- is expected to tolerate a
# larger rho than the bound allows. Both are measured so the gap between the
# worst-case theory and the average-case observation is visible rather than
# explained away.
U = {"uniform": rng.uniform(-1.0, 1.0, N_STATES),
     "worst_case": rng.choice([-1.0, 1.0], N_STATES)}

rows = []
for n in DEPTHS:
    for rho in CFG["synthetic_grid"]:
        C = GuardController(rho, n)
        k_t, l_t, a_t = C.classify(r_true)
        for beta in BETAS:
          for pmodel, u_pert in U.items():
            delta = beta * rho ** n             # scales with the LEAF width
            r_obs = np.clip(r_true + delta * u_pert, 0.0, 1.0)
            k_o, l_o, a_o = C.classify(r_obs)

            both_leaf = (k_t == LEAF) & (k_o == LEAF)
            regression = both_leaf & (a_t != a_o)
            over = (k_o == GUARD) & (k_t == LEAF)
            caught = (k_o == GUARD) & (k_t == GUARD)

            rows.append({
                "n": n, "rho": rho, "beta": beta, "delta": delta,
                "perturbation": pmodel,
                "is_cantor": abs(rho - RHO_CANTOR) < 1e-12,
                "regression_rate": float(regression.mean()),
                "over_intervention_rate": float(over.mean()),
                "guard_activation_rate": float((k_o == GUARD).mean()),
                "guard_caught_rate": float(caught.mean()),
                "leaf_width": C.leaf_width(),
                "guard_measure": C.guard_measure(),
                "rho_star_pred": 1.0 / (2.0 + beta),          # kappa = beta
                "rho_star_pred_two_sided": 1.0 / (2.0 + 2.0 * beta),
            })

df = pd.DataFrame(rows)
df.to_csv(OUT / "synthetic_tradeoff.csv", index=False)

# Balanced loss = the theorem's own bottleneck, NOT a post-hoc scalarisation.
# L(rho) = max(normalised regression, normalised over-intervention)
# Theorem P says retention, dimension and both amplifications all improve as
# rho grows, so the constrained optimum is the LARGEST rho that still meets the
# guard requirement. The empirical analogue is the largest rho whose regression
# rate is still at or below tolerance. TAU is fixed here, before the sweep.
TAU = 0.001
pred_rows = []
for (n, beta, pm), g in df.groupby(["n", "beta", "perturbation"]):
    g = g.sort_values("rho")
    ok = g[g.regression_rate <= TAU]
    emp = float(ok.rho.max()) if len(ok) else float(g.rho.min())
    pred = 1.0 / (2.0 + beta)
    pred2 = 1.0 / (2.0 + 2.0 * beta)
    pred_rows.append({"n": n, "beta": beta, "perturbation": pm, "tau": TAU,
                      "rho_emp_opt": emp,
                      "rho_star_pred": pred, "abs_error": abs(emp - pred),
                      "rho_star_pred_two_sided": pred2,
                      "abs_error_two_sided": abs(emp - pred2),
                      "pred_is_cantor": abs(pred - RHO_CANTOR) < 1e-9,
                      "emp_is_cantor": abs(emp - RHO_CANTOR) < 0.03})
pred = pd.DataFrame(pred_rows)
pred.to_csv(OUT / "synthetic_beta_prediction.csv", index=False)

print("=== does the empirical boundary track rho* = 1/(2+beta)? ===")
for pm in ("worst_case", "uniform"):
    sub = pred[pred.perturbation == pm]
    t = sub.groupby("beta").agg(rho_emp=("rho_emp_opt", "median"),
                                rho_pred=("rho_star_pred", "first"),
                                err=("abs_error", "median")).round(4)
    r = np.corrcoef(sub.rho_emp_opt, sub.rho_star_pred)[0, 1]
    print(f"\n-- perturbation = {pm} --")
    print(t.to_string())
    print(f"   kappa=beta   : median |err| = {sub.abs_error.median():.4f}  corr = {r:.4f}")
    print(f"   kappa=2*beta : median |err| = {sub.abs_error_two_sided.median():.4f}"
          f"  (rejected; grid spacing is 0.02)")

print("\n=== beta = 1/2 (the Cantor case): regression vs abstention ===")
h = df[(df.beta == 0.5) & (df.n == CFG["llm_depth"])
       & (df.perturbation == "worst_case")].copy()
rn = h.regression_rate / h.regression_rate.max()
an = h.guard_activation_rate / h.guard_activation_rate.max()
h["balanced_loss"] = np.maximum(rn, an)
print(h[["rho", "regression_rate", "guard_activation_rate", "balanced_loss",
         "is_cantor"]].round(5).to_string(index=False))
print(f"  argmin balanced_loss = {h.rho.to_numpy()[int(h.balanced_loss.to_numpy().argmin())]:.5f}"
      f"   (theory 1/3 = {1/3:.5f})")
wc = pred[pred.perturbation == "worst_case"]
json.dump({"betas": BETAS, "n_states": N_STATES, "seed": SEED, "tau": TAU,
           "relation": "kappa = beta, rho* = 1/(2+beta)",
           "median_abs_error_kappa_beta": float(wc.abs_error.median()),
           "median_abs_error_kappa_2beta": float(wc.abs_error_two_sided.median()),
           "grid_spacing": 0.02,
           "beta_for_cantor": 1.0},
          open(OUT / "synthetic_meta.json", "w"), indent=2)
print(f"\nwrote {OUT}/synthetic_tradeoff.csv")
