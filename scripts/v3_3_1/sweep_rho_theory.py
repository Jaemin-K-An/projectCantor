"""V3.3.1 PHASE 2/3/8 -- exact analytic metrics across the rho family."""
import sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
from cantor_guard_v331.guard_geometry import (
    guard_width, retention, hausdorff_dim, alpha_field, alpha_sensitivity,
    rho_star, kappa_of_rho, bottleneck, retained_measure, new_guard_measure,
    cumulative_coverage, new_coverage_argmax, RHO_CANTOR)

CFG = json.loads(pathlib.Path("configs/v3_3_1/rho_grid.json").read_text())
OUT = pathlib.Path("results/v3_3_1/tables"); OUT.mkdir(parents=True, exist_ok=True)

rows = []
for rho in CFG["theory_grid"]:
    row = {"rho": rho, "is_cantor": abs(rho - RHO_CANTOR) < 1e-12,
           "g": guard_width(rho), "retention": retention(rho),
           "hausdorff_dim": hausdorff_dim(rho),
           "alpha_field": alpha_field(rho),
           "alpha_sensitivity": alpha_sensitivity(rho),
           "kappa_implied": kappa_of_rho(rho)}
    for k in CFG["kappa_grid"]:
        row[f"B_kappa_{k}"] = bottleneck(rho, k)
    for n in CFG["depths"]:
        row[f"mu_K_{n}"] = retained_measure(rho, n)
        row[f"new_cov_{n}"] = new_guard_measure(rho, n)
        row[f"cum_cov_{n}"] = cumulative_coverage(rho, n)
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(OUT / "theory_rho_sweep.csv", index=False)

# dense verification that argmax B_kappa = 1/(2+kappa)
dense = np.linspace(1e-4, 0.5 - 1e-4, 200_001)
ver = []
for k in CFG["kappa_grid"]:
    b = np.minimum(dense, (1 - 2 * dense) / k)
    i = int(b.argmax())
    ver.append({"kappa": k, "argmax_numeric": float(dense[i]),
                "argmax_theory": rho_star(k),
                "abs_error": abs(float(dense[i]) - rho_star(k)),
                "max_value_numeric": float(b[i]),
                "max_value_theory": 1.0 / (2.0 + k)})
vdf = pd.DataFrame(ver); vdf.to_csv(OUT / "theory_argmax_verification.csv", index=False)

# new-coverage argmax, the honest counterexample
cov = [{"n": n, "argmax_theory": new_coverage_argmax(n),
        "argmax_numeric": float(dense[int(((1 - 2 * dense) * (2 * dense) ** n).argmax())]),
        "equals_one_third": abs(new_coverage_argmax(n) - RHO_CANTOR) < 1e-12}
       for n in range(1, 13)]
cdf = pd.DataFrame(cov); cdf.to_csv(OUT / "theory_new_coverage_argmax.csv", index=False)

print("=== argmax B_kappa (dense grid of 200k points) ===")
print(vdf.to_string(index=False))
print("\n=== new-coverage argmax = n/(2(n+1)) -- 1/3 ONLY at n=2 ===")
print(cdf.head(8).to_string(index=False))
print("\n=== Cantor vs rho=0.28, on the right quantities ===")
sub = df[df.rho.isin([0.28, RHO_CANTOR])][
    ["rho", "g", "retention", "alpha_field", "alpha_sensitivity", "B_kappa_1.0"]]
print(sub.to_string(index=False))
json.dump({"argmax_verification": ver, "new_coverage": cov},
          open(OUT / "theory_verification.json", "w"), indent=2)
print(f"\nwrote {OUT}/theory_rho_sweep.csv")
