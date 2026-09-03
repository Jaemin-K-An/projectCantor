"""V3.3.5a PHASE 8 -- where does the refusal coordinate have causal leverage?

Standardized so the three phases are comparable despite different residual
scales:

    beta_std   = b_raw * sigma_phase     log-odds per 1 sigma of projection
    dP_2sigma  = P(refusal | z+sigma) - P(refusal | z-sigma)

This is a mechanistic question, asked identically at every phase. It is not a
search for a phase where Cantor wins -- the controller is not involved at all.
"""
import sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
from cantor_guard_v333.behavioral_boundary import fit_logistic
TAB = pathlib.Path("results/v3_3_5a/tables")


def leverage(z, y, name, zcol_clean=None):
    a, b = fit_logistic(z, y)
    sig = float(np.std(zcol_clean, ddof=1)) if zcol_clean is not None else float(np.std(z, ddof=1))
    zc = float(np.mean(zcol_clean)) if zcol_clean is not None else float(np.mean(z))
    sg = lambda x: 1 / (1 + np.exp(-(a + b * x)))
    return {"phase": name, "b_raw": float(b), "sigma_phase": sig,
            "beta_std": float(b * sig), "dP_2sigma": float(sg(zc + sig) - sg(zc - sig)),
            "n_obs": int(len(z))}

rows = []
# P0 (this version)
p = pd.read_csv("results/v3_3_5a/raw/p0_dose_D_beh_P0_confirm.csv")
rows.append(leverage(p.z_p0.to_numpy(), p.refusal.to_numpy(float), "P0 (pre-token-1)",
                     p[p.dose == 0.0].z_p0.to_numpy()))
# G1 (V3.3.5)
g = pd.read_csv("results/v3_3_5/raw/g1_dose_D_beh_g1_confirm.csv")
rows.append(leverage(g.z_g1.to_numpy(), g.refusal.to_numpy(float), "G1 (first decode)",
                     g[g.dose == 0.0].z_g1.to_numpy()))
# global / all-forward (V3.3.3)
b3 = pd.read_csv("results/v3_3_3/raw/behavioral_dose_response.csv")
rows.append(leverage(b3.z.to_numpy(), b3.refusal.to_numpy(float), "GLOBAL (all forwards)",
                     b3[b3.dose_sigma == 0.0].z.to_numpy()))

df = pd.DataFrame(rows)
df["leverage_vs_P0"] = df.beta_std / df.beta_std.iloc[0]
df.to_csv(TAB / "phase_causality.csv", index=False)
print("=== standardized causal leverage of the refusal coordinate ===")
print(df[["phase", "b_raw", "sigma_phase", "beta_std", "dP_2sigma",
          "leverage_vs_P0"]].round(5).to_string(index=False))
print()
best = df.loc[df.beta_std.abs().idxmax()]
print(f"strongest leverage: {best.phase}  beta_std={best.beta_std:.4f}"
      f"  dP over 2 sigma = {best.dP_2sigma:+.4f}")
print(f"P0 vs G1 ratio: {df.beta_std.iloc[0]/df.beta_std.iloc[1]:.3f}"
      f"   (>1 would mean P0 has more leverage)")
interp = ("GLOBAL >> single-state: refusal control is temporally distributed, "
          "not localised at any one residual state"
          if df.beta_std.iloc[2] > 3 * max(df.beta_std.iloc[0], df.beta_std.iloc[1])
          else "no phase dominates")
print(f"\ninterpretation: {interp}")
json.dump({"rows": rows, "interpretation": interp,
           "p0_over_g1": float(df.beta_std.iloc[0]/df.beta_std.iloc[1])},
          open(TAB / "phase_causality.json", "w"), indent=2)
