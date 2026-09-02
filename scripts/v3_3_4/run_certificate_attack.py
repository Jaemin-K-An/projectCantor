"""V3.3.4 PHASE 15/16 -- certificate validation and the depth-shift test.

Two claims are tested on REAL G1 residual states:

  CERTIFICATE. For a pure directional attack h' = h - eps*v, |Delta z| = eps
  exactly. The theorem says a DIRECT terminal-leaf switch is impossible while
  eps < eps_cert(rho). Any violation is a CERTIFICATE_IMPLEMENTATION_FAILURE
  and blocks the positive claim.

  DEPTH SHIFT. rho_theory(n) = (n-1)/(2n) predicts the empirical
  policy-switch threshold should peak near 1/4 at n=2, 1/3 at n=3 and 2/5 at
  n=5. A single peak at 1/3 could be coincidence; the whole law tracking is not.

No generation is needed: the policy-switch event is a coordinate-level
property, measured directly on the residual states.
"""
import argparse, sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd, torch
from cantor_guard.models import load_model
from cantor_guard.io import seed_everything
from cantor_guard_v332.phase_residuals import collect_phase_residuals
from cantor_guard_v334.guarded_policy import CantorGuardedPolicy
from cantor_guard_v334.certified_geometry import rho_max, M_n, RHO_CANTOR

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--prompts", default="results/v3_3_3/cache/d_beh_prompts.csv")
ap.add_argument("--tag", default="dev")
a = ap.parse_args()

TAB = pathlib.Path("results/v3_3_4/tables"); TAB.mkdir(parents=True, exist_ok=True)
RAW = pathlib.Path("results/v3_3_4/raw"); RAW.mkdir(parents=True, exist_ok=True)
B = json.loads(pathlib.Path("results/v3_3_3/tables/behavioral_boundary.json").read_text())
V332 = json.loads(pathlib.Path("results/v3_3_2/tables/"
                               "phase_calibration_qwen2.5-0.5b-instruct.json").read_text())
TAU, SIG, GAM, LAYER = B["tau_beh"], B["sigma_G1"], V332["gamma"], V332["layer"]
seed_everything(20260904)
RHOS = [0.20, 0.22, 0.24, 0.25, 0.26, 0.28, 0.30, 1/3, 0.36, 0.38, 0.40, 0.42, 0.44]
DEPTHS = [2, 3, 5]
LAMBDAS = [0.0, 0.25, 0.50, 0.75, 0.90, 0.99, 1.01, 1.10, 1.25, 1.50, 2.00]
# COMMON ABSOLUTE grid, identical for every rho and depth. The certificate-
# normalised grid CANNOT be used for the depth-shift test: if no switch occurs
# anywhere in range then the "empirical threshold" is just lam_max*eps_cert(rho),
# whose argmax is rho_theory BY CONSTRUCTION. Measuring the threshold on a
# common absolute scale makes the test non-circular.
EPS_COMMON = np.linspace(0.0, 6.0, 121)

pr = pd.read_csv(a.prompts)
b = load_model(a.model)
v = np.load(f"results/v3_3_2/cache/{a.model}_v332_phase.npz")["v"]
R = collect_phase_residuals(b, list(pr.prompt), LAYER, batch_size=a.batch)
z0 = R["decode1"] @ v
print(f"{a.tag}: {len(z0)} G1 states  z mean={z0.mean():+.4f} sd={z0.std():.4f}")
print(f"tau_beh={TAU:+.4f} sigma={SIG:.4f} gamma={GAM}")

rows = []
for n in DEPTHS:
    for rho in RHOS:
        C = CantorGuardedPolicy(rho, n, tau_beh=TAU, sigma=SIG, gamma=GAM)
        eps_c = C.certificate_z_lipschitz()
        r0 = C.coordinate(z0); k0, i0 = C.classify_r(r0)
        interior = (k0 == "leaf")
        for lam in LAMBDAS:
            eps = lam * eps_c
            for sgn in (-1.0, +1.0):       # both directions, worst case kept
                z1 = z0 + sgn * eps
                r1 = C.coordinate(z1); k1, i1 = C.classify_r(r1)
                switch = interior & (k1 == "leaf") & (i1 != i0)
                capture = interior & (k1 == "guard")
                rows.append({"n": n, "rho": rho, "lam": lam, "eps": eps,
                             "sign": sgn, "eps_cert": eps_c,
                             "is_cantor": abs(rho - RHO_CANTOR) < 1e-9,
                             "n_interior": int(interior.sum()),
                             "switch_rate": float(switch.sum() / max(interior.sum(), 1)),
                             "capture_rate": float(capture.sum() / max(interior.sum(), 1)),
                             "n_switch": int(switch.sum())})
df = pd.DataFrame(rows)
df.to_csv(RAW / f"certificate_attack_{a.tag}.csv", index=False)

below = df[df.lam < 1.0]
viol = below[below.n_switch > 0]
print(f"\n=== CERTIFICATE VALIDATION ===")
print(f"  configs with eps < eps_cert: {len(below)}")
print(f"  VIOLATIONS (a direct switch below the certificate): {len(viol)}")
if len(viol):
    print(viol[["n", "rho", "lam", "n_switch"]].head(10).to_string(index=False))
above = df[df.lam > 1.0]
print(f"  switch rate just below cert (lam=0.99): "
      f"{below[below.lam==0.99].switch_rate.mean():.5f}")
print(f"  switch rate just above cert (lam=1.01): "
      f"{df[df.lam==1.01].switch_rate.mean():.5f}")
# how conservative is the Lipschitz bound on THESE states?
r_here = 1.0 / (1.0 + np.exp(np.clip(GAM * (z0 - TAU) / SIG, -60, 60)))
local = (GAM / SIG) * r_here * (1 - r_here)
print(f"  local |dr/dz| median = {np.median(local):.5f} vs Lipschitz max "
      f"{GAM/(4*SIG):.5f}  -> bound is {GAM/(4*SIG)/np.median(local):.1f}x "
      f"conservative on these states")

# ---- depth shift on the COMMON ABSOLUTE grid (non-circular) ----
print(f"\n=== DEPTH-SHIFT TEST (common absolute epsilon grid) ===")
crows = []
for n in DEPTHS:
    for rho in RHOS:
        C = CantorGuardedPolicy(rho, n, tau_beh=TAU, sigma=SIG, gamma=GAM)
        r0 = C.coordinate(z0); k0, i0 = C.classify_r(r0)
        interior = (k0 == "leaf")
        first = np.full(interior.sum(), np.nan)
        zi = z0[interior]
        for e in EPS_COMMON:
            if e == 0:
                continue
            hit = np.zeros(len(zi), bool)
            for sgn in (-1.0, +1.0):
                r1 = C.coordinate(zi + sgn * e)
                k1, i1 = C.classify_r(r1)
                hit |= (k1 == "leaf") & (i1 != i0[interior])
            new = hit & np.isnan(first)
            first[new] = e
        crows.append({"n": n, "rho": rho,
                      "median_switch_eps": float(np.nanmedian(first)),
                      "frac_switched": float(np.mean(~np.isnan(first))),
                      "eps_cert": C.certificate_z_lipschitz()})
cdf = pd.DataFrame(crows)
cdf.to_csv(RAW / f"depthshift_common_{a.tag}.csv", index=False)
ds = []
for n in DEPTHS:
    p = cdf[cdf.n == n].reset_index(drop=True)
    emp = float(p.rho[p.median_switch_eps.idxmax()])
    th = rho_max(n)
    ds.append({"n": n, "rho_empirical_peak": emp, "rho_theory": th,
               "abs_error": abs(emp - th),
               "peak_median_switch_eps": float(p.median_switch_eps.max()),
               "frac_switched_at_peak": float(
                   p.frac_switched[p.median_switch_eps.idxmax()])})
    print(f"  n={n}: empirical peak rho={emp:.4f}   theory (n-1)/(2n)={th:.4f}"
          f"   |err|={abs(emp-th):.4f}   (median switch eps "
          f"{p.median_switch_eps.max():.3f}, {100*p.frac_switched[p.median_switch_eps.idxmax()]:.0f}% switched)")
dd = pd.DataFrame(ds); dd.to_csv(TAB / f"depthshift_{a.tag}.csv", index=False)
corr = float(np.corrcoef(dd.rho_empirical_peak, dd.rho_theory)[0, 1]) if len(dd) > 1 else np.nan
print(f"  corr(empirical, theory) = {corr:.4f}   median |err| = {dd.abs_error.median():.4f}")

json.dump({"violations": int(len(viol)), "n_below_cert_configs": int(len(below)),
           "depthshift": ds, "corr": corr,
           "median_abs_error": float(dd.abs_error.median())},
          open(TAB / f"certificate_summary_{a.tag}.json", "w"), indent=2)
print(f"\nwrote {RAW}/certificate_attack_{a.tag}.csv")
