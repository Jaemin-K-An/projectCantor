"""V3.3.4 PHASE 17 -- behavioural-centre ablation, the direct fix to V3.3.3.

V3.3.3 measured tau_beh and then ran its controller on tau_mid. Here the SAME
rho family is evaluated under both centres so the consequence of that error is
visible.

The two centres put the real states in completely different regimes:
    tau_mid = +0.9887  ->  states land near the middle of the coordinate
    tau_beh = -2.6263  ->  states land in the saturated tail
and the logistic slope differs by an order of magnitude between them, which
governs how much a residual perturbation moves the policy at all.
"""
import argparse, sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
from cantor_guard.models import load_model
from cantor_guard.io import seed_everything
from cantor_guard_v332.phase_residuals import collect_phase_residuals
from cantor_guard_v334.guarded_policy import CantorGuardedPolicy
from cantor_guard_v334.certified_geometry import rho_max, RHO_CANTOR

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
ap.add_argument("--batch", type=int, default=8)
a = ap.parse_args()
TAB = pathlib.Path("results/v3_3_4/tables")
B = json.loads(pathlib.Path("results/v3_3_3/tables/behavioral_boundary.json").read_text())
V = json.loads(pathlib.Path("results/v3_3_2/tables/"
                            "phase_calibration_qwen2.5-0.5b-instruct.json").read_text())
TAU_BEH, TAU_MID = B["tau_beh"], V["calibrations"]["G1"]["tau"]
SIG, GAM, LAYER = B["sigma_G1"], V["gamma"], V["layer"]
seed_everything(20260904)

pr = pd.read_csv("results/v3_3_3/cache/d_beh_prompts.csv")
b = load_model(a.model)
v = np.load(f"results/v3_3_2/cache/{a.model}_v332_phase.npz")["v"]
z0 = collect_phase_residuals(b, list(pr.prompt), LAYER, batch_size=a.batch)["decode1"] @ v
EPS = np.linspace(0.0, 6.0, 121)
RHOS = [0.25, 0.28, 0.30, 1/3, 0.36, 0.40, 0.44]
rows = []
for cname, tau in (("tau_mid", TAU_MID), ("tau_beh", TAU_BEH)):
    r = 1.0 / (1.0 + np.exp(np.clip(GAM * (z0 - tau) / SIG, -60, 60)))
    slope = (GAM / SIG) * r * (1 - r)
    print(f"{cname:8s} tau={tau:+.4f}  r median={np.median(r):.4f}  "
          f"|dr/dz| median={np.median(slope):.5f}  "
          f"(Lipschitz max {GAM/(4*SIG):.5f})")
    for rho in RHOS:
        C = CantorGuardedPolicy(rho, 3, tau_beh=tau, sigma=SIG, gamma=GAM)
        r0 = C.coordinate(z0); k0, i0 = C.classify_r(r0)
        it = (k0 == "leaf")
        first = np.full(int(it.sum()), np.nan); zi = z0[it]
        for e in EPS[1:]:
            hit = np.zeros(len(zi), bool)
            for s in (-1.0, 1.0):
                k1, i1 = C.classify_r(C.coordinate(zi + s * e))
                hit |= (k1 == "leaf") & (i1 != i0[it])
            new = hit & np.isnan(first); first[new] = e
        rows.append({"centre": cname, "rho": rho,
                     "is_cantor": abs(rho - RHO_CANTOR) < 1e-9,
                     "frac_in_leaf": float(it.mean()),
                     "median_r": float(np.median(r)),
                     "median_slope": float(np.median(slope)),
                     "median_switch_eps": float(np.nanmedian(first)),
                     "frac_switched": float(np.mean(~np.isnan(first))),
                     "eps_cert": C.certificate_z_lipschitz(),
                     "action_spread": float(C.action(r0).max() - C.action(r0).min())})
df = pd.DataFrame(rows)
df.to_csv("results/v3_3_4/raw/center_ablation.csv", index=False)
print("\n=== behavioural centre ablation (depth 3) ===")
print(df[["centre", "rho", "is_cantor", "frac_in_leaf", "median_switch_eps",
          "action_spread"]].round(4).to_string(index=False))
for c, g in df.groupby("centre"):
    peak = float(g.rho[g.median_switch_eps.idxmax()])
    print(f"  {c}: empirical robustness peak at rho={peak:.4f} "
          f"(theory 1/3={1/3:.4f})   action spread median={g.action_spread.median():.4f}")
json.dump(df.to_dict("records"), open(TAB/"center_ablation.json", "w"), indent=2)
