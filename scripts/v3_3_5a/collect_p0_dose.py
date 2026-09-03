"""V3.3.5a PHASE 5/7/8 -- P0 dose-response and the P0 vs G1 phase comparison.

The narrow +-2 sigma gate of section 7 produced ZERO behavioural variation, so
it is reported as failed. This script measures P0's causal leverage on the SAME
amplitude scale already used for G1 in V3.3.5, which sections 13 and 37 require
in order to compare phases at all. It is the identical protocol at a larger
amplitude on the same direction, token and layer -- not a search over
placements for one that works.
"""
import argparse, sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
from cantor_guard.models import load_model
from cantor_guard.io import seed_everything
from cantor_guard_v32.metrics32 import is_refusal32, coherence32, safe_score32
from cantor_guard_v335a.p0_residual import generate_p0_only
from cantor_guard_v333.behavioral_boundary import (fit_logistic, isotonic_crossing,
                                                   tau_beh_bootstrap)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
ap.add_argument("--batch", type=int, default=10)
ap.add_argument("--split", default="D_beh_P0_dev")
ap.add_argument("--doses", default="-100,-80,-60,-45,-30,-20,-14,-8,-4,0,4")
ap.add_argument("--n-boot", type=int, default=20000)
a = ap.parse_args()
seed_everything(20260907)
TAB = pathlib.Path("results/v3_3_5a/tables")
D = json.loads((TAB/"p0_direction.json").read_text())
LAYER, SIG_P0 = D["layer"], D["sigma_P0"]
v = np.load("results/v3_3_5a/cache/v_p0.npy")
b = load_model(a.model)
pr = pd.read_csv(f"results/v3_3_5a/cache/{a.split}.csv")
DOSES = [float(x) for x in a.doses.split(",")]
print(f"{a.split}: n={len(pr)}  sigma_P0={SIG_P0:.4f}  doses(abs)={DOSES}", flush=True)

rows = []
for d in DOSES:
    R = generate_p0_only(b, list(pr.prompt), v=v, layer=LAYER, dose=d,
                         max_new_tokens=48, batch_size=a.batch)
    for i, t in enumerate(R["texts"]):
        rows.append({"pid": pr.pid.iloc[i], "dose": d,
                     "z_p0": float(R["z_attacked"][i]),
                     "refusal": int(is_refusal32(t)), "safe": safe_score32(t),
                     "coherence": coherence32(t)})
    s = pd.DataFrame([r for r in rows if r["dose"] == d])
    print(f"  dose {d:+7.1f} ({d/SIG_P0:+6.1f} sig)  refusal={s.refusal.mean():.3f}"
          f"  z_P0={s.z_p0.mean():+.3f}  coh={s.coherence.mean():.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"results/v3_3_5a/raw/p0_dose_{a.split}.csv", index=False)
z, y, pid = df.z_p0.to_numpy(), df.refusal.to_numpy(float), df.pid.to_numpy()
a0, b0 = fit_logistic(z, y)
tau = -a0/b0 if abs(b0) > 1e-12 else float("nan")
R2 = tau_beh_bootstrap(z, y, pid, n_boot=a.n_boot, seed=7)
beta_std = b0 * SIG_P0
zc = float(df[df.dose == 0.0].z_p0.mean())
sg = lambda x: 1/(1+np.exp(-(a0 + b0*x)))
dP = float(sg(zc + SIG_P0) - sg(zc - SIG_P0))
props = df.groupby("dose").refusal.mean()
gate = {
 "both_classes": bool(y.sum() >= 10 and (len(y)-y.sum()) >= 10),
 "transition_observed": bool(props.min() < 0.5 < props.max()),
 "tau_inside_range": bool(np.isfinite(tau) and z.min() <= tau <= z.max()),
 "ci_finite": bool(np.isfinite(R2["tau_ci95"][0]) and np.isfinite(R2["tau_ci95"][1])),
 "ci_width_le_3sigma": bool((R2["tau_ci95"][1]-R2["tau_ci95"][0]) <= 3*SIG_P0),
 "beta_std_ge_0p10": bool(abs(beta_std) >= 0.10),
 "slope_direction": bool(b0 > 0),
 "no_degeneration": bool(df.groupby("dose").coherence.mean().min() >= 0.6),
}
gate["all_pass"] = bool(all(gate.values()))
print(f"\nlogistic a={a0:+.4f} b_raw={b0:.6f}  beta_std={beta_std:.6f}  tau={tau:+.4f}")
print(f"  dP over [-1s,+1s] = {dP:+.6f}   CI95 {np.round(R2['tau_ci95'],3).tolist()}"
      f"   width/sigma = {(R2['tau_ci95'][1]-R2['tau_ci95'][0])/SIG_P0:.1f}")
print(f"  isotonic {isotonic_crossing(z,y)}")
print("  gate:", json.dumps(gate))
status = ("B1_P0_BOUNDARY_IDENTIFIED" if gate["all_pass"] else
          ("B2_P0_CAUSAL_BUT_BOUNDARY_IMPRECISE"
           if (R2["slope_ci95"][0] > 0 and props.max()-props.min() > 0.02)
           else "B3_P0_NOT_CAUSAL"))
print(f"\n*** {status} ***")
json.dump({"split": a.split, "doses": DOSES, "sigma_P0": SIG_P0,
           "logistic": {"a": a0, "b_raw": b0, "beta_std": beta_std, "tau": tau},
           "dP_2sigma": dP, "isotonic": isotonic_crossing(z, y),
           "bootstrap": {k: v2 for k, v2 in R2.items() if k != "tau_samples"},
           "gate": gate, "status": status,
           "refusal_by_dose": props.to_dict(),
           "refusal_span": float(props.max()-props.min())},
          open(TAB/f"p0_boundary_{a.split}.json", "w"), indent=2)
