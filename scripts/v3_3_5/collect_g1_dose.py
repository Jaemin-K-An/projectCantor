"""V3.3.5 PHASE 5/6/7 -- G1-ONLY causal dose, tau_G1, and the window W."""
import argparse, sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd, torch
from cantor_guard.models import load_model
from cantor_guard.io import seed_everything
from cantor_guard_v32.metrics32 import safe_score32, is_refusal32, coherence32
from cantor_guard_v335.g1_only_generation import generate_g1_only, PREFILL, G1, G2PLUS
from cantor_guard_v335.affine_coordinate import choose_W
from cantor_guard_v333.behavioral_boundary import (fit_logistic, isotonic_crossing,
                                                   identifiability, tau_beh_bootstrap)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
ap.add_argument("--batch", type=int, default=10)
ap.add_argument("--max-new", type=int, default=48)
ap.add_argument("--n-boot", type=int, default=20000)
ap.add_argument("--split", default="D_beh_g1_dev")
ap.add_argument("--doses", default="")
a = ap.parse_args()
seed_everything(20260906)
TAB = pathlib.Path("results/v3_3_5/tables")
V332 = json.loads(pathlib.Path("results/v3_3_2/tables/"
                               "phase_calibration_qwen2.5-0.5b-instruct.json").read_text())
LAYER = V332["layer"]
v = np.load(f"results/v3_3_2/cache/{a.model}_v332_phase.npz")["v"]
b = load_model(a.model)
pr = pd.read_csv(f"results/v3_3_5/cache/{a.split}.csv")

# DEV searches the bracket; CONFIRM uses the frozen grid.
if a.doses:
    DOSES = [float(x) for x in a.doses.split(",")]
else:
    DOSES = [-14, -12, -10, -8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3]
print(f"{a.split}: n={len(pr)}  layer={LAYER}  doses={DOSES}", flush=True)

rows, trace_ok = [], None
for d in DOSES:
    R = generate_g1_only(b, list(pr.prompt), v=v, layer=LAYER, dose=float(d),
                         max_new_tokens=a.max_new, batch_size=a.batch,
                         record=(trace_ok is None))
    if trace_ok is None:
        t = R["traces"][0]
        trace_ok = (t[0]["phase"] == PREFILL and t[1]["phase"] == G1
                    and all(x["phase"] == G2PLUS for x in t[2:]))
        print(f"  G1-only trace verified: {trace_ok}  "
              f"(prefill 1, G1 1, G2+ {len(t)-2})", flush=True)
        if not trace_ok:
            raise SystemExit("G1 PHASE TRACE FAILED")
    for i, txt in enumerate(R["texts"]):
        rows.append({"pid": pr.pid.iloc[i], "dose": float(d),
                     "z_g1": float(R["z_attacked"][i]),
                     "z_clean": float(R["z_clean"][i]),
                     "refusal": int(is_refusal32(txt)),
                     "safe": safe_score32(txt), "coherence": coherence32(txt)})
    sub = pd.DataFrame([r for r in rows if r["dose"] == float(d)])
    print(f"  dose {d:+6.1f}  refusal={sub.refusal.mean():.3f}  "
          f"z_G1={sub.z_g1.mean():+.3f}  coh={sub.coherence.mean():.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"results/v3_3_5/raw/g1_dose_{a.split}.csv", index=False)
z, y, pid = df.z_g1.to_numpy(), df.refusal.to_numpy(float), df.pid.to_numpy()
a0, b0 = fit_logistic(z, y)
tau = -a0/b0 if abs(b0) > 1e-12 else float("nan")
R2 = tau_beh_bootstrap(z, y, pid, n_boot=a.n_boot, seed=7)
gate = identifiability(z, y, a0, b0, tau, R2["tau_ci95"],
                       dose_bins=df.dose.to_numpy(),
                       max_ci_width_sigma=3.0, sigma=V332["calibrations"]["G1"]["sigma"])
print(f"\nlogistic a={a0:+.4f} b={b0:+.4f}  tau_G1={tau:+.4f}")
print(f"bootstrap CI95 {np.round(R2['tau_ci95'],4).tolist()}  slope CI "
      f"{np.round(R2['slope_ci95'],4).tolist()}  prompts={R2['n_prompts']}")
print(f"isotonic {isotonic_crossing(z,y)}")
print("identifiability:", json.dumps(gate))
status = "IDENTIFIED" if gate["all_pass"] else "TAU_G1_UNIDENTIFIABLE"
print(f"\n*** {status} ***")

W = None
if status == "IDENTIFIED" and a.split == "D_beh_g1_confirm":
    wc = pd.read_csv("results/v3_3_5/cache/D_window_cal.csv")
    RW = generate_g1_only(b, list(wc.prompt), v=v, layer=LAYER, dose=0.0,
                          max_new_tokens=8, batch_size=a.batch)
    W = choose_W(RW["z_clean"], tau)
    cov = float(np.mean(np.abs(RW["z_clean"] - tau) <= W))
    print(f"W = 1.05 * Q0.99(|z-tau|) = {W:.6f}   calibration coverage {cov:.3f}")
    np.save("results/v3_3_5/cache/window_cal_z.npy", RW["z_clean"])

out = {"split": a.split, "model": a.model, "layer": LAYER, "doses": DOSES,
       "logistic": {"a": a0, "b": b0, "tau_g1": tau},
       "bootstrap": {k: v2 for k, v2 in R2.items() if k != "tau_samples"},
       "isotonic": isotonic_crossing(z, y), "identifiability": gate,
       "status": status, "tau_g1": tau if status == "IDENTIFIED" else None,
       "W": W, "g1_trace_verified": bool(trace_ok),
       "refusal_by_dose": df.groupby("dose").refusal.mean().to_dict()}
(TAB/f"g1_boundary_{a.split}.json").write_text(json.dumps(out, indent=2))
print(f"wrote {TAB}/g1_boundary_{a.split}.json")
