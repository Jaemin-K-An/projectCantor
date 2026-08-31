"""V3.3.2 PHASE 5/6/7/8 -- collect, calibrate, bootstrap. D_calibration only."""
import argparse, sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, torch
from cantor_guard.models import load_model
from cantor_guard.probes import diff_in_means_direction, separability
from cantor_guard.representations import last_token_residuals
from cantor_guard.datasets import load_jbb
from cantor_guard.io import seed_everything
from cantor_guard_v32.splits import load_split
from cantor_guard_v332.phase_residuals import collect_phase_residuals
from cantor_guard_v332.calibration import phase_calibration, threat_coordinate
from cantor_guard_v332.uncertainty import (u_est_bootstrap, u_phase_bias,
                                           u_state_dispersion)
from cantor_guard_v332.absolute_guard import (feasible_interval, rho_abs_star,
                                              G_n_max, rho_guard_max, RHO_CANTOR)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--gamma", type=float, default=0.7)
ap.add_argument("--n-boot", type=int, default=20000)
ap.add_argument("--seed", type=int, default=20260901)
a = ap.parse_args()
seed_everything(a.seed)

TAB = pathlib.Path("results/v3_3_2/tables"); TAB.mkdir(parents=True, exist_ok=True)
CACHE = pathlib.Path("results/v3_3_2/cache"); CACHE.mkdir(parents=True, exist_ok=True)
split = load_split("configs/v3_2/split.json")
harm, ben = load_jbb(); H = harm.set_index("pid")
d_dir = [H.loc[p, "prompt"] for p in split["blocks"]["D_direction"]]
d_cal = [H.loc[p, "prompt"] for p in split["blocks"]["D_calibration"]]
benign = list(ben.prompt)
cal_benign = benign[len(d_dir):len(d_dir) + len(d_cal)]

b = load_model(a.model)
LAYERS = sorted({min(b.n_layers - 1, max(1, round(f * b.n_layers)))
                 for f in (0.30, 0.45, 0.60, 0.75)})
ah = last_token_residuals(b, d_dir, LAYERS, batch_size=a.batch)
al = last_token_residuals(b, benign[:len(d_dir)], LAYERS, batch_size=a.batch)
V = diff_in_means_direction(ah, al)
sep = np.atleast_1d(separability(ah, al, V))
LAYER = LAYERS[int(sep.argmax())]
v = np.squeeze(V[[LAYERS.index(LAYER)]])
print(f"{a.model}: layer {LAYER}  separability {dict(zip(LAYERS, np.round(sep,3)))}",
      flush=True)

print("\ncollecting phase residuals (controller OFF, CLEAN prompts)", flush=True)
RH = collect_phase_residuals(b, d_cal, LAYER, batch_size=a.batch, record_trace=True)
RB = collect_phase_residuals(b, cal_benign, LAYER, batch_size=a.batch, record_trace=True)
tr = RH["traces"] + RB["traces"]
ok = all(t["ok"] for t in tr)
print(f"  phase-trace consistency: {ok}  "
      f"(prefill/decode per batch: {tr[0]['n_prefill']}/{tr[0]['n_decode']})")
if not ok:
    raise SystemExit("PHASE TRACE INCONSISTENT -- refusing to continue")

cals, proj = {}, {}
for key, name in (("prompt_last", "P"), ("decode1", "G1"),
                  ("decode1_4", "G1_4"), ("decode5_8", "G5_8")):
    zh, zb = RH[key] @ v, RB[key] @ v
    proj[name] = {"harmful": zh.tolist(), "harmless": zb.tolist()}
    cals[name] = phase_calibration(zh, zb)
    c = cals[name]
    print(f"  phase {name:5s}  tau={c['tau']:+8.4f}  sigma={c['sigma']:7.4f}  "
          f"sep={c['separability']:+6.3f}  (nH={c['n_harmful']}, nB={c['n_harmless']})")

print("\nU_EST -- prompt-clustered bootstrap of the boundary LOCATION", flush=True)
zh1, zb1 = RH["decode1"] @ v, RB["decode1"] @ v
U = u_est_bootstrap(zh1, zb1, gamma=a.gamma, n_boot=a.n_boot, seed=a.seed)
dq = U["delta_abs_quantiles"]
print(f"  delta_abs (G1): " + json.dumps({k: round(x, 5) for k, x in dq.items()}))
print(f"  tau 95% CI {np.round(U['tau_ci95'],4).tolist()}   "
      f"sigma 95% CI {np.round(U['sigma_ci95'],4).tolist()}")

print("\nU_PHASE -- SYSTEMATIC BIAS (not uncertainty)")
UP = u_phase_bias(cals["P"], cals["G1"], a.gamma)
print(f"  m_phase = {UP['m_phase_sigma']:+.3f} sigma   "
      f"delta_phase_r = {UP['delta_phase_r']:.5f}   "
      f"sigma_G/sigma_P = {UP['scale_ratio_sigma_G_over_P']:.3f}")

print("\nabsolute guard feasibility from delta_abs (rho-INDEPENDENT input)")
pred = {}
for n in (2, 3, 5):
    row = {}
    for k, dv in dq.items():
        iv = feasible_interval(n, dv)
        row[k] = {"delta": dv, "interval": None if iv is None else [iv[0], iv[1]],
                  "rho_abs_star": None if iv is None else iv[1],
                  "cantor_feasible": bool(iv and iv[0] <= RHO_CANTOR <= iv[1])}
    pred[n] = row
    q50 = row["q50"]
    star = q50["rho_abs_star"]
    shown = "INFEASIBLE" if star is None else f"rho* = {star:.5f}"
    print(f"  n={n}: G_max={G_n_max(n):.6f}  rho_guard_max={rho_guard_max(n):.4f}"
          f"  | q50 delta={q50['delta']:.5f} -> {shown}"
          f"  cantor_feasible={q50['cantor_feasible']}")

# bootstrap CI for the predicted rho at each depth
samples = U["delta_abs_samples"]
sub = samples[np.random.default_rng(a.seed).integers(0, len(samples), 4000)]
rho_ci = {}
for n in (2, 3, 5):
    rs = np.array([rho_abs_star(n, float(d)) if rho_abs_star(n, float(d)) else np.nan
                   for d in sub], float)
    frac = float(np.mean(~np.isnan(rs)))
    rho_ci[n] = {"feasible_fraction": frac,
                 "rho_pred_median": float(np.nanmedian(rs)) if frac > 0 else None,
                 "rho_pred_ci95": [float(np.nanquantile(rs, .025)),
                                   float(np.nanquantile(rs, .975))] if frac > 0 else None}
    print(f"  n={n}: feasible in {frac:.1%} of bootstraps; "
          f"rho_pred={rho_ci[n]['rho_pred_median']}")

np.savez(CACHE / f"{a.model}_v332_phase.npz", v=v, layer=np.array([LAYER]),
         zh_P=RH["prompt_last"] @ v, zb_P=RB["prompt_last"] @ v,
         zh_G1=zh1, zb_G1=zb1)
out = {"model": a.model, "layer": int(LAYER), "gamma": a.gamma,
       "separability": {str(l): float(x) for l, x in zip(LAYERS, sep)},
       "calibrations": cals, "projections": proj,
       "U_EST": {k: v2 for k, v2 in U.items() if k != "delta_abs_samples"},
       "U_PHASE": UP, "absolute_guard_prediction": pred,
       "rho_pred_bootstrap": rho_ci,
       "phase_trace_ok": ok, "n_calibration_prompts": len(d_cal)}
(TAB / f"phase_calibration_{a.model}.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote {TAB}/phase_calibration_{a.model}.json")
