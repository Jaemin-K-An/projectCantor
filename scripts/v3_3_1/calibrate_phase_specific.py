"""V3.3.1 PHASE 12/13 -- phase-specific calibration, uncertainty, budget fit.

V3.2 DEFECT. tau/sigma were fitted at the last PROMPT position and then applied
to every GENERATED position. The measured drift between those two phases was
2.18 sigma, so the controller's belief about where the boundary sits was
systematically wrong during exactly the tokens it was steering. V3.3.1 fits
separate calibrations and uses the generation one while generating.

It also measures the quantity the guard theory needs: the calibration
uncertainty radius in threat-coordinate units,

    eps_cal = |r(prompt calibration) - r(generation calibration)|

which is what a guard has to absorb. The theory then predicts the feasible
ratios directly: a crossing at the finest level must traverse the level-n
guard, so

    rho^(n-1) * (1 - 2 rho)  >=  eps_cal

Nothing here touches D_test.
"""
import argparse, sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, torch
from cantor_guard.models import load_model
from cantor_guard.probes import (RefusalDirections, diff_in_means_direction,
                                 calibrate, separability)
from cantor_guard.representations import last_token_residuals
from cantor_guard.datasets import load_jbb
from cantor_guard.attacks import TEST_ATTACKS, apply_prompt_attack
from cantor_guard.io import seed_everything
from cantor_guard_v32.generation32 import generate32
from cantor_guard_v32.splits import load_split
from cantor_guard_v331.guard_geometry import RHO_CANTOR

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--max-new", type=int, default=48)
ap.add_argument("--gamma", type=float, default=0.7)
ap.add_argument("--depth", type=int, default=5)
ap.add_argument("--seed", type=int, default=20260831)
a = ap.parse_args()
seed_everything(a.seed)

CACHE = pathlib.Path("results/v3_3_1/cache"); CACHE.mkdir(parents=True, exist_ok=True)
TAB = pathlib.Path("results/v3_3_1/tables"); TAB.mkdir(parents=True, exist_ok=True)
split = load_split("configs/v3_2/split.json")
harm, ben = load_jbb(); H = harm.set_index("pid")
blk = lambda b: [H.loc[p, "prompt"] for p in split["blocks"][b]]
d_dir, d_cal = blk("D_direction"), blk("D_calibration")
benign = list(ben.prompt)

b = load_model(a.model)
print(f"{a.model}: {b.n_layers}L {b.dtype} {b.device}", flush=True)

# direction on D_direction only (same rule as V3.2: depth fractions)
LAYERS = sorted({min(b.n_layers - 1, max(1, round(f * b.n_layers)))
                 for f in (0.30, 0.45, 0.60, 0.75)})
ah = last_token_residuals(b, d_dir, LAYERS, batch_size=a.batch)
al = last_token_residuals(b, benign[:len(d_dir)], LAYERS, batch_size=a.batch)
V = diff_in_means_direction(ah, al)
sep = np.atleast_1d(separability(ah, al, V))
BEST = LAYERS[int(sep.argmax())]
print(f"  separability {dict(zip(LAYERS, np.round(sep,3)))} -> layer {BEST}")

# ---- PROMPT-phase calibration (what V3.2 used everywhere) ----
Vb = V[[LAYERS.index(BEST)]]
ah_c = last_token_residuals(b, d_cal, [BEST], batch_size=a.batch)
al_c = last_token_residuals(b, benign[len(d_dir):len(d_dir) + len(d_cal)],
                            [BEST], batch_size=a.batch)
dirs_prompt = calibrate(ah_c, al_c, Vb, [BEST], "diff_means")
tau_p, sig_p = float(dirs_prompt.tau[0]), float(dirs_prompt.sigma[0])
print(f"  PROMPT calibration:     tau={tau_p:+.4f} sigma={sig_p:.4f}")

# ---- GENERATION-phase calibration (new) ----
gm = []
for tpl in ("authority_test", "stepwise_test"):
    P = [apply_prompt_attack(g, TEST_ATTACKS[tpl]) for g in d_cal]
    _, st = generate32(b, P, max_new_tokens=a.max_new, batch_size=a.batch,
                       dirs=dirs_prompt, layers=[BEST], controller=None,
                       harm_mask=np.ones(len(P), bool))
    gm.append(st)
# margins were reported in PROMPT units; convert back to raw projection
m_mean = float(np.mean([s["m_mean"] for s in gm]))
m_std = float(np.mean([s["m_std"] for s in gm]))
tau_g = tau_p + m_mean * sig_p          # generation-phase midpoint
sig_g = m_std * sig_p                   # generation-phase spread
print(f"  GENERATION calibration: tau={tau_g:+.4f} sigma={sig_g:.4f}")
print(f"  phase drift = {abs(tau_g - tau_p) / sig_p:.3f} sigma  (V3.2 applied "
      f"the prompt values throughout)")

# ---- uncertainty radius in THREAT-COORDINATE units ----
sig_r = lambda z, tau, sg: 1.0 / (1.0 + np.exp(np.clip(a.gamma * (z - tau) / sg, -60, 60)))
zs = np.concatenate([np.squeeze(ah_c) @ np.squeeze(Vb),
                     np.squeeze(al_c) @ np.squeeze(Vb)])
r_prompt = sig_r(zs, tau_p, sig_p)
r_gen = sig_r(zs, tau_g, sig_g)
eps = np.abs(r_prompt - r_gen)
q = {f"q{p}": float(np.quantile(eps, p / 100)) for p in (50, 75, 90, 95)}
print(f"  eps_cal (threat-coordinate units): {json.dumps({k: round(v,5) for k,v in q.items()})}")

# ---- what the guard theory predicts from that uncertainty ----
n = a.depth
grid = np.linspace(0.005, 0.495, 20001)
Gn = grid ** (n - 1) * (1 - 2 * grid)          # finest-level guard width
pred = {}
for key, e in q.items():
    ok = grid[Gn >= e]
    pred[key] = {"eps": e,
                 "feasible_lo": float(ok.min()) if len(ok) else None,
                 "feasible_hi": float(ok.max()) if len(ok) else None,
                 "cantor_feasible": bool(len(ok) and ok.min() <= RHO_CANTOR <= ok.max())}
print(f"  guard-feasible rho at depth {n}: " +
      json.dumps({k: (round(v['feasible_lo'], 4) if v['feasible_lo'] else None,
                      round(v['feasible_hi'], 4) if v['feasible_hi'] else None,
                      v['cantor_feasible']) for k, v in pred.items()}))
# rho maximising the finest guard -- another honest non-1/3 optimum
print(f"  rho maximising the level-{n} guard = (n-1)/(2n) = {(n-1)/(2*n):.4f}")

# MAXIMUM USEFUL DEPTH. The guard at level n has width rho^(n-1)*(1-2rho), which
# shrinks geometrically, so beyond some depth the guard is narrower than the
# calibration uncertainty and the level cannot separate anything. Solving
# rho^(n-1)(1-2rho) >= eps for n:
#     n <= 1 + [ln(eps) - ln(1-2rho)] / ln(rho)
import math
depth_rows = {}
for key, e in q.items():
    per_rho = {}
    for rr in (0.25, RHO_CANTOR, 0.40):
        nmax = 1.0 + (math.log(e) - math.log(1 - 2 * rr)) / math.log(rr)
        per_rho[round(rr, 4)] = int(math.floor(nmax + 1e-9))
    best = max(
        (1.0 + (math.log(e) - math.log(1 - 2 * x)) / math.log(x))
        for x in np.linspace(0.05, 0.45, 2001))
    depth_rows[key] = {"eps": e, "n_max_by_rho": per_rho,
                       "n_max_over_all_rho": int(math.floor(best + 1e-9))}
print("\n  MAXIMUM USEFUL DEPTH (guard wider than calibration uncertainty):")
for k, v in depth_rows.items():
    print(f"    {k} eps={v['eps']:.4f}  n_max: rho=1/3 -> {v['n_max_by_rho'][round(RHO_CANTOR,4)]}"
          f"   best-over-rho -> {v['n_max_over_all_rho']}")
print(f"    >>> V3.1/V3.2 ran n=5. Levels beyond n_max are below the "
      f"calibration noise floor.")

np.savez(CACHE / f"{a.model}_v331_dirs.npz", layers=np.array([BEST]),
         diff_means_v=dirs_prompt.v, tau_prompt=np.array([tau_p]),
         sigma_prompt=np.array([sig_p]), tau_gen=np.array([tau_g]),
         sigma_gen=np.array([sig_g]), separability=sep)
out = {"model": a.model, "layer": int(BEST), "gamma": a.gamma, "depth": n,
       "separability": {str(l): float(v) for l, v in zip(LAYERS, sep)},
       "tau_prompt": tau_p, "sigma_prompt": sig_p,
       "tau_generation": tau_g, "sigma_generation": sig_g,
       "phase_drift_sigma": abs(tau_g - tau_p) / sig_p,
       "eps_cal_quantiles": q, "guard_feasible": pred,
       "rho_max_finest_guard": (n - 1) / (2 * n),
       "max_useful_depth": depth_rows,
       "split_sha256": split["split_sha256"]}
(TAB / f"phase_calibration_{a.model}.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote {TAB}/phase_calibration_{a.model}.json")
