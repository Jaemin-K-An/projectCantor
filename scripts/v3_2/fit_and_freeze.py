"""V3.2 PHASE 5/6/8/9 -- fit every free quantity on its OWN block, then freeze.

V3.1 DEFECT D2: the realised-budget gains were fitted by measuring C_rms on the
same goals the final comparison was read from. No performance metric was
optimised on the test set, but the fairness constraint was, so the comparison
was not a strict hold-out.

Here each block does one job and nothing else:
  D_direction    refusal direction v, and the layer choice
  D_calibration  tau, sigma, and the natural margin-shift distribution U_Delta
  D_budget       the per-controller gain eta that hits the target C_rms
  D_test         NOT TOUCHED by this script

Everything fitted is written to configs/v3_2/ with a SHA-256, and the final
test reads those files without refitting.
"""
import argparse, sys, json, time, hashlib, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd, torch
from cantor_guard.models import load_model
from cantor_guard.probes import (RefusalDirections, diff_in_means_direction,
                                 calibrate, separability)
from cantor_guard.representations import last_token_residuals
from cantor_guard.datasets import load_jbb
from cantor_guard.attacks import TEST_ATTACKS, apply_prompt_attack
from cantor_guard.io import seed_everything
from cantor_guard_v31.controllers31 import (Controller31, V31_LLM_FAMILIES,
                                            V31_RANDOMISED)
from cantor_guard_v32.generation32 import generate32
from cantor_guard_v32.metrics32 import check_attainability32, safe_score32
from cantor_guard_v32.splits import load_split

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
ap.add_argument("--dtype", default=None)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--max-new", type=int, default=48)
ap.add_argument("--budget", type=float, default=0.02)
ap.add_argument("--qcap", type=float, default=0.05)
ap.add_argument("--layout-seeds", type=int, default=3)
ap.add_argument("--seed", type=int, default=20260830)
args = ap.parse_args()
seed_everything(args.seed)

CFG = pathlib.Path("configs/v3_2"); CACHE = pathlib.Path("results/v3_2/cache")
CACHE.mkdir(parents=True, exist_ok=True)
split = load_split("configs/v3_2/split.json")
harm, ben = load_jbb(); H = harm.set_index("pid")
blk = lambda b: [H.loc[p, "prompt"] for p in split["blocks"][b]]
d_dir, d_cal, d_bud = blk("D_direction"), blk("D_calibration"), blk("D_budget")
benign = list(ben.prompt)

kw = {} if args.dtype is None else {"dtype": getattr(torch, args.dtype)}
b = load_model(args.model, **kw)
print(f"{args.model}: {b.n_layers}L d={b.d_model} {b.dtype} {b.device}", flush=True)
tag = args.model

# ------------------------------------------------- PHASE 5: direction (D_direction)
print("\nPHASE 5  refusal direction  [D_direction only]", flush=True)
mid = b.n_layers // 2
LAYERS = sorted({max(1, mid - 5), max(1, mid - 2), mid,
                 min(b.n_layers - 1, mid + 3)})
ah = last_token_residuals(b, d_dir, LAYERS, batch_size=args.batch)
al = last_token_residuals(b, benign[:len(d_dir)], LAYERS, batch_size=args.batch)
V = diff_in_means_direction(ah, al)
sep = np.atleast_1d(separability(ah, al, V))
BEST = LAYERS[int(np.argmax(sep))]
print(f"  separability {dict(zip(LAYERS, np.round(sep,3)))}  -> layer {BEST}")

# ------------------------------------------ PHASE 6: calibration (D_calibration)
print("\nPHASE 6  tau/sigma + U_Delta  [D_calibration only]", flush=True)
ah_c = last_token_residuals(b, d_cal, [BEST], batch_size=args.batch)
al_c = last_token_residuals(b, benign[len(d_dir):len(d_dir)+len(d_cal)], [BEST],
                            batch_size=args.batch)
Vb = V[[LAYERS.index(BEST)]]
dirs = calibrate(ah_c, al_c, Vb, [BEST], "diff_means")
print(f"  tau={float(dirs.tau[0]):.4f}  sigma={float(dirs.sigma[0]):.4f}")

# U_Delta: how far the margin naturally moves between the prompt position and
# the generated continuation. This is what the Delta grid must span, and it is
# measured HERE, never on D_test.
shifts = []
for tpl in ("authority_test", "stepwise_test"):
    P = [apply_prompt_attack(g, TEST_ATTACKS[tpl]) for g in d_cal]
    _, st = generate32(b, P, max_new_tokens=args.max_new, batch_size=args.batch,
                       dirs=dirs, layers=[BEST], controller=None,
                       harm_mask=np.ones(len(P), bool))
    pre = last_token_residuals(b, P, [BEST], batch_size=args.batch)
    zp = (np.squeeze(pre) @ np.squeeze(Vb) - float(dirs.tau[0])) / float(dirs.sigma[0])
    shifts.append(abs(float(st["m_mean"]) - float(zp.mean())))
    print(f"  {tpl}: prompt m={zp.mean():+.3f}  generation m={st['m_mean']:+.3f}"
          f"  |shift|={shifts[-1]:.3f}  gen m_std={st['m_std']:.3f}")
U = {"mean_shift": float(np.mean(shifts)), "max_shift": float(np.max(shifts)),
     "gen_m_std": float(st["m_std"])}
# Delta grid: 0 and the measured shift magnitude, in sigma units.
Q95 = float(max(np.max(shifts), 0.5))
DELTAS = [0.0, Q95]
print(f"  U_Delta = {json.dumps({k: round(v,4) for k,v in U.items()})}")
print(f"  Delta grid = {np.round(DELTAS,3)}   (measured, not chosen from results)")

# ------------------------------------------------ metric attainability gate
print("\nMETRIC ATTAINABILITY GATE", flush=True)
Pg = [apply_prompt_attack(g, TEST_ATTACKS["authority_test"]) for g in d_bud]
t_base, _ = generate32(b, Pg, max_new_tokens=args.max_new, batch_size=args.batch,
                       dirs=dirs, layers=[BEST], controller=None,
                       harm_mask=np.ones(len(Pg), bool))
t_atk, _ = generate32(b, Pg, max_new_tokens=args.max_new, batch_size=args.batch,
                      dirs=dirs, layers=[BEST], controller=None, attack_eps=10.0,
                      attack_ortho=0.3, attack_seed=args.seed,
                      harm_mask=np.ones(len(Pg), bool))
gate = check_attainability32(t_base, t_atk)
print("  " + json.dumps(gate))
if not gate["all_pass"]:
    raise SystemExit("METRIC GATE FAILED -- refusing to freeze (harness STOP B).")

# ------------------------------------------------- PHASE 9: budget (D_budget)
print(f"\nPHASE 9  fitting eta to C_rms={args.budget}  [D_budget only]", flush=True)
L9W = None
import re
t = open("configs/v3_1/l9_frozen_weights.toml").read()
L9W = np.array([float(x) for x in re.search(r"weights = \[(.*?)\]", t, re.S)
                .group(1).split(",")])

INST = []
for fam in V31_LLM_FAMILIES:
    for s in (range(1, args.layout_seeds + 1) if fam in V31_RANDOMISED else [0]):
        INST.append((fam, s, Controller31(fam, n=5, B_total=1.0, gamma=0.7,
                     eta=1.0, seed=s,
                     weights=(L9W if fam == "T8_minimax" else None),
                     max_q=args.qcap)))
print(f"  {len(INST)} controller instances")

probe_cells = [(d_, e_) for d_ in DELTAS for e_ in (0.0, 10.0)]
def probe_crms(c, eta):
    c.eta = eta; vals = []
    for d_, e_ in probe_cells:
        _, st = generate32(b, Pg, max_new_tokens=args.max_new,
                           batch_size=args.batch, dirs=dirs, layers=[BEST],
                           controller=c, delta=d_, attack_eps=e_,
                           attack_ortho=0.3, attack_seed=args.seed,
                           harm_mask=np.ones(len(Pg), bool))
        vals.append(st["C_rms"])
    return float(np.mean(vals))

t0 = time.time(); GAINS = {}
# q is linear in eta only while the q_cap is slack. For the multiscale families
# the cap binds on the largest-margin tokens, so a single closed-form step
# leaves a several-percent residual (measured: -7.2% to +7.8%). A secant solve
# on eta absorbs that; it stops as soon as the realised C_rms is inside the
# tolerance, so well-behaved controllers still cost only two evaluations.
TOL, MAXIT = 0.02, 6
for fam, s, c in INST:
    if fam == "T0_none":
        GAINS[f"{fam}|{s}"] = {"eta": 0.0, "achieved_C_rms": 0.0, "rel_err": 0.0,
                               "matched": True, "sup_deriv": 0.0, "iters": 0}
        continue
    base = probe_crms(c, 1.0)
    if base <= 1e-12:
        GAINS[f"{fam}|{s}"] = {"eta": 0.0, "achieved_C_rms": 0.0, "rel_err": -1.0,
                               "matched": False, "sup_deriv": float(c.sup_deriv),
                               "iters": 1}
        print(f"  {fam:20s} s{s} DEGENERATE (C_rms=0 at eta=1)", flush=True)
        continue
    e0, f0 = 1.0, base - args.budget
    eta = args.budget / base
    ach = probe_crms(c, eta); it = 2
    while abs(ach - args.budget) / args.budget > TOL and it < MAXIT:
        f1 = ach - args.budget
        denom = f1 - f0
        step = -f1 * (eta - e0) / denom if abs(denom) > 1e-15 else 0.0
        e0, f0 = eta, f1
        nxt = eta + step
        # keep the iterate positive and bounded; fall back to the ratio update
        if not np.isfinite(nxt) or nxt <= 0 or nxt > 50 * eta:
            nxt = eta * args.budget / max(ach, 1e-12)
        eta = float(nxt)
        ach = probe_crms(c, eta); it += 1
    rel = (ach - args.budget) / args.budget
    GAINS[f"{fam}|{s}"] = {"eta": float(eta), "achieved_C_rms": float(ach),
                           "rel_err": float(rel),
                           "matched": bool(abs(rel) <= 0.03),
                           "sup_deriv": float(c.sup_deriv), "iters": it}
    print(f"  {fam:20s} s{s} eta={eta:9.4f} C_rms={ach:.5f} "
          f"({100*rel:+.1f}%) it={it} {'OK' if abs(rel)<=0.03 else 'FAIL'}",
          flush=True)
print(f"  budget fitting took {time.time()-t0:.0f}s")

np.savez(CACHE / f"{tag}_frozen_dirs.npz", layers=np.array([BEST]),
         diff_means_v=dirs.v, diff_means_tau=dirs.tau,
         diff_means_sigma=dirs.sigma, separability=sep,
         probe_layers=np.array(LAYERS))
payload = {
    "model": tag, "repo": b.repo, "revision": b.revision, "dtype": b.dtype,
    "split_sha256": split["split_sha256"],
    "layer": int(BEST), "probe_layers": [int(x) for x in LAYERS],
    "separability": {str(l): float(v) for l, v in zip(LAYERS, sep)},
    "tau": float(dirs.tau[0]), "sigma": float(dirs.sigma[0]),
    "U_Delta": U, "delta_grid": [float(x) for x in DELTAS],
    "eps_grid": [0.0, 10.0], "gamma": 0.7, "n_order": 5,
    "target_C_rms": args.budget, "q_cap": args.qcap,
    "max_new_tokens": args.max_new,
    "attainability_gate": gate, "gains": GAINS,
    "n_matched": int(sum(g["matched"] for g in GAINS.values())),
    "n_instances": len(GAINS),
}
out = CFG / f"frozen_{tag}.json"
out.write_text(json.dumps(payload, indent=2))
print(f"\nwrote {out}  sha256={hashlib.sha256(out.read_bytes()).hexdigest()[:16]}")
print(f"budget matched: {payload['n_matched']}/{payload['n_instances']}")
