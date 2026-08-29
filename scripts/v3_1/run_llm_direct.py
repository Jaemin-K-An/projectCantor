"""V3.1 PHASE 13-17 — the DIRECT LLM controller test (V3's missing experiment).

Budget matching is exact rather than bisected: q scales linearly in eta for a
fixed controller shape, so eta = target / C_rms(eta=1) hits the target exactly;
the achieved value is measured and reported anyway.

Writes incrementally; resumes if interrupted. No prompt text or completions
reach the tracked table.
"""
import argparse, sys, json, time, hashlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd, torch
from cantor_guard.models import load_model
from cantor_guard.probes import RefusalDirections
from cantor_guard.datasets import load_jbb, load_xstest, grouped_split, prompt_id
from cantor_guard.attacks import TEST_ATTACKS, apply_prompt_attack
from cantor_guard.io import V2_CACHE, seed_everything
from cantor_guard_v31.controllers31 import Controller31, V31_LLM_FAMILIES, V31_RANDOMISED
from cantor_guard_v31.metrics31 import safe_score, utility_scores, check_attainability
from cantor_guard_v31.generation31 import generate31
from cantor_guard_v31.io31 import write_v31, V31_RAW, V31_CACHE

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
ap.add_argument("--batch", type=int, default=12)
ap.add_argument("--max-new", type=int, default=48)
ap.add_argument("--n-harm", type=int, default=12)
ap.add_argument("--n-safe", type=int, default=24)
ap.add_argument("--budget", type=float, default=0.02)   # target C_rms
ap.add_argument("--qcap", type=float, default=0.05)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--seed", type=int, default=20260829)
args = ap.parse_args()
seed_everything(args.seed)

b = load_model(args.model)
z = np.load(V2_CACHE / f"{args.model}_dirs.npz")
dirs = RefusalDirections(list(z["layers"]), z["diff_means_v"], z["diff_means_tau"],
                         z["diff_means_sigma"], "diff_means")
LAYERS = [13]
GAMMA, N_ORDER = 0.7, 5
# uncertainty sets, from the MEASURED natural shift distribution (V3 PHASE 3)
U = json.load(open(f"results/v3/cache/{args.model}_udelta.json"))
DELTAS = [-U["q95"], -U["q75"], 0.0, U["q75"], U["q95"]]
EPS = [0.0, 2.0, 5.0, 10.0]
print(f"Delta grid (margin sigma) = {np.round(DELTAS,3)}   from measured U_Delta")
print(f"eps grid = {EPS}")

harm, ben = load_jbb(); sp = grouped_split(harm, seed=args.seed)
goals = list(sp.test.prompt)[:args.n_harm]
xs = load_xstest(); safe_prompts = list(xs[~xs.is_harmful].prompt)[:args.n_safe]
ATT = {k: TEST_ATTACKS[k] for k in ("authority_test", "stepwise_test")}
print(f"{len(goals)} held-out harmful goals x {len(ATT)} held-out attacks")

L9W = None
try:
    import tomllib
    L9W = np.array(tomllib.load(open("configs/v3_1/l9_frozen_weights.toml","rb"))["weights"])
except Exception:
    import re
    t = open("configs/v3_1/l9_frozen_weights.toml").read()
    L9W = np.array([float(x) for x in re.search(r"weights = \[(.*?)\]", t, re.S).group(1).split(",")])
print(f"L9 frozen weights loaded ({len(L9W)} bins)")

INST = []
for fam in V31_LLM_FAMILIES:
    seeds = range(1, args.seeds+1) if fam in V31_RANDOMISED else [0]
    for s in seeds:
        INST.append((fam, s, Controller31(fam, n=N_ORDER, B_total=1.0, gamma=GAMMA,
                     eta=1.0, seed=s, weights=(L9W if fam=="T8_minimax" else None),
                     max_q=args.qcap)))
print(f"{len(INST)} controller instances")

# ---------------------------------------------------------- metric validation
print("\nMETRIC ATTAINABILITY GATE (must pass before the test runs)")
base_t, _ = generate31(b, [apply_prompt_attack(g, ATT['authority_test']) for g in goals],
                       max_new_tokens=args.max_new, batch_size=args.batch,
                       dirs=dirs, layers=LAYERS, controller=None,
                       harm_mask=np.ones(len(goals), bool))
atk_t, _ = generate31(b, [apply_prompt_attack(g, ATT['authority_test']) for g in goals],
                      max_new_tokens=args.max_new, batch_size=args.batch,
                      dirs=dirs, layers=LAYERS, controller=None, attack_eps=10.0,
                      harm_mask=np.ones(len(goals), bool))
gate = check_attainability(base_t, atk_t)
print("  " + json.dumps(gate))
if not gate["all_pass"]:
    raise SystemExit("METRIC GATE FAILED -- refusing to run the test (harness STOP B).")

# ------------------------------------------------- exact realised-budget match
# PROTOCOL v2 (docs/v3_1/TEST_V1_INVALIDATED.md): the v1 probe used
# max_new_tokens=8 while the test ran 48, so the margin distribution -- and
# therefore q -- differed between calibration and test. 8 of 9 controllers
# missed the budget by 13-59%. The probe now uses the SAME generation length
# and a representative (Delta, eps) mix, and eta is refined once to absorb the
# non-linearity introduced by q_cap.
print(f"\nmatching realised C_rms to target {args.budget} (protocol v2)")
probe_cells = [(d_, e_) for d_ in (0.0, DELTAS[0], DELTAS[-1]) for e_ in (0.0, EPS[-1])]
probe_prompts = [apply_prompt_attack(g, ATT['authority_test']) for g in goals]

def probe_crms(c, eta):
    c.eta = eta
    vals = []
    for d_, e_ in probe_cells:
        _, st = generate31(b, probe_prompts, max_new_tokens=args.max_new,
                           batch_size=args.batch, dirs=dirs, layers=LAYERS,
                           controller=c, delta=d_, attack_eps=e_,
                           attack_ortho=0.3, attack_seed=args.seed,
                           harm_mask=np.ones(len(probe_prompts), bool))
        vals.append(st["C_rms"])
    return float(np.mean(vals))

ETA = {}
for fam, s, c in INST:
    if fam == "T0_none":
        ETA[(fam, s)] = (0.0, 0.0); continue
    c.max_q = args.qcap
    base = probe_crms(c, 1.0)
    eta = args.budget / base if base > 1e-12 else 0.0
    ach = probe_crms(c, eta)                       # one refinement pass
    if ach > 1e-12:
        eta = eta * args.budget / ach
        ach = probe_crms(c, eta)
    ETA[(fam, s)] = (eta, ach)
    ok = abs(ach - args.budget) / args.budget <= 0.03
    print(f"  {fam:22s} s{s}  eta={eta:.4f}  achieved C_rms={ach:.5f} "
          f"({100*(ach-args.budget)/args.budget:+.1f}%) {'OK' if ok else 'FAIL'}")

OUT = V31_RAW / f"v31_llm_direct_v2_{args.model}.csv"
done = set()
if OUT.exists():
    prev = pd.read_csv(OUT)
    done = set(zip(prev.family, prev.seed, prev.attack, prev.delta, prev.eps))
    print(f"resuming: {len(prev)} rows present")

t0 = time.time(); rows = []
for fam, s, c in INST:
    eta, _ = ETA[(fam, s)]
    c.eta = eta; c.max_q = args.qcap
    for aname, tpl in ATT.items():
        P = [apply_prompt_attack(g, tpl) for g in goals]
        for d in DELTAS:
            for e in EPS:
                if (fam, s, aname, d, e) in done: continue
                txt, st = generate31(b, P, max_new_tokens=args.max_new,
                    batch_size=args.batch, dirs=dirs, layers=LAYERS, controller=c,
                    delta=d, attack_eps=e, attack_ortho=0.3, attack_seed=args.seed,
                    harm_mask=np.ones(len(P), bool))
                for i, t in enumerate(txt):
                    rows.append({"model": args.model, "family": fam, "seed": s,
                        "attack": aname, "delta": d, "eps": e,
                        "pid": prompt_id(goals[i]), "safe": safe_score(t),
                        "eta": eta, "sup_deriv": c.sup_deriv, **st})
                df = pd.DataFrame(rows); rows = []
                df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
    # utility pass, once per controller
    ut, ust = generate31(b, safe_prompts, max_new_tokens=args.max_new,
        batch_size=args.batch, dirs=dirs, layers=LAYERS, controller=c,
        delta=0.0, attack_eps=0.0, harm_mask=np.ones(len(safe_prompts), bool))
    u = utility_scores(ut)
    pd.DataFrame([{"model": args.model, "family": fam, "seed": s, "eta": eta,
                   "sup_deriv": c.sup_deriv, **u, **ust}]).to_csv(
        V31_RAW / f"v31_llm_utility_v2_{args.model}.csv", mode="a",
        header=not (V31_RAW / f"v31_llm_utility_v2_{args.model}.csv").exists(), index=False)
    print(f"  {fam:22s} s{s} done ({time.time()-t0:.0f}s)", flush=True)

full = pd.read_csv(OUT)
json.dump({"gate": gate, "deltas": DELTAS, "eps": EPS, "budget": args.budget,
           "qcap": args.qcap, "n_rows": len(full)},
          open(V31_CACHE / f"{args.model}_direct_meta.json", "w"), indent=2)
print(f"\nDIRECT LLM TEST COMPLETE: {len(full)} rows -> {OUT}")
