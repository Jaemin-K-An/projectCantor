"""V3.3.1 PHASE 15 -- LLM rho sweep with phase-specific calibration.

Purpose is NOT to find a setting where Cantor wins on safety; V3.2 settled that
and it is not re-opened. The question is whether the GUARD theory's prediction
shows up: the measured calibration uncertainty admits a useful depth of only
n <= 2, so at n = 5 every rho should be indistinguishable (levels 3-5 sit below
the noise floor), while at a feasible depth rho should start to matter.

Only rho and depth vary. Model, direction, layer, split, budget target, q_cap,
attacks and evaluator are all held fixed, and eta is fitted per (rho, depth) on
D_budget and never retuned on D_test.
"""
import argparse, sys, json, time, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd, torch
from cantor_guard.models import load_model
from cantor_guard.probes import RefusalDirections
from cantor_guard.datasets import load_jbb, load_xstest
from cantor_guard.attacks import TEST_ATTACKS, apply_prompt_attack
from cantor_guard.io import seed_everything
from cantor_guard_v32.generation32 import generate32
from cantor_guard_v32.metrics32 import safe_score32, utility_scores32
from cantor_guard_v32.splits import load_split
from cantor_guard_v331.rho_family import RhoBarrier
from cantor_guard_v331.guard_geometry import RHO_CANTOR

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
ap.add_argument("--batch", type=int, default=10)
ap.add_argument("--max-new", type=int, default=48)
ap.add_argument("--budget", type=float, default=0.02)
ap.add_argument("--qcap", type=float, default=0.05)
ap.add_argument("--rhos", default="0.24,0.3333333333333333,0.40")
ap.add_argument("--depths", default="2,5")
ap.add_argument("--attacks", default="authority_test")
ap.add_argument("--n-safe", type=int, default=24)
ap.add_argument("--seed", type=int, default=20260831)
a = ap.parse_args()
seed_everything(a.seed)

TAB = pathlib.Path("results/v3_3_1/tables"); RAW = pathlib.Path("results/v3_3_1/raw")
RAW.mkdir(parents=True, exist_ok=True)
CAL = json.loads((TAB / f"phase_calibration_{a.model}.json").read_text())
z = np.load(f"results/v3_3_1/cache/{a.model}_v331_dirs.npz")
LAYER = int(CAL["layer"])
# GENERATION-phase calibration is what steers the generated tokens (V3.2 defect)
dirs = RefusalDirections([LAYER], z["diff_means_v"], z["tau_gen"],
                         z["sigma_gen"], "diff_means")
print(f"layer={LAYER}  tau_gen={float(z['tau_gen'][0]):+.4f} "
      f"sigma_gen={float(z['sigma_gen'][0]):.4f}  (phase drift "
      f"{CAL['phase_drift_sigma']:.2f} sigma)")

split = load_split("configs/v3_2/split.json")
harm, ben = load_jbb(); H = harm.set_index("pid")
budget_goals = [H.loc[p, "prompt"] for p in split["blocks"]["D_budget"]]
TEST_PIDS = split["blocks"]["D_test"]
test_goals = [H.loc[p, "prompt"] for p in TEST_PIDS]
xs = load_xstest(); safe_prompts = list(xs[~xs.is_harmful].prompt)[:a.n_safe]
ATT = {k: TEST_ATTACKS[k] for k in a.attacks.split(",")}
RHOS = [float(x) for x in a.rhos.split(",")]
DEPTHS = [int(x) for x in a.depths.split(",")]
DELTAS = [0.0, 2.0]
EPS = [0.0, 2.0]
print(f"rho={[round(r,4) for r in RHOS]}  depth={DEPTHS}  "
      f"D_test={len(test_goals)} goals")

b = load_model(a.model)

def make(rho, n, eta):
    c = RhoBarrier(rho, n, 1.0 / n)      # E0 = B_total/n, fixed total action
    c.eta, c.gamma, c.max_q, c.harm_gate = eta, 0.7, a.qcap, True
    c.family = f"rho{rho:.4f}_n{n}"
    return c

# ---------------- budget fit on D_budget only ----------------
Pb = [apply_prompt_attack(g, list(ATT.values())[0]) for g in budget_goals]
def crms(c, eta):
    c.eta = eta; vals = []
    for d_ in DELTAS:
        for e_ in EPS:
            _, st = generate32(b, Pb, max_new_tokens=a.max_new, batch_size=a.batch,
                               dirs=dirs, layers=[LAYER], controller=c, delta=d_,
                               attack_eps=e_, attack_ortho=0.3, gen_seed=a.seed,
                               harm_mask=np.ones(len(Pb), bool))
            vals.append(st["C_rms"])
    return float(np.mean(vals))

print("\nfitting eta on D_budget (bisection, C_rms monotone in eta)", flush=True)
GAINS = {}
for n in DEPTHS:
    for rho in RHOS:
        c = make(rho, n, 1.0)
        base = crms(c, 1.0)
        if base <= 1e-12:
            GAINS[(rho, n)] = (0.0, 0.0); continue
        lo, hi = 0.0, a.budget / base
        ach = crms(c, hi); it = 2; grow = 0
        while ach < a.budget * 0.98 and grow < 8:
            lo, hi = hi, hi * 2.0; ach = crms(c, hi); it += 1; grow += 1
        eta = hi
        while abs(ach - a.budget) / a.budget > 0.02 and it < 14:
            mid = 0.5 * (lo + hi); am = crms(c, mid); it += 1
            if am < a.budget: lo = mid
            else: hi = mid
            eta, ach = mid, am
        GAINS[(rho, n)] = (eta, ach)
        ok = abs(ach - a.budget) / a.budget <= 0.03
        print(f"  rho={rho:.4f} n={n}  eta={eta:9.4f}  C_rms={ach:.5f} "
              f"({100*(ach-a.budget)/a.budget:+.1f}%) {'OK' if ok else 'FAIL'}",
              flush=True)

# ---------------- D_test sweep ----------------
OUT = RAW / f"v331_llm_rho_{a.model}.csv"
t0 = time.time(); done = set()
if OUT.exists():
    prev = pd.read_csv(OUT); done = set(zip(prev.rho, prev.depth, prev.attack,
                                            prev.delta, prev.eps))
print(f"\nD_test sweep ({len(RHOS)*len(DEPTHS)*len(ATT)*len(DELTAS)*len(EPS)} cells)",
      flush=True)
for n in DEPTHS:
    for rho in RHOS:
        eta, ach = GAINS[(rho, n)]
        c = make(rho, n, eta)
        for an, tpl in ATT.items():
            P = [apply_prompt_attack(g, tpl) for g in test_goals]
            for d_ in DELTAS:
                for e_ in EPS:
                    if (rho, n, an, d_, e_) in done: continue
                    txt, st = generate32(b, P, max_new_tokens=a.max_new,
                        batch_size=a.batch, dirs=dirs, layers=[LAYER],
                        controller=c, delta=d_, attack_eps=e_, attack_ortho=0.3,
                        gen_seed=a.seed, harm_mask=np.ones(len(P), bool))
                    rows = [{"model": a.model, "rho": rho, "depth": n,
                             "is_cantor": abs(rho - RHO_CANTOR) < 1e-9,
                             "attack": an, "delta": d_, "eps": e_,
                             "pid": TEST_PIDS[i], "safe": safe_score32(t),
                             "eta": eta, "fitted_C_rms": ach, **st}
                            for i, t in enumerate(txt)]
                    pd.DataFrame(rows).to_csv(OUT, mode="a",
                                              header=not OUT.exists(), index=False)
        ut, _ = generate32(b, safe_prompts, max_new_tokens=a.max_new,
            batch_size=a.batch, dirs=dirs, layers=[LAYER], controller=c,
            delta=0.0, attack_eps=0.0, harm_mask=np.ones(len(safe_prompts), bool))
        U = RAW / f"v331_llm_utility_{a.model}.csv"
        pd.DataFrame([{"model": a.model, "rho": rho, "depth": n, "eta": eta,
                       **utility_scores32(ut)}]).to_csv(
            U, mode="a", header=not U.exists(), index=False)
        print(f"  rho={rho:.4f} n={n} done ({time.time()-t0:.0f}s)", flush=True)

full = pd.read_csv(OUT)
print(f"\nCOMPLETE: {len(full)} rows -> {OUT}")
print(full.groupby(["depth", "rho"]).agg(
    safe=("safe", "mean"), C_rms=("C_rms", "mean")).round(4).to_string())
