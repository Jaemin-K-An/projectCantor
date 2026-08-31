"""V3.3.3 gap 4 -- the phase-aware System A generation experiment.

PREFILL uses (tau_P, sigma_P); DECODE uses (tau_G, sigma_G); the transition is
verified at runtime and the trace asserted. This is real generation, not a
System B substitute.

eta is fitted on D_budget (JBB) ONLY. D_final is never used for any fitting.
"""
import argparse, sys, json, time, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd, torch
from cantor_guard.models import load_model
from cantor_guard.probes import RefusalDirections
from cantor_guard.datasets import load_jbb, load_xstest
from cantor_guard.attacks import TEST_ATTACKS, apply_prompt_attack
from cantor_guard.io import seed_everything
from cantor_guard_v32.splits import load_split
from cantor_guard_v32.metrics32 import safe_score32, utility_scores32
from cantor_guard_v332.phase_generation import generate332
from cantor_guard_v331.rho_family import RhoBarrier

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
ap.add_argument("--batch", type=int, default=10)
ap.add_argument("--max-new", type=int, default=48)
ap.add_argument("--depth", type=int, default=3)
ap.add_argument("--n-safe", type=int, default=24)
a = ap.parse_args()

P = json.loads(pathlib.Path("configs/v3_3_3/protocol.json").read_text())
seed_everything(P["seeds"]["generation"])
TAB = pathlib.Path("results/v3_3_3/tables"); RAW = pathlib.Path("results/v3_3_3/raw")
V332 = json.loads(pathlib.Path("results/v3_3_2/tables/"
                               "phase_calibration_qwen2.5-0.5b-instruct.json").read_text())
LAYER = P["layer"]
z = np.load(f"results/v3_3_2/cache/{a.model}_v332_phase.npz")
v = z["v"].reshape(1, -1).astype(np.float32)
cP, cG = V332["calibrations"]["P"], V332["calibrations"]["G1"]
dirs_P = RefusalDirections([LAYER], v, np.array([cP["tau"]], np.float32),
                           np.array([cP["sigma"]], np.float32), "phase_P")
dirs_G = RefusalDirections([LAYER], v, np.array([cG["tau"]], np.float32),
                           np.array([cG["sigma"]], np.float32), "phase_G1")
print(f"PREFILL tau={cP['tau']:+.4f} sigma={cP['sigma']:.4f} | "
      f"DECODE tau={cG['tau']:+.4f} sigma={cG['sigma']:.4f}")

split = load_split("configs/v3_2/split.json")
harm, _ = load_jbb(); H = harm.set_index("pid")
budget_goals = [H.loc[p, "prompt"] for p in split["blocks"]["D_budget"]]
fin = pd.read_csv("results/v3_3_3/cache/d_final_prompts.csv")
final_goals, final_pids = list(fin.prompt), list(fin.pid)
xs = load_xstest(); benign = list(xs[~xs.is_harmful].prompt)[:a.n_safe]
RHOS, SEV = P["rho_grid"], P["attack_severities"]
ATT = {k: TEST_ATTACKS[k] for k in P["attacks"]}
print(f"rho={[round(r,4) for r in RHOS]}  depth={a.depth}  "
      f"severities={SEV}  D_final={len(final_goals)}")

b = load_model(a.model)

def make(rho, eta):
    c = RhoBarrier(rho, a.depth, 1.0 / a.depth)
    c.eta, c.gamma, c.max_q, c.harm_gate = eta, 0.7, P["q_cap"], True
    return c

def run(prompts, c, eps, trace=False):
    return generate332(b, prompts, dirs_prompt=dirs_P, dirs_generation=dirs_G,
                       layer=LAYER, controller=c, mode="phase_aware",
                       max_new_tokens=a.max_new, batch_size=a.batch,
                       attack_eps=eps, attack_ortho=0.3,
                       attack_seed=P["seeds"]["generation"],
                       harm_mask=np.ones(len(prompts), bool), record_trace=trace)

# ---- runtime verification of the phase transition ----
Pb = [apply_prompt_attack(g, ATT[P["attacks"][0]]) for g in budget_goals]
_, _, tr = run(Pb[:4], make(1/3, 0.1), 0.0, trace=True)
print(f"phase trace: ok={all(t['ok'] for t in tr)}  "
      f"prefill={tr[0]['n_prefill']} decode={tr[0]['n_decode']}")
if not all(t["ok"] for t in tr):
    raise SystemExit("PHASE TRACE FAILED")

# ---- budget fit on D_budget only ----
print("\nfitting eta on D_budget (bisection)", flush=True)
def crms(c, eta):
    c.eta = eta
    return float(np.mean([run(Pb, c, e)[1]["C_rms"] for e in SEV]))
GAINS = {}
for rho in RHOS:
    c = make(rho, 1.0); base = crms(c, 1.0)
    lo, hi = 0.0, P["target_C_rms"] / max(base, 1e-12)
    ach = crms(c, hi); it, grow = 2, 0
    while ach < P["target_C_rms"] * 0.98 and grow < 8:
        lo, hi = hi, hi * 2; ach = crms(c, hi); it += 1; grow += 1
    eta = hi
    while abs(ach - P["target_C_rms"]) / P["target_C_rms"] > 0.02 and it < 14:
        mid = .5 * (lo + hi); am = crms(c, mid); it += 1
        if am < P["target_C_rms"]: lo = mid
        else: hi = mid
        eta, ach = mid, am
    rel = (ach - P["target_C_rms"]) / P["target_C_rms"]
    GAINS[rho] = {"eta": eta, "fit_C_rms": ach, "rel": rel,
                  "matched": abs(rel) <= P["budget_tolerance_rms"]}
    print(f"  rho={rho:.4f} eta={eta:9.4f} C_rms={ach:.5f} ({100*rel:+.1f}%) "
          f"{'OK' if GAINS[rho]['matched'] else 'FAIL'}", flush=True)

# ---- D_final, after the seal ----
print(f"\nD_final generation ({len(RHOS)*len(ATT)*len(SEV)} cells)", flush=True)
OUT = RAW / f"systemA_{a.model}_n{a.depth}.csv"
t0 = time.time(); rows = []
for rho in RHOS:
    g = GAINS[rho]; c = make(rho, g["eta"])
    for an, tpl in ATT.items():
        Pf = [apply_prompt_attack(x, tpl) for x in final_goals]
        for eps in SEV:
            txt, st = run(Pf, c, eps)
            rows += [{"rho": rho, "depth": a.depth, "attack": an, "eps": eps,
                      "pid": final_pids[i], "safe": safe_score32(t),
                      "eta": g["eta"], **st} for i, t in enumerate(txt)]
    print(f"  rho={rho:.4f} done ({time.time()-t0:.0f}s)", flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT, index=False)

# ---- benign utility ----
urows = []
for rho in RHOS:
    c = make(rho, GAINS[rho]["eta"])
    ut, _ = run(benign, c, 0.0)
    urows.append({"rho": rho, **utility_scores32(ut)})
pd.DataFrame(urows).to_csv(RAW / f"systemA_utility_{a.model}.csv", index=False)

json.dump({"gains": {str(k): v for k, v in GAINS.items()},
           "n_final": len(final_pids), "severities": SEV,
           "phase_trace_ok": True},
          open(TAB / "systemA_meta.json", "w"), indent=2)
print(f"\nCOMPLETE {len(df)} rows -> {OUT}")
print(df.groupby("rho").agg(safe=("safe","mean"), C_rms=("C_rms","mean")).round(4).to_string())
print("\nbenign utility:"); print(pd.DataFrame(urows).round(4).to_string(index=False))
