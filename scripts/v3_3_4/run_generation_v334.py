"""V3.3.4 PHASE 18/19 -- HARD CantorGuardedPolicy in real generation.

Cantor runs as an actual inference-time controller, centred on tau_beh, with
phase-aware calibration. The budget gate is computed on D_FINAL, not D_budget
(the V3.3.3 defect that let rho=0.40 and 0.44 into the confirmatory comparison
while actually missing the target by -5.6% and -6.1%).
"""
import argparse, sys, json, time, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd, torch
from cantor_guard.models import load_model
from cantor_guard.probes import RefusalDirections
from cantor_guard.datasets import load_xstest
from cantor_guard.attacks import TEST_ATTACKS, apply_prompt_attack
from cantor_guard.io import seed_everything
from cantor_guard_v32.metrics32 import safe_score32, utility_scores32
from cantor_guard_v332.phase_generation import generate332
from cantor_guard_v334.guarded_policy import CantorGuardedPolicy

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
ap.add_argument("--batch", type=int, default=10)
ap.add_argument("--max-new", type=int, default=48)
ap.add_argument("--n-safe", type=int, default=50)
a = ap.parse_args()
P = json.loads(pathlib.Path("configs/v3_3_4/protocol.json").read_text())
seed_everything(P["seeds"]["generation"])
TAB = pathlib.Path("results/v3_3_4/tables"); RAW = pathlib.Path("results/v3_3_4/raw")
B = json.loads(pathlib.Path("results/v3_3_3/tables/behavioral_boundary.json").read_text())
V = json.loads(pathlib.Path("results/v3_3_2/tables/"
                            "phase_calibration_qwen2.5-0.5b-instruct.json").read_text())
TAU_BEH, TAU_MID = B["tau_beh"], V["calibrations"]["G1"]["tau"]
SIG, GAM, LAYER = B["sigma_G1"], V["gamma"], V["layer"]
z = np.load(f"results/v3_3_2/cache/{a.model}_v332_phase.npz")
vv = z["v"].reshape(1, -1).astype(np.float32)
cP = V["calibrations"]["P"]
dirs_P = RefusalDirections([LAYER], vv, np.array([cP["tau"]], np.float32),
                           np.array([cP["sigma"]], np.float32), "P")

bud = pd.read_csv("results/v3_3_4/cache/D_budget_new.csv")
fin = pd.read_csv("results/v3_3_4/cache/D_final_334.csv")
xs = load_xstest(); benign = list(xs[~xs.is_harmful].prompt)[:a.n_safe]
ATT = TEST_ATTACKS[P["attacks"][0]]
SEV = P["attack_severities"]
CONFIGS = [(r, "tau_beh") for r in P["rho_grid_generation"]] + [(1/3, "tau_mid")]
print(f"configs={len(CONFIGS)}  D_budget={len(bud)}  D_final={len(fin)}")

b = load_model(a.model)

def make(rho, centre, eta):
    tau = TAU_BEH if centre == "tau_beh" else TAU_MID
    C = CantorGuardedPolicy(rho, P["depth_primary"], tau_beh=tau, sigma=SIG,
                            gamma=GAM, max_q=P["q_cap"], eta=eta)
    dirs_G = RefusalDirections([LAYER], vv, np.array([tau], np.float32),
                               np.array([SIG], np.float32), "G")
    return C, dirs_G

def run(prompts, C, dirs_G, eps):
    return generate332(b, prompts, dirs_prompt=dirs_P, dirs_generation=dirs_G,
                       layer=LAYER, controller=C, mode="phase_aware",
                       max_new_tokens=a.max_new, batch_size=a.batch,
                       attack_eps=eps, attack_ortho=0.3,
                       attack_seed=P["seeds"]["generation"],
                       harm_mask=np.ones(len(prompts), bool))

Pb = [apply_prompt_attack(g, ATT) for g in bud.prompt]
print("\nfitting eta on D_budget_new", flush=True)
GAINS = {}
for rho, centre in CONFIGS:
    def crms(eta):
        C, dG = make(rho, centre, eta)
        return float(np.mean([run(Pb, C, dG, e)[1]["C_rms"] for e in (0.0, SEV[-1])]))
    base = crms(1.0)
    if base <= 1e-12:
        GAINS[(rho, centre)] = {"eta": 0.0, "fit": 0.0, "ok": False}; continue
    lo, hi = 0.0, P["target_C_rms"] / base
    ach = crms(hi); it, grow = 2, 0
    while ach < P["target_C_rms"] * .98 and grow < 8:
        lo, hi = hi, hi * 2; ach = crms(hi); it += 1; grow += 1
    eta = hi
    while abs(ach - P["target_C_rms"]) / P["target_C_rms"] > .02 and it < 12:
        mid = .5 * (lo + hi); am = crms(mid); it += 1
        if am < P["target_C_rms"]: lo = mid
        else: hi = mid
        eta, ach = mid, am
    GAINS[(rho, centre)] = {"eta": eta, "fit": ach,
                            "ok": abs(ach/P["target_C_rms"]-1) <= .03}
    print(f"  rho={rho:.4f} {centre:8s} eta={eta:9.4f} C_rms={ach:.5f} "
          f"({100*(ach/P['target_C_rms']-1):+.1f}%)", flush=True)

print(f"\nD_final_334 generation ({len(CONFIGS)*len(SEV)} cells)", flush=True)
Pf = [apply_prompt_attack(g, ATT) for g in fin.prompt]
rows, t0 = [], time.time()
for rho, centre in CONFIGS:
    g = GAINS[(rho, centre)]
    C, dG = make(rho, centre, g["eta"])
    for eps in SEV:
        txt, st = run(Pf, C, dG, eps)
        rows += [{"rho": rho, "centre": centre, "eps": eps, "pid": fin.pid.iloc[i],
                  "safe": safe_score32(t), "eta": g["eta"], **st}
                 for i, t in enumerate(txt)]
    print(f"  rho={rho:.4f} {centre} done ({time.time()-t0:.0f}s)", flush=True)
df = pd.DataFrame(rows); df.to_csv(RAW/"generation_v334.csv", index=False)

urows = []
for rho, centre in CONFIGS:
    C, dG = make(rho, centre, GAINS[(rho, centre)]["eta"])
    ut, _ = run(benign, C, dG, 0.0)
    urows.append({"rho": rho, "centre": centre, **utility_scores32(ut)})
pd.DataFrame(urows).to_csv(RAW/"generation_utility_v334.csv", index=False)

fb = df.groupby(["rho", "centre"]).C_rms.mean()
print("\n=== D_FINAL actual budget gate (the V3.3.3 fix) ===")
meta = {}
for (rho, centre), c in fb.items():
    ok = abs(c/P["target_C_rms"] - 1) <= P["budget_tolerance_final"]
    meta[f"{rho}|{centre}"] = {"eta": GAINS[(rho, centre)]["eta"],
                               "fit_C_rms": GAINS[(rho, centre)]["fit"],
                               "final_C_rms": float(c),
                               "matched_final": bool(ok)}
    print(f"  rho={rho:.4f} {centre:8s} final C_rms={c:.5f} "
          f"({100*(c/P['target_C_rms']-1):+5.1f}%)  matched={ok}")
json.dump(meta, open(TAB/"generation_meta.json", "w"), indent=2)
print("\nmean safety:"); print(df.groupby(["rho","centre"]).safe.mean().round(4).to_string())
print("\nutility:"); print(pd.DataFrame(urows).round(4).to_string(index=False))
