"""V3.3.3 -- System A analysis: robust safety AUC, prompt-clustered, max-T."""
import sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd

P = json.loads(pathlib.Path("configs/v3_3_3/protocol.json").read_text())
TAB = pathlib.Path("results/v3_3_3/tables")
META = json.loads((TAB / "systemA_meta.json").read_text())
df = pd.read_csv("results/v3_3_3/raw/systemA_qwen2.5-0.5b-instruct_n3.csv")
ut = pd.read_csv("results/v3_3_3/raw/systemA_utility_qwen2.5-0.5b-instruct.csv")
RHO_C, SEV, SESOI = 1/3, P["attack_severities"], P["sesoi_auc"]
NB, SEED = 20000, P["seeds"]["bootstrap"]

# budget gate: matched rho only
gains = META["gains"]
matched = [float(k) for k, v in gains.items() if v["matched"]]
excluded = {k: round(v["rel"], 4) for k, v in gains.items() if not v["matched"]}
print(f"budget-matched rho: {[round(r,4) for r in sorted(matched)]}")
print(f"EXCLUDED for budget mismatch: {excluded}")

# robust safety AUC per prompt: trapezoid over the frozen severity grid
cell = df.groupby(["rho", "pid", "eps"]).safe.mean().reset_index()
piv = cell.pivot_table(index=["rho", "pid"], columns="eps", values="safe")
sev = np.array(SEV, float)
auc = piv.apply(lambda r: np.trapz(r.values, sev) / (sev[-1] - sev[0]), axis=1)
auc = auc.rename("auc").reset_index()
auc.to_csv(TAB / "systemA_auc_per_prompt.csv", index=False)

summ = auc.groupby("rho").auc.agg(["mean", "std", "count"])
print("\n=== robust safety AUC by rho (D_final, 70 prompts) ===")
print(summ.round(5).to_string())
attain = float(summ["mean"].max() - summ["mean"].min())
print(f"spread across rho = {attain:.5f}   SESOI = {SESOI}")

W = auc.pivot_table(index="pid", columns="rho", values="auc").dropna()
rng = np.random.default_rng(SEED)
idx = rng.integers(0, len(W), size=(NB, len(W)))

def boot(a_, b_):
    d = (W[a_] - W[b_]).to_numpy()
    bs = d[idx].mean(axis=1)
    return {"mean_diff": float(d.mean()),
            "ci_lo": float(np.quantile(bs, .025)),
            "ci_hi": float(np.quantile(bs, .975)), "samples": bs}

# PRIMARY A: rho=1/3 vs the calibration-derived alternative (protocol froze
# rho_abs_star(n=3, U_EST_mid q50) = 0.4324; nearest frozen grid point is 0.44)
ALT = 0.44
print(f"\n=== PRIMARY A: rho=1/3 vs rho={ALT} (calibration-derived, "
      f"nearest grid point to 0.4324) ===")
pa = boot(RHO_C, ALT)
print(f"  d={pa['mean_diff']:+.5f}  CI95 [{pa['ci_lo']:+.5f},{pa['ci_hi']:+.5f}]")

# PRIMARY B: max-T over the pre-specified family, matched rho only
others = [r for r in sorted(matched) if abs(r - RHO_C) > 1e-9]
raw, T = {}, []
for r in others:
    s = boot(RHO_C, r); raw[r] = s
    T.append(np.abs(s["samples"] - s["mean_diff"]) / max(s["samples"].std(), 1e-12))
Tmax = np.max(np.vstack(T), axis=0)
crit = float(np.quantile(Tmax, .95))
print(f"\n=== PRIMARY B: max-T simultaneous bands (crit={crit:.3f}) ===")
rows = []
for r in others:
    s = raw[r]; sd = s["samples"].std()
    lo, hi = s["mean_diff"] - crit * sd, s["mean_diff"] + crit * sd
    sig = lo > 0 or hi < 0
    eq = lo > -SESOI and hi < SESOI
    print(f"  1/3 vs {r:.4f}: d={s['mean_diff']:+.5f} simult[{lo:+.5f},{hi:+.5f}]"
          f"  sig={sig}  equiv={eq}")
    rows.append({"rho_other": r, "mean_diff": s["mean_diff"],
                 "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
                 "simult_lo": lo, "simult_hi": hi,
                 "significant": bool(sig), "equivalent": bool(eq)})
pd.DataFrame(rows).to_csv(TAB / "systemA_comparisons.csv", index=False)

print("\n=== benign utility ===")
print(ut.round(5).to_string(index=False))
u_spread = float(ut.false_refusal.max() - ut.false_refusal.min())

gate = {"budget_matched": len(matched) >= 5,
        "endpoint_attainable": bool(0.0 < summ["mean"].min()
                                    and summ["mean"].max() < 1.0 and attain > 0),
        "cantor_vs_alt_ci": [pa["ci_lo"], pa["ci_hi"]],
        "sesoi": SESOI,
        "any_significant_simultaneous": bool(any(r["significant"] for r in rows)),
        "all_equivalent_simultaneous": bool(all(r["equivalent"] for r in rows)),
        "auc_spread": attain, "utility_spread": u_spread,
        "excluded_for_budget": excluded}
(TAB / "systemA_gate.json").write_text(json.dumps(gate, indent=2))
print("\nsystemA gate:", json.dumps(gate, indent=2))
