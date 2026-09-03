"""V3.3.5b PHASE 7/8 -- matched-budget temporal inference, prompt-paired."""
import sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd

P = json.loads(pathlib.Path("configs/v3_3_5b/protocol_stageA.json").read_text())
TAB = pathlib.Path("results/v3_3_5b/tables")
SESOI, NB = P["sesoi_dP"], 20000
df = pd.read_csv("results/v3_3_5b/raw/temporal_D_temporal_confirm.csv")
df = df[df.coherence >= P["coherence_gate"]]
print(f"confirm: {df.pid.nunique()} prompts, {len(df)} rows after coherence gate")
bad = df[(df.B2_target > 0) & (abs(df.B2_realised/df.B2_target - 1) > 0.03)]
print(f"budget audit: {len(bad)} rows outside +-3%  -> "
      f"{'PASS' if len(bad)==0 else 'BUDGET_MISMATCH'}")

tab = df[df.B2_target > 0].pivot_table(index="schedule", columns="B2_target",
                                       values="refusal", aggfunc="mean")
base = float(df[df.B2_target == 0].refusal.mean())
print(f"\nno-intervention refusal = {base:.4f}")
print("refusal by schedule x matched B2:")
print(tab.round(4).to_string())
print("\ncausal effect |dP| from baseline (larger = more control):")
print((base - tab).round(4).to_string())

rng = np.random.default_rng(P["seeds"]["bootstrap"])
rows, samples = [], {}
for B in sorted(df[df.B2_target > 0].B2_target.unique()):
    sub = df[df.B2_target == B]
    W = sub.pivot_table(index="pid", columns="schedule", values="refusal").dropna()
    idx = rng.integers(0, len(W), size=(NB, len(W)))
    for a_, b_ in P["primary_contrasts"]:
        if a_ not in W or b_ not in W: continue
        # effect = reduction in refusal; distributed-minus-single on that scale
        d = ((base - W[a_]) - (base - W[b_])).to_numpy()
        bs = d[idx].mean(axis=1)
        rows.append({"B2": B, "contrast": f"{a_} - {b_}",
                     "mean_diff": float(d.mean()),
                     "ci_lo": float(np.quantile(bs, .025)),
                     "ci_hi": float(np.quantile(bs, .975)),
                     "sd": float(bs.std())})
        samples[(B, a_, b_)] = bs - d.mean()
T = np.max(np.vstack([np.abs(s)/max(s.std(), 1e-12) for s in samples.values()]), 0)
crit = float(np.quantile(T, .95))
out = pd.DataFrame(rows)
out["simult_lo"] = out.mean_diff - crit*out.sd
out["simult_hi"] = out.mean_diff + crit*out.sd
out["significant"] = (out.simult_lo > 0) | (out.simult_hi < 0)
out["distributed_wins"] = out.simult_lo > SESOI
out.to_csv(TAB/"temporal_contrasts.csv", index=False)
print(f"\n=== pre-declared contrasts, max-T simultaneous (crit={crit:.3f}) ===")
print(out[["B2","contrast","mean_diff","simult_lo","simult_hi",
           "significant","distributed_wins"]].round(4).to_string(index=False))

any_dist = bool(out.distributed_wins.any())
single_wins = bool((out.simult_hi < -SESOI).any())
verdict = ("TD1_DISTRIBUTED_SUPPORTED" if any_dist else
           ("TD3_SINGLE_STATE_BETTER" if single_wins else
            "TD2_ACCUMULATION_EXPLAINS_GLOBAL"))
print(f"\n*** {verdict} ***")
json.dump({"verdict": verdict, "baseline_refusal": base,
           "n_prompts": int(df.pid.nunique()),
           "budget_mismatch_rows": int(len(bad)),
           "maxT_crit": crit, "sesoi": SESOI,
           "any_distributed_wins": any_dist, "any_single_wins": single_wins,
           "contrasts": rows,
           "effect_table": (base - tab).round(5).to_dict()},
          open(TAB/"temporal_verdict.json", "w"), indent=2)
