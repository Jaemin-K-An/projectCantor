"""V3.3.5b -- where does the effect regime actually start, in q units?

The pre-declared budget grid (q <= 0.05) produced ZERO behavioural variation in
every schedule, so Stage A as specified cannot discriminate distributed from
single-state control: neither arm moves. Before concluding anything, locate the
q at which effects begin, and convert the historical V3.3.3 "global" dose into
the same q units so the two regimes can be compared on one scale.
"""
import sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
from cantor_guard.models import load_model
from cantor_guard.io import seed_everything
from cantor_guard_v32.metrics32 import is_refusal32, coherence32
from cantor_guard_v335a.p0_residual import last_valid_prompt_residuals
from cantor_guard_v335b.temporal_budget import q_from_budget
from cantor_guard_v335b.temporal_generation import generate_temporal

seed_everything(20260908)
TAB = pathlib.Path("results/v3_3_5b/tables")
LAYER = json.loads(pathlib.Path("results/v3_3_2/tables/"
       "phase_calibration_qwen2.5-0.5b-instruct.json").read_text())["layer"]
v = np.load("results/v3_3_5a/cache/v_p0.npy")
b = load_model("qwen2.5-0.5b-instruct")
pr = pd.read_csv("results/v3_3_5b/cache/D_temporal_dev.csv")

H = last_valid_prompt_residuals(b, list(pr.prompt), LAYER, batch_size=8)
hn = float(np.linalg.norm(H, axis=-1).mean())
print(f"mean ||h_P0||_2 at layer {LAYER} = {hn:.3f}")

# historical V3.3.3 global dose, expressed as a relative magnitude
hist = json.loads(pathlib.Path("results/v3_3_3/tables/"
                               "behavioral_boundary.json").read_text())
sig = hist["sigma_G1"]
print("\nhistorical V3.3.3 GLOBAL dose, converted to q = |dh|/||h||:")
for ds in (-3, -6, -10):
    lam = ds * sig
    print(f"  {ds:+3d} sigma -> |lambda| = {abs(lam):6.3f} -> q ~ {abs(lam)/hn:.4f}"
          f"   ({abs(lam)/hn/0.05:.1f}x the 0.05 cap)")

# where do effects begin? single-state P0, escalating q, DEV only
print("\nDEV escalation, S1_P0_ONLY (exits the pre-declared q<=0.05 regime):",
      flush=True)
rows = []
for q in (0.05, 0.10, 0.20, 0.40, 0.80, 1.60):
    R = generate_temporal(b, list(pr.prompt), v=v, layer=LAYER,
                          q_by_state={0: q}, sign=-1.0, max_new_tokens=48,
                          batch_size=10)
    ref = float(np.mean([is_refusal32(t) for t in R["texts"]]))
    coh = float(np.mean([coherence32(t) for t in R["texts"]]))
    rows.append({"q": q, "refusal": ref, "coherence": coh})
    print(f"  q={q:.2f}  refusal={ref:.3f}  coherence={coh:.3f}"
          + ("   <- DEGENERATE" if coh < 0.8 else ""), flush=True)
d = pd.DataFrame(rows)
d.to_csv(TAB/"regime_gap.csv", index=False)
eff = d[(d.refusal != d.refusal.iloc[0]) & (d.coherence >= 0.8)]
first = float(eff.q.min()) if len(eff) else None
print(f"\nfirst q with a behavioural change AND coherence>=0.8: {first}")
print(f"pre-declared cap: 0.05")
json.dump({"mean_h_norm": hn, "cap": 0.05,
           "first_effective_q_nondegenerate": first,
           "historical_global_q_estimate": {f"{ds}sigma": abs(ds*sig)/hn
                                            for ds in (-3, -6, -10)},
           "escalation": rows},
          open(TAB/"regime_gap.json", "w"), indent=2)
