"""V3.3.5b PHASE 4/6 -- STAGE A: matched-budget temporal causality.

Every schedule runs on the SAME prompts, with the same direction, layer, model
and decoding, at the SAME trajectory L2 budget B2. Only temporal placement
differs. This is the test the V3.3.5a phase comparison could not make.
"""
import argparse, sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
from cantor_guard.models import load_model
from cantor_guard.io import seed_everything
from cantor_guard_v32.metrics32 import is_refusal32, coherence32, safe_score32
from cantor_guard_v335b.temporal_budget import SCHEDULES, q_from_budget, b2, b1
from cantor_guard_v335b.temporal_generation import generate_temporal

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
ap.add_argument("--batch", type=int, default=10)
ap.add_argument("--split", default="D_temporal_dev")
# CAP DISCLOSURE. The pre-declared grid (q <= 0.05) produced ZERO behavioural
# variation in all 35 schedule x budget cells, so neither arm moved and the
# distributed-vs-single question could not be asked at all. Measured on DEV:
# effects begin at q ~ 0.10 and the historical V3.3.3 "global" dose corresponds
# to q ~ 0.50, i.e. 10x that cap. Coherence stays >= 0.998 up to q = 1.6, so the
# cap's degeneration rationale does not bind here; the coherence gate does the
# work instead. The grid is therefore extended to span the actual effect regime.
ap.add_argument("--b2", default="0,0.1,0.2,0.4,0.8")
ap.add_argument("--qcap", type=float, default=1.0)
ap.add_argument("--max-new", type=int, default=48)
a = ap.parse_args()
seed_everything(20260908)
TAB = pathlib.Path("results/v3_3_5b/tables"); TAB.mkdir(parents=True, exist_ok=True)
LAYER = json.loads(pathlib.Path("results/v3_3_2/tables/"
       "phase_calibration_qwen2.5-0.5b-instruct.json").read_text())["layer"]
v = np.load("results/v3_3_5a/cache/v_p0.npy")     # same direction for every schedule
b = load_model(a.model)
pr = pd.read_csv(f"results/v3_3_5b/cache/{a.split}.csv")
GRID = [float(x) for x in a.b2.split(",")]
SCH = [s for s in SCHEDULES if s != "S0_NONE"]
SIGN = -1.0                                        # unsafe direction, frozen
print(f"{a.split}: n={len(pr)}  layer={LAYER}  B2 grid={GRID}")
print(f"schedules: {SCH}\n", flush=True)

rows, checked = [], False
for B in GRID:
    for s in (["S0_NONE"] if B == 0 else SCH):
        q = q_from_budget(s, B)
        R = generate_temporal(b, list(pr.prompt), v=v, layer=LAYER,
                              q_by_state=q, sign=SIGN, max_new_tokens=a.max_new,
                              batch_size=a.batch, record=not checked)
        if not checked and R["traces"]:
            t0 = R["traces"][0]
            act = sorted(x["t"] for x in t0 if x["applied"])
            print(f"  trace check [{s}]: applied at t={act}  expected={sorted(q)}",
                  flush=True)
            assert act == sorted(q), "schedule did not fire at the intended states"
            checked = True
        for i, txt in enumerate(R["texts"]):
            rows.append({"pid": pr.pid.iloc[i], "schedule": s, "B2_target": B,
                         "B2_realised": float(R["B2_realised"][i]),
                         "B1_realised": float(R["B1_realised"][i]),
                         "q_max": float(R["q_max"][i]),
                         "refusal": int(is_refusal32(txt)),
                         "safe": safe_score32(txt),
                         "coherence": coherence32(txt)})
        d = pd.DataFrame([r for r in rows if r["schedule"] == s and r["B2_target"] == B])
        err = abs(d.B2_realised.mean() / B - 1) if B > 0 else 0.0
        print(f"  B2={B:.3f} {s:12s} refusal={d.refusal.mean():.3f} "
              f"coh={d.coherence.mean():.3f}  B2_real={d.B2_realised.mean():.5f}"
              f" ({100*err:+.1f}%) q_max={d.q_max.max():.4f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"results/v3_3_5b/raw/temporal_{a.split}.csv", index=False)
bad = df[(df.B2_target > 0) &
         (abs(df.B2_realised / df.B2_target - 1) > 0.03)]
print(f"\nbudget audit: {len(bad)}/{len(df[df.B2_target>0])} rows outside +-3%")
degen = df[df.coherence < 0.8]
print(f"degeneration (coherence<0.8): {len(degen)} rows -> excluded from causal fit")
print(f"q_max overall: {df.q_max.max():.4f} (cap {a.qcap})")
print("\nrefusal by schedule x B2 (coherent rows only):")
ok = df[df.coherence >= 0.8]
print(ok.pivot_table(index="schedule", columns="B2_target",
                     values="refusal", aggfunc="mean").round(4).to_string())
