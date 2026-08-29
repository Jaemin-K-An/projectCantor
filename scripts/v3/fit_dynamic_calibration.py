"""V3 PHASE 4 — fixed vs dynamic calibration, and the §38 question:
is estimating the boundary better more valuable than controller geometry?

Fit C0/C1/C2 on CALIBRATION prompts, measure residual boundary error on DEV.
"""
import sys, time, json
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd, torch
from cantor_guard.models import load_model, chat_prompt, decoder_layers
from cantor_guard.probes import RefusalDirections
from cantor_guard.datasets import load_jbb, grouped_split
from cantor_guard.attacks import CALIBRATION_ATTACKS, DEV_ATTACKS, apply_prompt_attack
from cantor_guard.io import V2_CACHE, seed_everything
from cantor_guard_v3.io3 import write_v3 as write_table
from cantor_guard_v3.calibration import fit_calibrator, calibration_error, bin_of

MK = "qwen2.5-0.5b-instruct"
seed_everything(20260829)
b = load_model(MK)
z = np.load(V2_CACHE / f"{MK}_dirs.npz")
dirs = RefusalDirections(list(z["layers"]), z["diff_means_v"], z["diff_means_tau"],
                         z["diff_means_sigma"], "diff_means")
LAYERS = [8, 11, 13, 16]
harm, ben = load_jbb()
sp_h, sp_b = grouped_split(harm, seed=20260829), grouped_split(ben, seed=20260829)


@torch.no_grad()
def collect(prompts, cls, tpl, max_new=20, batch=12):
    tok = b.tokenizer; blocks = decoder_layers(b); rows = []
    L2I = {l: dirs.layers.index(l) for l in LAYERS}
    for i in range(0, len(prompts), batch):
        ch = [apply_prompt_attack(p, tpl) for p in prompts[i:i+batch]]
        enc = tok([chat_prompt(b, p) for p in ch], return_tensors="pt",
                  padding=True).to(b.device)
        store = {}; handles = []
        def mk(l):
            def hook(_m,_i,out):
                h = out[0] if isinstance(out, tuple) else out
                store.setdefault(l, []).append(h[:,-1,:].detach().float().cpu())
                return out
            return hook
        for l in LAYERS: handles.append(blocks[l].register_forward_hook(mk(l)))
        try:
            b.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        finally:
            for hd in handles: hd.remove()
        for l in LAYERS:
            v = torch.tensor(dirs.v[L2I[l]], dtype=torch.float32)
            for s, hs in enumerate(store[l]):
                for zz in (hs @ v).numpy():
                    rows.append({"layer": l, "pos": s, "z": float(zz), "cls": cls})
    return pd.DataFrame(rows)


t0 = time.time()
cal_frames, dev_frames = [], []
for tname, tpl in CALIBRATION_ATTACKS.items():
    cal_frames.append(collect(list(sp_h.calibration.prompt)[:24], "harmful", tpl))
    cal_frames.append(collect(list(sp_b.calibration.prompt)[:24], "harmless", tpl))
for tname, tpl in DEV_ATTACKS.items():
    dev_frames.append(collect(list(sp_h.dev.prompt)[:24], "harmful", tpl))
    dev_frames.append(collect(list(sp_b.dev.prompt)[:24], "harmless", tpl))
CAL = pd.concat(cal_frames, ignore_index=True)
DEV = pd.concat(dev_frames, ignore_index=True)
print(f"calibration {len(CAL)} / dev {len(DEV)} projections  ({time.time()-t0:.0f}s)")

rows = []
for method in ("C0_fixed", "C1_phase", "C2_token_bin"):
    cal = fit_calibrator(CAL, method)
    e_in = calibration_error(CAL, cal)
    e_out = calibration_error(DEV, cal)
    rows.append({"model": MK, "method": method, "err_calibration": e_in,
                 "err_dev": e_out, "n_cond": len(cal.tau_cond)})
    print(f"  {method:14s} residual boundary error: "
          f"calib {e_in:.3f} sigma | DEV {e_out:.3f} sigma  ({len(cal.tau_cond)} conditions)")
df = pd.DataFrame(rows)
write_table(df, f"v3_calibration_{MK}.csv",
            meta={**b.provenance(), "phase": "V3-PHASE4"})
best = df.sort_values("err_dev").iloc[0]
red = 100*(df[df.method=="C0_fixed"].err_dev.iloc[0] - best.err_dev)/df[df.method=="C0_fixed"].err_dev.iloc[0]
print(f"\n  best method on DEV: {best.method}  ({red:.1f}% reduction in boundary error vs V2's C0)")
json.dump({"best_method": best.method, "err_dev": float(best.err_dev),
           "reduction_pct": float(red)},
          open(f"results/v3/cache/{MK}_calibration.json", "w"), indent=2)
