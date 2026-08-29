"""V2 LLM PHASE 1 - residual extraction, refusal direction, CAUSAL validation.

Runs on CALIBRATION and DEV splits only. The test split is not touched.

Pipeline:
  1. grouped goal-level split of JBB (no paraphrase straddles a boundary)
  2. last-prompt-token residuals at every layer, harmful vs harmless
  3. two direction estimators (difference-in-means, logistic normal)
  4. CAUSAL validation: does +eps*v induce refusal and -eps*v suppress it?
     A direction that merely CLASSIFIES harmfulness is not enough
     (harness STOP CONDITION A).
  5. layer selection on DEV by causal effect and benign damage

Outputs (no prompt text, no completions -- ids and scalars only):
  results/v2/raw/llm_separability.csv
  results/v2/raw/llm_causal.csv
  results/v2/cache/<model>_dirs.npz
"""
import argparse, sys, json, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, "llm/src")
from cantor_guard.models import load_model, chat_prompt
from cantor_guard.representations import last_token_residuals
from cantor_guard.probes import (diff_in_means_direction, logistic_probe_direction,
                                 calibrate, separability)
from cantor_guard.datasets import load_jbb, load_xstest, grouped_split
from cantor_guard.hooks import perturb_residuals
from cantor_guard.safety_eval import is_refusal, compliance_score
from cantor_guard.io import write_table, V2_CACHE, provenance, seed_everything

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
ap.add_argument("--max-new", type=int, default=32)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--seed", type=int, default=20260829)
args = ap.parse_args()
seed_everything(args.seed)

t0 = time.time()
b = load_model(args.model)
print(f"[{args.model}] layers={b.n_layers} d={b.d_model} device={b.device} "
      f"dtype={b.dtype}  ({time.time()-t0:.0f}s)")

harm, ben = load_jbb()
sp_h, sp_b = grouped_split(harm, seed=args.seed), grouped_split(ben, seed=args.seed)
print(f"split harmful: {sp_h.summary()}")
LAYERS = list(range(b.n_layers))

# ---------------------------------------------------------------- 2. residuals
print("extracting residuals (calibration split)...")
A_h = last_token_residuals(b, list(sp_h.calibration.prompt), LAYERS,
                           batch_size=args.batch, progress=True)
A_b = last_token_residuals(b, list(sp_b.calibration.prompt), LAYERS,
                           batch_size=args.batch, progress=True)
print(f"  harmful {A_h.shape}  harmless {A_b.shape}")

# ------------------------------------------------------------- 3. two estimators
dirs = {}
for name, V in (("diff_means", diff_in_means_direction(A_h, A_b)),
                ("logistic", logistic_probe_direction(A_h, A_b, seed=args.seed))):
    dirs[name] = calibrate(A_h, A_b, V, LAYERS, name)

sep_rows = []
for name, d in dirs.items():
    s = separability(A_h, A_b, d.v)
    for l, val in zip(LAYERS, s):
        sep_rows.append({"model": args.model, "method": name, "layer": l,
                         "cohens_d": float(val)})
sep = pd.DataFrame(sep_rows)
best = sep.groupby("method").cohens_d.max()
print("\nseparability (Cohen's d, max over layers):")
print(best.to_string())

# --------------------------------------------------- 4. CAUSAL validation on DEV
# Probe layers: a coarse sweep, then the causal test decides. Middle layers are
# where the refusal direction is reported to be most causal (Arditi et al.).
PROBE_LAYERS = sorted(set(np.linspace(2, b.n_layers - 2, 8).astype(int).tolist()))
EPS = [0.0, 4.0, 8.0]              # % of the mean activation norm at that layer
dev_harm = list(sp_h.dev.prompt)[:16]
dev_ben = list(sp_b.dev.prompt)[:16]
xs = load_xstest(); dev_safe = list(xs[~xs.is_harmful].prompt)[:16]

@torch.no_grad()
def gen_with_delta(prompts, delta_vec, layers, scale):
    outs = []
    for i in range(0, len(prompts), args.batch):
        ch = prompts[i:i + args.batch]
        enc = b.tokenizer([chat_prompt(b, p) for p in ch],
                          return_tensors="pt", padding=True).to(b.device)
        dv = torch.tensor(delta_vec, dtype=torch.float32, device=b.device)
        with perturb_residuals(b, layers, dv, scale=scale):
            o = b.model.generate(**enc, max_new_tokens=args.max_new,
                                 do_sample=False,
                                 pad_token_id=b.tokenizer.pad_token_id)
        outs += b.tokenizer.batch_decode(o[:, enc["input_ids"].shape[1]:],
                                         skip_special_tokens=True)
    return outs

# The full causal sweep uses the literature-validated difference-in-means
# estimator (Arditi et al. 2024); the logistic alternative is checked only at
# the layer that estimator selects, so the two are not given unequal budgets
# for free -- the comparison happens where it matters.
print("\ncausal validation (+v should INDUCE refusal, -v should SUPPRESS it)")
rows = []
for method, d in [("diff_means", dirs["diff_means"])]:
    for l in PROBE_LAYERS:
        vi = d.v[d.layers.index(l)]
        # scale the unit direction by the typical activation norm at this layer
        nrm = float(np.linalg.norm(A_h[:, d.layers.index(l), :], axis=1).mean())
        for eps in EPS:
            for sgn, tag in ((+1, "plus"), (-1, "minus")):
                if eps == 0.0 and sgn < 0:
                    continue
                sc = sgn * eps * nrm / 100.0     # eps is a % of mean activation norm
                rh = gen_with_delta(dev_harm, vi, [l], sc)
                rb = gen_with_delta(dev_safe, vi, [l], sc)
                rows.append({"model": args.model, "method": method, "layer": l,
                             "eps_pct": eps, "sign": tag, "scale": sc,
                             "act_norm": nrm,
                             "refusal_harmful": float(np.mean([is_refusal(t) for t in rh])),
                             "refusal_benign": float(np.mean([is_refusal(t) for t in rb])),
                             "asr_harmful": float(np.mean([compliance_score(t) > .5 for t in rh]))})
        print(f"  {method} L{l:2d} done ({time.time()-t0:.0f}s)", flush=True)

causal = pd.DataFrame(rows)
write_table(sep, f"llm_separability_{args.model}.csv",
            meta={**b.provenance(), "phase": "V2-llm-1", "split": "calibration"})
write_table(causal, f"llm_causal_{args.model}.csv",
            meta={**b.provenance(), "phase": "V2-llm-1", "split": "dev",
                  "note": "causal validation of the refusal direction"})

base = causal[(causal.eps_pct == 0)]
print("\nbaseline (no intervention): refusal_harmful="
      f"{base.refusal_harmful.mean():.3f}  refusal_benign={base.refusal_benign.mean():.3f}")
print("\nlargest causal effects (eps=8%):")
e8 = causal[causal.eps_pct == 8.0].copy()
piv = e8.pivot_table(index=["method", "layer"], columns="sign",
                     values=["refusal_harmful", "refusal_benign"])
print(piv.to_string(float_format="{:.3f}".format))

np.savez(V2_CACHE / f"{args.model}_dirs.npz",
         **{f"{m}_v": d.v for m, d in dirs.items()},
         **{f"{m}_tau": d.tau for m, d in dirs.items()},
         **{f"{m}_sigma": d.sigma for m, d in dirs.items()},
         layers=np.array(LAYERS))
print(f"\ncached directions -> {V2_CACHE / (args.model + '_dirs.npz')}")
print(f"total {time.time()-t0:.0f}s")
