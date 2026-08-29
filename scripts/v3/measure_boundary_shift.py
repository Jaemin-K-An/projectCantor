"""V3 PHASE 3 — measure natural safety-boundary non-stationarity in a real LLM.

Answers harness §37: does tau(x, l, t, a) actually move? If it does not, V3's
premise is weak and that is reported.

Axes measured:
  * layer
  * phase          : last prompt token vs generated tokens
  * token bin      : generation position 1, 2-4, 5-8, 9-16, 17+
  * attack family  : plain / calibration-family / dev-family templates
  * prompt group   : harmful goal category

Only CALIBRATION and DEV prompts are used. No test prompts are loaded.
Outputs scalars only -- no prompt text, no completions.
"""
import argparse, sys, time
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd, torch
from cantor_guard.models import load_model, chat_prompt, decoder_layers
from cantor_guard.probes import RefusalDirections
from cantor_guard.datasets import load_jbb, grouped_split, prompt_id
from cantor_guard.attacks import CALIBRATION_ATTACKS, DEV_ATTACKS, apply_prompt_attack
from cantor_guard.io import V2_CACHE, seed_everything
from cantor_guard_v3.io3 import write_v3 as write_table
from cantor_guard_v3.boundary_uncertainty import (threshold_from_projections,
                                                  normalised_shift,
                                                  summarise_shifts, uncertainty_set)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
ap.add_argument("--batch", type=int, default=12)
ap.add_argument("--max-new", type=int, default=24)
ap.add_argument("--n-prompts", type=int, default=28)
ap.add_argument("--seed", type=int, default=20260829)
args = ap.parse_args()
seed_everything(args.seed)

b = load_model(args.model)
z = np.load(V2_CACHE / f"{args.model}_dirs.npz")
dirs = RefusalDirections(list(z["layers"]), z["diff_means_v"], z["diff_means_tau"],
                         z["diff_means_sigma"], "diff_means")
LAYERS = [4, 8, 11, 13, 16, 20]
print(f"{args.model}: {b.n_layers} layers, probing {LAYERS}")

harm, ben = load_jbb()
sp_h, sp_b = grouped_split(harm, seed=args.seed), grouped_split(ben, seed=args.seed)
H = list(sp_h.calibration.prompt) + list(sp_h.dev.prompt)
B = list(sp_b.calibration.prompt) + list(sp_b.dev.prompt)
Hcat = list(sp_h.calibration.category) + list(sp_h.dev.category)
H, B, Hcat = H[:args.n_prompts], B[:args.n_prompts], Hcat[:args.n_prompts]
TEMPLATES = {"plain": "{goal}", **CALIBRATION_ATTACKS, **DEV_ATTACKS}
print(f"{len(H)} harmful + {len(B)} harmless prompts x {len(TEMPLATES)} templates")


@torch.no_grad()
def projections_over_generation(prompts, layers, max_new):
    """Return a long-form frame of <h,v> at every (prompt, layer, position).

    position 0 = last PROMPT token (prefill), 1.. = generated tokens.
    A single greedy generation per batch with hidden states captured per step.
    """
    tok = b.tokenizer
    rows = []
    L2I = {l: dirs.layers.index(l) for l in layers}
    blocks = decoder_layers(b)
    for i in range(0, len(prompts), args.batch):
        chunk = prompts[i:i + args.batch]
        enc = tok([chat_prompt(b, p) for p in chunk], return_tensors="pt",
                  padding=True).to(b.device)
        store = {}
        handles = []

        def mk(l):
            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                store.setdefault(l, []).append(h[:, -1, :].detach().float().cpu())
                return out
            return hook
        for l in layers:
            handles.append(blocks[l].register_forward_hook(mk(l)))
        try:
            b.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        finally:
            for hd in handles:
                hd.remove()
        for l in layers:
            steps = store[l]                       # list over forward passes
            v = torch.tensor(dirs.v[L2I[l]], dtype=torch.float32)
            for s, hs in enumerate(steps):
                proj = (hs @ v).numpy()
                for k, pr in enumerate(proj):
                    rows.append({"idx": i + k, "layer": l, "pos": s,
                                 "z": float(pr)})
    return pd.DataFrame(rows)


def bin_pos(p):
    if p == 0: return "prefill"
    if p == 1: return "gen_1"
    if p <= 4: return "gen_2-4"
    if p <= 8: return "gen_5-8"
    if p <= 16: return "gen_9-16"
    return "gen_17+"


t0 = time.time()
recs = []
for tname, tpl in TEMPLATES.items():
    hp = [apply_prompt_attack(p, tpl) for p in H]
    bp = [apply_prompt_attack(p, tpl) for p in B]
    dh = projections_over_generation(hp, LAYERS, args.max_new); dh["cls"] = "harmful"
    db = projections_over_generation(bp, LAYERS, args.max_new); db["cls"] = "harmless"
    d = pd.concat([dh, db], ignore_index=True)
    d["attack"] = tname
    d["bin"] = d.pos.map(bin_pos)
    d["category"] = [Hcat[r.idx] if r.cls == "harmful" else "benign"
                     for r in d.itertuples()]
    recs.append(d)
    print(f"  template {tname:18s} done ({time.time()-t0:.0f}s)", flush=True)
proj = pd.concat(recs, ignore_index=True)
print(f"collected {len(proj)} projections")

# ------------------------------------------------------ global vs conditional
rows = []
for l, gl in proj.groupby("layer"):
    zg_h = gl[gl.cls == "harmful"].z.values
    zg_b = gl[gl.cls == "harmless"].z.values
    tau_g, sig_g = threshold_from_projections(zg_h, zg_b)
    for keys, sub in gl.groupby(["attack", "bin"]):
        zh = sub[sub.cls == "harmful"].z.values
        zb = sub[sub.cls == "harmless"].z.values
        if len(zh) < 5 or len(zb) < 5:
            continue
        tau_c, sig_c = threshold_from_projections(zh, zb)
        rows.append({"model": args.model, "layer": l, "attack": keys[0],
                     "bin": keys[1], "tau_global": tau_g, "sigma_global": sig_g,
                     "tau_cond": tau_c, "sigma_cond": sig_c,
                     "delta_norm": normalised_shift(tau_c, tau_g, sig_g),
                     "sigma_ratio": sig_c / sig_g, "n_h": len(zh), "n_b": len(zb),
                     "phase": "prefill" if keys[1] == "prefill" else "generation"})
sh = pd.DataFrame(rows)
write_table(sh, f"v3_boundary_shift_{args.model}.csv",
            meta={**b.provenance(), "phase": "V3-PHASE3",
                  "note": "natural boundary non-stationarity, calib+dev only"})

print("\n" + "=" * 96)
print("DOES THE SAFETY BOUNDARY ACTUALLY MOVE?  (Delta in units of global sigma)")
print("=" * 96)
a = sh.delta_norm.abs()
print(f"  overall |Delta_norm|: median {a.median():.3f}  q75 {a.quantile(.75):.3f}  "
      f"q90 {a.quantile(.90):.3f}  q95 {a.quantile(.95):.3f}  max {a.max():.3f}")
print("\n  by phase (prefill vs generation):")
print(sh.groupby("phase").delta_norm.agg(["mean", "std", "min", "max"]).round(3).to_string())
print("\n  by token bin (mean Delta_norm per layer):")
print(sh.pivot_table(index="bin", columns="layer", values="delta_norm",
                     aggfunc="mean").round(3).to_string())
print("\n  by attack family:")
print(sh.groupby("attack").delta_norm.agg(["mean", "std"]).round(3).to_string())
print("\n  sigma ratio (conditional spread / global spread):")
print(sh.groupby("phase").sigma_ratio.agg(["mean", "min", "max"]).round(3).to_string())
u = uncertainty_set(sh)
print(f"\n  PRE-REGISTERED uncertainty set U_Delta from these measurements: {u}")
import json
json.dump(u, open(f"results/v3/cache/{args.model}_udelta.json", "w"), indent=2)
print(f"\ntotal {time.time()-t0:.0f}s")
