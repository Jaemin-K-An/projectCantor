"""V3.2 -- does float16 preserve the refusal projection on Model B?

The 8 GB machine cannot hold TinyLlama-1.1B in float32 (4.4 GB) alongside
eager attention without thrashing swap. Halving the dtype is only acceptable
if it does not disturb the quantity the whole method depends on: the projection
of the residual stream onto the refusal direction, which is a small difference
of large numbers.

This measures that rather than assuming it. If fp16 margins do not track fp32
margins closely, fp16 is rejected and the design changes instead.
"""
import sys, json, time, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, torch
from cantor_guard.models import load_model
from cantor_guard.probes import diff_in_means_direction, calibrate, separability
from cantor_guard.representations import last_token_residuals
from cantor_guard.datasets import load_jbb
from cantor_guard_v32.splits import load_split

split = load_split("configs/v3_2/split.json")
harm, ben = load_jbb(); harm = harm.set_index("pid")
d_dir = [harm.loc[p, "prompt"] for p in split["blocks"]["D_direction"]][:12]
benign = list(ben.prompt)[:12]
LAYER_FRAC = 0.55

out = {}
acts = {}
for dt in ("float16", "float32"):
    t0 = time.time()
    b = load_model("tinyllama-1.1b-chat", dtype=getattr(torch, dt))
    L = [int(b.n_layers * LAYER_FRAC)]
    ah = last_token_residuals(b, d_dir, L, batch_size=2)
    al = last_token_residuals(b, benign, L, batch_size=2)
    acts[dt] = (np.asarray(ah, np.float64), np.asarray(al, np.float64))
    V = diff_in_means_direction(ah, al)
    rd = calibrate(ah, al, V, L, "diff_means")
    sep = float(np.atleast_1d(separability(ah, al, V))[0])
    out[dt] = {"layer": L[0], "sep": round(sep, 4), "tau": float(rd.tau[0]),
               "sigma": float(rd.sigma[0]),
               "load_and_probe_s": round(time.time() - t0, 1)}
    print(f"{dt}: sep={sep:.4f} tau={out[dt]['tau']:.4f} "
          f"sigma={out[dt]['sigma']:.4f}  ({out[dt]['load_and_probe_s']}s)", flush=True)
    del b, ah, al
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# Compare the margins fp16 vs fp32 on identical prompts, in sigma units.
h16, l16 = acts["float16"]; h32, l32 = acts["float32"]
V32 = diff_in_means_direction(h32, l32)
rd32 = calibrate(h32, l32, V32, [0], "diff_means")
def margins(a):
    p = np.squeeze(a) @ np.squeeze(V32)
    return (p - float(rd32.tau[0])) / float(rd32.sigma[0])
m16 = np.concatenate([margins(h16), margins(l16)])
m32 = np.concatenate([margins(h32), margins(l32)])
d = m16 - m32
res = {"max_abs_margin_diff_sigma": float(np.abs(d).max()),
       "rms_margin_diff_sigma": float(np.sqrt((d ** 2).mean())),
       "pearson_r": float(np.corrcoef(m16, m32)[0, 1]),
       "sign_agreement": float((np.sign(m16) == np.sign(m32)).mean()),
       "per_dtype": out}
# fp16 is acceptable only if it perturbs margins far less than the calibration
# shifts the experiment deliberately injects (Delta grid ~0.6-1.2 sigma).
res["FP16_ACCEPTABLE"] = bool(res["max_abs_margin_diff_sigma"] < 0.05
                              and res["pearson_r"] > 0.999)
print("\n" + json.dumps(res, indent=2))
pathlib.Path("results/v3_2/tables/dtype_fidelity_tinyllama.json").write_text(
    json.dumps(res, indent=2))
