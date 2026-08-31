"""V3.3.3 -- estimate tau_beh from ACTUAL generation behaviour, on D_beh only.

The dose is applied at the hooked layer during generation. v is a unit vector,
so adding lambda*v shifts the projection by exactly lambda; the regression is
run against the REALISED projection, never against the dose index, so nothing
fitted enters the predictor.
"""
import argparse, sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd, torch
from contextlib import contextmanager
from cantor_guard.models import load_model, decoder_layers
from cantor_guard.io import seed_everything
from cantor_guard_v31.generation31 import chat_prompt
from cantor_guard_v32.metrics32 import safe_score32, is_refusal32, coherence32
from cantor_guard_v332.phase_state import PhaseState, PREFILL
from cantor_guard_v333.behavioral_boundary import (
    fit_logistic, isotonic_crossing, identifiability, tau_beh_bootstrap,
    TAU_BEH_UNIDENTIFIABLE)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
ap.add_argument("--batch", type=int, default=10)
ap.add_argument("--max-new", type=int, default=48)
ap.add_argument("--n-boot", type=int, default=20000)
ap.add_argument("--seed", type=int, default=20260902)
a = ap.parse_args()
seed_everything(a.seed)

TAB = pathlib.Path("results/v3_3_3/tables")
V332 = json.loads(pathlib.Path(
    "results/v3_3_2/tables/phase_calibration_qwen2.5-0.5b-instruct.json").read_text())
LAYER = V332["layer"]
z332 = np.load(f"results/v3_3_2/cache/{a.model}_v332_phase.npz")
v = z332["v"]
SIG_G = V332["calibrations"]["G1"]["sigma"]
TAU_MID_G = V332["calibrations"]["G1"]["tau"]
# FROZEN dose grid, in units of sigma_G1 (chosen on calibration data, not D_final)
# GRID EXTENSION, DISCLOSED. The first frozen grid was +-3 sigma. On D_beh the
# observed refusal proportion never left [0.78, 0.98], so the 50% transition was
# never bracketed and tau_beh was UNIDENTIFIABLE by the gate. The grid is
# extended on the SAME calibration block (D_beh -- D_final is untouched) and
# refit ONCE. This is not result-chasing: the location of the behavioural
# boundary has no directional bearing on which rho wins, and the extension makes
# identification possible in either direction.
DOSE_SIGMA = [-10, -8, -6, -5, -4, -3, -2, -1.5, -1, -0.5,
              0, 0.5, 1, 1.5, 2, 3]
DOSES = np.array(DOSE_SIGMA, float) * SIG_G
print(f"layer {LAYER}  tau_mid,G1={TAU_MID_G:+.4f}  sigma_G1={SIG_G:.4f}")
print(f"dose grid (sigma units, EXTENDED): {DOSE_SIGMA}")

pr = pd.read_csv("results/v3_3_3/cache/d_beh_prompts.csv")
prompts, pids = list(pr.prompt), list(pr.pid)
b = load_model(a.model)
vt = torch.tensor(v, dtype=torch.float32, device=b.device)
blocks = decoder_layers(b)


@contextmanager
def dosed(lmbda, store):
    st = PhaseState(); st.reset()

    def hook(_m, _i, out):
        h, rest = (out[0], out[1:]) if isinstance(out, tuple) else (out, None)
        phase = st.observe(h.shape[1], h.shape[1] == 1)
        hf = h.float() + lmbda * vt.view(1, 1, -1)
        if phase != PREFILL and "z" not in store:
            # realised projection at the FIRST decode state (PHASE G1)
            store["z"] = torch.einsum("bsd,d->bs", hf, vt)[:, -1].detach().cpu().numpy()
        hn = hf.to(h.dtype)
        return hn if rest is None else (hn,) + rest

    hd = blocks[LAYER].register_forward_hook(hook)
    try:
        yield
    finally:
        hd.remove()


rows = []
with torch.no_grad():
    for lam in DOSES:
        for i in range(0, len(prompts), a.batch):
            ch = prompts[i:i + a.batch]
            enc = b.tokenizer([chat_prompt(b, p) for p in ch],
                              return_tensors="pt", padding=True).to(b.device)
            store = {}
            with dosed(float(lam), store):
                out = b.model.generate(**enc, max_new_tokens=a.max_new,
                                       do_sample=False,
                                       pad_token_id=b.tokenizer.pad_token_id)
            txt = b.tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                           skip_special_tokens=True)
            for j, t in enumerate(txt):
                rows.append({"pid": pids[i + j], "dose_sigma": float(lam / SIG_G),
                             "lambda": float(lam), "z": float(store["z"][j]),
                             "refusal": int(is_refusal32(t)),
                             "safe": safe_score32(t), "coherence": coherence32(t),
                             "n_words": len((t or "").split())})
        d = pd.DataFrame([r for r in rows if r["lambda"] == lam])
        print(f"  dose {lam/SIG_G:+5.2f} sigma  refusal={d.refusal.mean():.3f}  "
              f"z_mean={d.z.mean():+.3f}  coh={d.coherence.mean():.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/v3_3_3/raw/behavioral_dose_response.csv", index=False)

z, y, pid = df.z.to_numpy(), df.refusal.to_numpy(float), df.pid.to_numpy()
a0, b0 = fit_logistic(z, y)
tau0 = -a0 / b0 if abs(b0) > 1e-12 else float("nan")
R = tau_beh_bootstrap(z, y, pid, n_boot=a.n_boot, seed=a.seed)
iso = isotonic_crossing(z, y)
# MAX_CI_WIDTH_SIGMA is a frozen admissibility bound: a boundary whose 95% CI
# spans more than 3 sigma cannot constrain a guard of any useful width.
MAX_CI_WIDTH_SIGMA = 3.0
gate = identifiability(z, y, a0, b0, tau0, R["tau_ci95"],
                       dose_bins=df.dose_sigma.to_numpy(),
                       max_ci_width_sigma=MAX_CI_WIDTH_SIGMA, sigma=SIG_G)
print(f"\nlogistic: a={a0:+.4f} b={b0:+.4f}  tau_beh={tau0:+.4f}")
print(f"bootstrap tau_beh CI95 {np.round(R['tau_ci95'],4).tolist()}  "
      f"slope CI95 {np.round(R['slope_ci95'],4).tolist()}  prompts={R['n_prompts']}")
print(f"isotonic crossing: {iso}")
print("identifiability:", json.dumps(gate))

if gate["all_pass"]:
    tau_beh, status = tau0, "IDENTIFIED"
else:
    tau_beh, status = None, TAU_BEH_UNIDENTIFIABLE
    print(f"\n*** {TAU_BEH_UNIDENTIFIABLE} -- tau_mid must NOT be substituted ***")

# U_EST_beh: sampling uncertainty of tau_beh, mapped to controller coordinates
GAMMA = V332["gamma"]
def r_of(zv, tau, sg):
    return 1.0 / (1.0 + np.exp(np.clip(GAMMA * (zv - tau) / sg, -60, 60)))
UB = None
if status == "IDENTIFIED":
    dr = np.abs(r_of(R["tau_samples"], tau0, SIG_G) - 0.5)
    UB = {f"q{p}": float(np.quantile(dr, p / 100)) for p in (50, 75, 90, 95)}
    print(f"U_EST_beh quantiles: {json.dumps({k: round(x,5) for k,x in UB.items()})}")

out = {"model": a.model, "layer": LAYER, "n_prompts": len(pids),
       "dose_grid_sigma": DOSE_SIGMA,
       "grid_extended_from": [-3, 3],
       "grid_extension_disclosed": True,
       "tau_mid_G1": TAU_MID_G, "sigma_G1": SIG_G,
       "logistic": {"a": a0, "b": b0, "tau_beh": tau0},
       "bootstrap": {k: v2 for k, v2 in R.items() if k != "tau_samples"},
       "isotonic_crossing": iso, "identifiability": gate, "status": status,
       "max_ci_width_sigma": MAX_CI_WIDTH_SIGMA,
       "tau_beh": tau_beh,
       "gap_tau_mid_minus_tau_beh_sigma": (
           float((TAU_MID_G - tau0) / SIG_G) if status == "IDENTIFIED" else None),
       "U_EST_beh_quantiles": UB,
       "refusal_by_dose": df.groupby("dose_sigma").refusal.mean().to_dict()}
(TAB / "behavioral_boundary.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote {TAB}/behavioral_boundary.json")
