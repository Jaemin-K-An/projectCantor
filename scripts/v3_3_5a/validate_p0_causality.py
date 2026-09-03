"""V3.3.5a PHASE 2/3/4 -- P0 direction, the token-1 logit test, causality gate."""
import argparse, sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd, torch
from cantor_guard.models import load_model
from cantor_guard.probes import diff_in_means_direction, separability
from cantor_guard.io import seed_everything
from cantor_guard_v32.metrics32 import is_refusal32, coherence32
from cantor_guard_v335a.p0_residual import (last_valid_prompt_residuals,
                                            generate_p0_only)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
ap.add_argument("--batch", type=int, default=8)
a = ap.parse_args()
seed_everything(20260907)
TAB = pathlib.Path("results/v3_3_5a/tables"); TAB.mkdir(parents=True, exist_ok=True)
LAYER = json.loads(pathlib.Path("results/v3_3_2/tables/"
                  "phase_calibration_qwen2.5-0.5b-instruct.json").read_text())["layer"]
b = load_model(a.model)

dh = pd.read_csv("results/v3_3_5a/cache/D_direction_P0.csv")
hl = pd.read_csv("results/v3_3_5a/cache/harmless.csv")
Hh = last_valid_prompt_residuals(b, list(dh.prompt), LAYER, batch_size=a.batch)
Hl = last_valid_prompt_residuals(b, list(hl.prompt)[:len(dh)], LAYER, batch_size=a.batch)
v = np.squeeze(diff_in_means_direction(Hh[:, None, :], Hl[:, None, :]))
v = v / np.linalg.norm(v)
sep = float(np.atleast_1d(separability(Hh[:, None, :], Hl[:, None, :],
                                       v[None, :]))[0])
zc = Hh @ v
SIG_P0 = float(zc.std(ddof=1))
print(f"P0 direction: layer {LAYER}  separability {sep:.4f}  sigma_P0 {SIG_P0:.4f}")

# ---- PHASE 3: does a P0 intervention change the FIRST-TOKEN logits? ----
dev = pd.read_csv("results/v3_3_5a/cache/D_beh_P0_dev.csv")
base = generate_p0_only(b, list(dev.prompt)[:16], v=v, layer=LAYER, dose=0.0,
                        max_new_tokens=1, batch_size=a.batch,
                        want_first_logits=True, record=True)
dosed = generate_p0_only(b, list(dev.prompt)[:16], v=v, layer=LAYER,
                         dose=+4.0 * SIG_P0, max_new_tokens=1,
                         batch_size=a.batch, want_first_logits=True)
L0, L1 = base["first_logits"], dosed["first_logits"]
maxd = float(np.abs(L1 - L0).max())
p0 = torch.softmax(torch.tensor(L0), -1); p1 = torch.softmax(torch.tensor(L1), -1)
kl = float((p0 * (p0.clamp_min(1e-12).log() - p1.clamp_min(1e-12).log())).sum(-1).mean())
flip = float((L0.argmax(-1) != L1.argmax(-1)).mean())
tr = base["traces"][0]
p0_only = tr[0]["phase"] == "PREFILL" and all(not r["controller_applied"] and
                                              not r["attack_applied"] for r in tr[1:])
print(f"\nPHASE 3  first-token logit response to a +4 sigma P0 dose:")
print(f"  max|dlogit| = {maxd:.4f}   KL = {kl:.4f}   top-1 flipped = {flip:.1%}")
print(f"  intervention confined to PREFILL: {p0_only}")
if maxd < 1e-6:
    raise SystemExit("P0_INTERVENTION_NOT_UPSTREAM_OF_FIRST_TOKEN")

# ---- PHASE 4: causality gate, symmetric doses on DEV ----
print(f"\nPHASE 4  P0 causality gate (symmetric doses, DEV n={len(dev)})")
rows = []
for k in (-2, -1, 0, 1, 2):
    R = generate_p0_only(b, list(dev.prompt), v=v, layer=LAYER,
                         dose=float(k) * SIG_P0, max_new_tokens=48,
                         batch_size=a.batch)
    ref = float(np.mean([is_refusal32(t) for t in R["texts"]]))
    coh = float(np.mean([coherence32(t) for t in R["texts"]]))
    rows.append({"dose_sigma": k, "z_p0": float(R["z_attacked"].mean()),
                 "refusal": ref, "coherence": coh})
    print(f"  {k:+d} sigma  z_P0={rows[-1]['z_p0']:+.3f}  refusal={ref:.3f}  coh={coh:.3f}")
d = pd.DataFrame(rows)
mono = float(np.corrcoef(d.dose_sigma, d.refusal)[0, 1])
span = float(d.refusal.max() - d.refusal.min())
degen = bool(d.coherence.min() < 0.6)
print(f"  corr(dose, refusal) = {mono:+.4f}   refusal span = {span:.4f}"
      f"   degeneration = {degen}")
ok = (abs(mono) > 0.7) and (span >= 0.05) and not degen
print(f"\n  P0 direction causal: {ok}" + ("" if ok else "  -> P0_DIRECTION_NOT_CAUSAL"))
np.save("results/v3_3_5a/cache/v_p0.npy", v)
json.dump({"layer": LAYER, "separability": sep, "sigma_P0": SIG_P0,
           "first_token": {"max_abs_dlogit": maxd, "kl": kl, "top1_flip": flip,
                           "prefill_only": bool(p0_only)},
           "causality": rows, "corr_dose_refusal": mono, "refusal_span": span,
           "degeneration": degen, "causal": bool(ok),
           "v_sha": __import__("hashlib").sha256(v.tobytes()).hexdigest()[:16]},
          open(TAB/"p0_direction.json", "w"), indent=2)
print(f"wrote {TAB}/p0_direction.json")
