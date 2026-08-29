"""*** POST-HOC ANALYSIS -- NOT PRE-REGISTERED ***

Designed AFTER the frozen test, to DIAGNOSE why the pre-registered primary
endpoint came out degenerate. It does not re-open the test and it cannot
change any pre-registered verdict: all criteria in
docs/v2/PREREGISTRATION.md §4 already FAILED and stay failed.

Question: is the barrier controllers' apparent "refusal" real refusal, or is it
output degradation caused by a large, rapidly-varying residual push?

Same frozen controller configs, same held-out attack, reduced grid for cost.
"""
import sys, json, re
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
from cantor_guard.models import load_model
from cantor_guard.probes import RefusalDirections
from cantor_guard.datasets import load_jbb, load_xstest, grouped_split, prompt_id
from cantor_guard.control_baselines import make_controller
from cantor_guard.generation import generate
from cantor_guard.safety_eval import (is_refusal, compliance_score, coherence_score,
                                      coherent_refusal, nonascii_frac, repetition_frac)
from cantor_guard.attacks import TEST_ATTACKS, apply_prompt_attack
from cantor_guard.io import write_table, V2_CACHE, V2_PRIVATE, seed_everything

MK = "qwen2.5-0.5b-instruct"
seed_everything(20260829)
b = load_model(MK)
z = np.load(V2_CACHE / f"{MK}_dirs.npz")
dirs = RefusalDirections(list(z["layers"]), z["diff_means_v"], z["diff_means_tau"],
                         z["diff_means_sigma"], "diff_means")
cfgtxt = open("configs/v2/llm_test.yaml").read()
def parse_cfg(txt):
    out, cur = {}, None
    for line in txt.splitlines():
        m = re.match(r"^  (L\d_\w+):\s*$", line)
        if m: cur = m.group(1); out[cur] = {}; continue
        if cur:
            m2 = re.match(r"^    (\w+): (.+)$", line)
            if m2:
                k, v = m2.group(1), m2.group(2).strip()
                out[cur][k] = eval(v) if v[0] in "[0123456789." else v
    return out
CFG = parse_cfg(cfgtxt)

harm, _ = load_jbb(); sp = grouped_split(harm, seed=20260829)
goals = list(sp.test.prompt)[:35]
xs = load_xstest(); safe = list(xs[~xs.is_harmful].prompt)[:35]
P = [apply_prompt_attack(g, TEST_ATTACKS["authority_test"]) for g in goals]

FAMS = ["L0_none", "L1_constant", "L2_central", "L3_periodic",
        "L5_shuffled", "L6_center_anchored", "L7_cantor"]
rows, priv = [], []
ACT_NORM = 16.09     # measured mean ||h|| at layer 13 (diagnostic_outputs.txt)
for fam in FAMS:
    c = CFG[fam]
    ctrl = None if fam == "L0_none" else make_controller(
        fam, n=int(c["n"]), B_total=1.0, eta=float(c["eta"]),
        gamma=float(c["gamma"]), seed=1)
    for eps in (0.0, 6.0):
        for kind, prompts in (("harmful", P), ("safe", safe)):
            t, st = generate(b, prompts, max_new_tokens=60, batch_size=16,
                             dirs=dirs, layers=list(c["layers"]), controller=ctrl,
                             gamma=float(c["gamma"]), attack_eps=eps,
                             attack_layers=list(c["layers"]), attack_ortho=0.3,
                             harm_mask=np.ones(len(prompts), bool) if kind == "harmful"
                                       else np.ones(len(prompts), bool))
            for i, (x, s_) in enumerate(zip(t, st)):
                rows.append({"family": fam, "eps_pct": eps, "kind": kind,
                             "pid": prompt_id(prompts[i]),
                             "refusal": int(is_refusal(x)),
                             "coherence": coherence_score(x),
                             "nonascii": nonascii_frac(x),
                             "repetition": repetition_frac(x),
                             "coherent_refusal": int(coherent_refusal(x)),
                             "compliance": compliance_score(x),
                             "int_mean": s_["int_mean"],
                             "int_rel_actnorm": s_["int_mean"] / ACT_NORM,
                             "m_min": s_["m_min"]})
                priv.append({"family": fam, "eps": eps, "kind": kind, "text": x})
    print(f"  {fam} done", flush=True)

df = pd.DataFrame(rows)
write_table(df, f"llm_posthoc_degradation_{MK}.csv",
            meta={"phase": "V2-POSTHOC", "preregistered": False,
                  "note": "diagnosis of the degenerate primary endpoint"})
pd.DataFrame(priv).to_csv(V2_PRIVATE / f"posthoc_completions_{MK}.csv", index=False)

print("\n" + "=" * 104)
print("POST-HOC: is 'refusal' real, or is it output degradation?")
print("=" * 104)
g = df[df.kind == "harmful"].groupby("family").agg(
    int_rel=("int_rel_actnorm", "mean"), refusal=("refusal", "mean"),
    coherence=("coherence", "mean"), nonascii=("nonascii", "mean"),
    repetition=("repetition", "mean"),
    coherent_refusal=("coherent_refusal", "mean"),
    compliance=("compliance", "mean")).reindex(FAMS)
print(g.to_string(float_format="{:.4f}".format))
gs = df[df.kind == "safe"].groupby("family").agg(
    false_refusal=("refusal", "mean"),
    coherent_false_refusal=("coherent_refusal", "mean"),
    coherence=("coherence", "mean")).reindex(FAMS)
print("\nbenign (XSTest safe):")
print(gs.to_string(float_format="{:.4f}".format))
print("\nCOHERENT-refusal Pareto (harmful coherent refusal vs benign coherent false refusal):")
pa = pd.DataFrame({"harm_coh_refusal": g.coherent_refusal,
                   "benign_coh_false_refusal": gs.coherent_false_refusal,
                   "coherence_harm": g.coherence,
                   "intervention_rel_actnorm": g.int_rel}).reindex(FAMS)
print(pa.to_string(float_format="{:.4f}".format))
write_table(pa.reset_index(), f"llm_posthoc_pareto_{MK}.csv", raw=False,
            meta={"phase": "V2-POSTHOC", "preregistered": False})
