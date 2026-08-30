"""V3.2 PHASE 10/11 -- the final test, on D_test, AFTER the freeze.

Everything this script needs was fitted on another block and written to
configs/v3_2/frozen_<model>.json. Nothing is refitted here: the direction, the
calibration, the Delta grid and every controller gain are read from disk. The
realised C_rms is measured again and reported as a GENERALISATION result --
if it drifts outside tolerance on D_test, that is a finding, not something to
correct by retuning.

Two V3.1 defects are fixed in the output schema:
  * `layout_seed` and `gen_seed` are separate columns (D3). In V3.1 one `seed`
    field meant "layout instance" for the random families and nothing for the
    deterministic ones, which made strict pairing impossible.
  * completions are persisted (to results/v3_2/private/, gitignored) so an
    INDEPENDENT evaluator can score the same text afterwards. V3.1 discarded
    them, which is why no external scoring was possible.

Writes incrementally and resumes.
"""
import argparse, sys, json, time, hashlib, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd, torch
from cantor_guard.models import load_model
from cantor_guard.probes import RefusalDirections
from cantor_guard.datasets import load_jbb, load_xstest
from cantor_guard.attacks import TEST_ATTACKS, apply_prompt_attack
from cantor_guard.io import seed_everything
from cantor_guard_v31.controllers31 import (Controller31, V31_LLM_FAMILIES,
                                            V31_RANDOMISED)
from cantor_guard_v32.generation32 import generate32
from cantor_guard_v32.metrics32 import safe_score32, utility_scores32
from cantor_guard_v32.splits import load_split

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
ap.add_argument("--dtype", default=None)
ap.add_argument("--batch", type=int, default=10)
ap.add_argument("--attacks", default="authority_test,stepwise_test")
ap.add_argument("--n-safe", type=int, default=24)
ap.add_argument("--seed", type=int, default=20260830)
args = ap.parse_args()
seed_everything(args.seed)

FR = pathlib.Path(f"configs/v3_2/frozen_{args.model}.json")
FROZEN = json.loads(FR.read_text())
FROZEN_SHA = hashlib.sha256(FR.read_bytes()).hexdigest()
PLAN = pathlib.Path("configs/v3_2/PRE_ANALYSIS_FREEZE.json")
if not PLAN.exists():
    raise SystemExit("PRE_ANALYSIS_FREEZE.json missing -- refusing to touch "
                     "D_test before the freeze is recorded (harness STOP).")
freeze = json.loads(PLAN.read_text())
if freeze["frozen_configs"].get(args.model) != FROZEN_SHA:
    raise SystemExit(f"frozen config for {args.model} does not match the sealed "
                     f"hash. Refusing to run.\n  sealed: "
                     f"{freeze['frozen_configs'].get(args.model)}\n  actual: {FROZEN_SHA}")
print(f"freeze verified: {args.model} config sha {FROZEN_SHA[:16]}")

split = load_split("configs/v3_2/split.json")
if split["split_sha256"] != FROZEN["split_sha256"]:
    raise SystemExit("split hash mismatch")
harm, ben = load_jbb(); H = harm.set_index("pid")
TEST_PIDS = split["blocks"]["D_test"]
goals = [H.loc[p, "prompt"] for p in TEST_PIDS]
cats = [H.loc[p, "category"] for p in TEST_PIDS]
xs = load_xstest(); safe_prompts = list(xs[~xs.is_harmful].prompt)[:args.n_safe]
print(f"D_test: {len(goals)} held-out goals, {len(set(cats))} categories")

kw = {} if args.dtype is None else {"dtype": getattr(torch, args.dtype)}
b = load_model(args.model, **kw)
z = np.load(f"results/v3_2/cache/{args.model}_frozen_dirs.npz")
dirs = RefusalDirections([int(FROZEN["layer"])], z["diff_means_v"],
                         z["diff_means_tau"], z["diff_means_sigma"], "diff_means")
LAYER = int(FROZEN["layer"])
DELTAS, EPS = FROZEN["delta_grid"], FROZEN["eps_grid"]
ATT = {k: TEST_ATTACKS[k] for k in args.attacks.split(",")}
print(f"layer={LAYER}  Delta={np.round(DELTAS,3)}  eps={EPS}  attacks={list(ATT)}")

import re
L9W = np.array([float(x) for x in re.search(
    r"weights = \[(.*?)\]", open("configs/v3_1/l9_frozen_weights.toml").read(),
    re.S).group(1).split(",")])

INST = []
for key, g in FROZEN["gains"].items():
    fam, s = key.split("|"); s = int(s)
    c = Controller31(fam, n=FROZEN["n_order"], B_total=1.0,
                     gamma=FROZEN["gamma"], eta=g["eta"], seed=s,
                     weights=(L9W if fam == "T8_minimax" else None),
                     max_q=FROZEN["q_cap"])
    INST.append((fam, s, c, g))
print(f"{len(INST)} controller instances (gains READ from the frozen config)")

RAW = pathlib.Path("results/v3_2/raw"); PRIV = pathlib.Path("results/v3_2/private")
RAW.mkdir(parents=True, exist_ok=True); PRIV.mkdir(parents=True, exist_ok=True)
OUT = RAW / f"v32_final_{args.model}.csv"
TXT = PRIV / f"v32_completions_{args.model}.csv"
done = set()
if OUT.exists():
    prev = pd.read_csv(OUT)
    done = set(zip(prev.family, prev.layout_seed, prev.attack, prev.delta, prev.eps))
    print(f"resuming: {len(prev)} rows present, {len(done)} cells done")

t0 = time.time(); n_cells = len(INST) * len(ATT) * len(DELTAS) * len(EPS)
k = 0
for fam, s, c, g in INST:
    for aname, tpl in ATT.items():
        P = [apply_prompt_attack(x, tpl) for x in goals]
        for d in DELTAS:
            for e in EPS:
                k += 1
                if (fam, s, aname, d, e) in done:
                    continue
                txt, st = generate32(b, P, max_new_tokens=FROZEN["max_new_tokens"],
                    batch_size=args.batch, dirs=dirs, layers=[LAYER], controller=c,
                    delta=d, attack_eps=e, attack_ortho=0.3,
                    gen_seed=args.seed, harm_mask=np.ones(len(P), bool))
                rows = [{"model": args.model, "family": fam, "layout_seed": s,
                         "gen_seed": args.seed, "attack": aname, "delta": d,
                         "eps": e, "pid": TEST_PIDS[i], "category": cats[i],
                         "safe_lex32": safe_score32(t),
                         "eta": g["eta"], "sup_deriv": g["sup_deriv"], **st}
                        for i, t in enumerate(txt)]
                pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(),
                                          index=False)
                # completions stay OUT of the tracked tree; needed for external scoring
                pd.DataFrame([{"model": args.model, "family": fam, "layout_seed": s,
                               "attack": aname, "delta": d, "eps": e,
                               "pid": TEST_PIDS[i], "text": t}
                              for i, t in enumerate(txt)]).to_csv(
                    TXT, mode="a", header=not TXT.exists(), index=False)
        print(f"  [{k:4d}/{n_cells}] {fam:20s} s{s} {aname:15s} "
              f"({time.time()-t0:.0f}s)", flush=True)
    ut, ust = generate32(b, safe_prompts, max_new_tokens=FROZEN["max_new_tokens"],
        batch_size=args.batch, dirs=dirs, layers=[LAYER], controller=c,
        delta=0.0, attack_eps=0.0, harm_mask=np.ones(len(safe_prompts), bool))
    U = RAW / f"v32_utility_{args.model}.csv"
    pd.DataFrame([{"model": args.model, "family": fam, "layout_seed": s,
                   "eta": g["eta"], **utility_scores32(ut), **ust}]).to_csv(
        U, mode="a", header=not U.exists(), index=False)

full = pd.read_csv(OUT)
json.dump({"frozen_sha256": FROZEN_SHA, "split_sha256": split["split_sha256"],
           "n_rows": len(full), "n_goals": int(full.pid.nunique()),
           "elapsed_s": time.time() - t0},
          open(f"results/v3_2/cache/{args.model}_final_meta.json", "w"), indent=2)
print(f"\nFINAL TEST COMPLETE: {len(full)} rows, {full.pid.nunique()} goals -> {OUT}")
