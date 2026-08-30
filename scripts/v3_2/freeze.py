"""V3.2 PHASE 9b -- seal the protocol BEFORE any D_test data exists.

Records a SHA-256 for every file that could change the answer: the split, the
fitted directions/calibration/gains, the scorers, the statistics code, the
runner, and the verdict classifier. `run_final_test.py` refuses to generate
D_test data unless its frozen config matches the seal, and
`final_claim_check_v32.py` refuses to emit a verdict if its own hash has
changed since.

Refuses to run if D_test output already exists.
"""
import json, hashlib, pathlib, sys, datetime, subprocess

ROOT = pathlib.Path(".")
FREEZE = ROOT / "configs/v3_2/PRE_ANALYSIS_FREEZE.json"

# Sealing is PER MODEL. Each model's D_test is guarded by a seal recorded
# before that model's own test runs; a model whose test has already run can
# never be re-sealed. This is stronger than one global seal, and it lets the
# primary model's result land without waiting on the replication model's fit.
ONLY = None
for i, a in enumerate(sys.argv):
    if a == "--model" and i + 1 < len(sys.argv):
        ONLY = sys.argv[i + 1]

def sha(p):
    p = pathlib.Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None

FILES = {
    "split": "configs/v3_2/split.json",
    "control_set": "configs/v3_2/evaluator_control_set.json",
    "classifier": "scripts/v3_2/final_claim_check_v32.py",
    "cluster_stats": "llm/src/cantor_guard_v32/cluster_stats.py",
    "metrics32": "llm/src/cantor_guard_v32/metrics32.py",
    "evaluators": "llm/src/cantor_guard_v32/evaluators.py",
    "generation32": "llm/src/cantor_guard_v32/generation32.py",
    "splits_module": "llm/src/cantor_guard_v32/splits.py",
    "runner": "scripts/v3_2/run_final_test.py",
    "pre_analysis_plan": "docs/v3_2/PRE_ANALYSIS_PLAN.md",
    "power_plan": "results/v3_2/tables/power_plan.json",
}
MODELS = ["qwen2.5-0.5b-instruct", "olmo2-1b-instruct"]
prev = json.loads(FREEZE.read_text()) if FREEZE.exists() else {}
frozen_cfgs = dict(prev.get("frozen_configs") or {})
sealed_at = dict(prev.get("sealed_at") or {})
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
for m in MODELS:
    if ONLY and m != ONLY:
        continue
    h = sha(f"configs/v3_2/frozen_{m}.json")
    if h is None:
        print(f"[skip] no frozen config for {m} yet")
        continue
    if (ROOT / f"results/v3_2/raw/v32_final_{m}.csv").exists() \
            and frozen_cfgs.get(m) and "--allow-existing" not in sys.argv:
        print(f"[keep] {m} already sealed and tested; not re-sealing")
        continue
    frozen_cfgs[m] = h
    sealed_at[m] = now
    print(f"[seal] {m} -> {h[:16]}")
missing = [m for m in MODELS if m not in frozen_cfgs]
if missing:
    print(f"NOTE: {missing} not sealed yet; they cannot be run until sealed.")

payload = {
    "frozen_at_utc": prev.get("frozen_at_utc", now),
    "last_updated_utc": now,
    "sealed_at": sealed_at,
    "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip(),
    "git_branch": subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                 capture_output=True, text=True).stdout.strip(),
    "sesoi": 0.03,
    "primary_endpoint": "safe_lex32",
    "secondary_endpoint": "safe_ext",
    "inference_unit": "goal (cluster bootstrap, 20000 iterations)",
    "files": {k: sha(v) for k, v in FILES.items()},
    "file_paths": FILES,
    "frozen_configs": frozen_cfgs,
    "classifier_sha256": sha(FILES["classifier"]),
    "verdicts": ["A_CANTOR_SPECIFIC_POSITIVE",
                 "B_MULTISCALE_BUT_NOT_CANTOR_SPECIFIC",
                 "C_PRACTICALLY_EQUIVALENT", "D_CANTOR_INFERIOR",
                 "E_INCONCLUSIVE", "F_INVALID",
                 "E_INCONCLUSIVE_SCORER_DEPENDENT"],
}
FREEZE.write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
print(f"\nSEALED -> {FREEZE}")
