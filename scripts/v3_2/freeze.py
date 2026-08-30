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

existing = list((ROOT / "results/v3_2/raw").glob("v32_final_*.csv"))
if existing and "--allow-existing" not in sys.argv:
    raise SystemExit(f"D_test output already exists: {[p.name for p in existing]}\n"
                     "Refusing to (re)freeze after the test has been run.")

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
frozen_cfgs = {m: sha(f"configs/v3_2/frozen_{m}.json") for m in MODELS}
missing = [m for m, v in frozen_cfgs.items() if v is None]
if missing:
    print(f"WARNING: no frozen config yet for {missing}; they cannot be run "
          f"until re-frozen.")

payload = {
    "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
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
