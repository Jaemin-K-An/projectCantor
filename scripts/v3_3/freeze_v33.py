"""V3.3 PHASE 14 -- seal the structural protocol before benchmarks run."""
import json, hashlib, pathlib, sys, datetime, subprocess

ROOT = pathlib.Path(".")
FREEZE = ROOT / "configs/v3_3/PRE_ANALYSIS_FREEZE.json"
existing = list((ROOT / "results/v3_3/tables").glob("benchmark_summary.json"))
if existing and "--allow-existing" not in sys.argv:
    raise SystemExit("benchmark output already exists; refusing to re-freeze")

def sha(p):
    p = pathlib.Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None

FILES = {
    "encoding": "configs/v3_3/encoding.json",
    "classifier": "scripts/v3_3/final_claim_check_v33.py",
    "symbolic_cantor": "llm/src/cantor_guard_v33/symbolic_cantor.py",
    "general_recursive": "llm/src/cantor_guard_v33/general_recursive.py",
    "certificates": "llm/src/cantor_guard_v33/certificates.py",
    "complexity": "llm/src/cantor_guard_v33/complexity.py",
    "pareto": "llm/src/cantor_guard_v33/pareto.py",
    "pre_analysis_plan": "docs/v3_3/PRE_ANALYSIS_PLAN.md",
}
payload = {
    "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip(),
    "git_branch": subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                 capture_output=True, text=True).stdout.strip(),
    "inherits_v32_verdict": "C_PRACTICALLY_EQUIVALENT (model A, both scorers)",
    "h1_status": "NOT SUPPORTED -- not re-tested in V3.3",
    "sesoi": 0.03,
    "strict_margin": 0.05,
    "primary_metrics": ["M1_canonical_bits", "M2_certificate_assertions",
                        "M3_point_query_words"],
    "files": {k: sha(v) for k, v in FILES.items()},
    "file_paths": FILES,
    "classifier_sha256": sha(FILES["classifier"]),
    "verdicts": ["A_CANTOR_SPECIFIC_STRUCTURAL_ADVANTAGE",
                 "B_RECURSIVE_SELF_SIMILARITY_ADVANTAGE",
                 "C_DESCRIPTION_ONLY_ADVANTAGE",
                 "D_CERTIFICATION_ONLY_ADVANTAGE",
                 "E_NO_STRUCTURAL_ADVANTAGE", "F_INCONCLUSIVE"],
}
FREEZE.write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
print(f"\nSEALED -> {FREEZE}")
