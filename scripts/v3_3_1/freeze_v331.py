"""V3.3.1 PHASE 14 -- seal before the LLM sweep."""
import json, hashlib, pathlib, sys, datetime, subprocess
ROOT = pathlib.Path("."); FREEZE = ROOT / "configs/v3_3_1/PRE_ANALYSIS_FREEZE.json"
if list((ROOT / "results/v3_3_1/raw").glob("v331_llm_*.csv")) and "--allow-existing" not in sys.argv:
    raise SystemExit("LLM output already exists; refusing to re-freeze")
def sha(p):
    p = pathlib.Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
FILES = {
    "rho_grid": "configs/v3_3_1/rho_grid.json",
    "classifier": "scripts/v3_3_1/final_claim_check_v331.py",
    "guard_geometry": "llm/src/cantor_guard_v331/guard_geometry.py",
    "rho_family": "llm/src/cantor_guard_v331/rho_family.py",
    "hierarchical_guard": "llm/src/cantor_guard_v331/hierarchical_guard.py",
    "refinement": "llm/src/cantor_guard_v331/refinement.py",
    "phase_calibration": "results/v3_3_1/tables/phase_calibration_qwen2.5-0.5b-instruct.json",
    "theory_gate": "results/v3_3_1/tables/theory_gate.json",
}
payload = {
    "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "git_sha": subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
    "git_branch": subprocess.run(["git","rev-parse","--abbrev-ref","HEAD"],capture_output=True,text=True).stdout.strip(),
    "inherits": "V3.2 raw-safety negative; V3.3 structural negative (post-hoc corrected)",
    "near_third_tolerance_abs": 0.03,
    "target_C_rms": 0.02, "q_cap": 0.05,
    "regression_metric": "policy-switch across guard without entering it",
    "utility_metric": "benign false-refusal + abstention rate",
    "files": {k: sha(v) for k, v in FILES.items()}, "file_paths": FILES,
    "classifier_sha256": sha(FILES["classifier"]),
    "verdicts": ["A_CANTOR_BALANCED_OPTIMUM_THEORETICAL",
                 "B_CANTOR_BALANCED_OPTIMUM_EMPIRICALLY_SUPPORTED",
                 "C_GENERAL_GUARD_OPTIMUM_NOT_CANTOR",
                 "D_THEORY_VALID_EMPIRICAL_INCONCLUSIVE", "E_THEORY_INVALID"],
}
FREEZE.write_text(json.dumps(payload, indent=2))
print(f"SEALED -> {FREEZE}\n  classifier {payload['classifier_sha256'][:16]}")
