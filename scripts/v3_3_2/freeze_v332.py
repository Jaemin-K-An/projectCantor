"""V3.3.2 PHASE 12 -- seal. Records honestly what already ran."""
import json, hashlib, pathlib, datetime, subprocess
ROOT = pathlib.Path("."); F = ROOT/"configs/v3_3_2/PRE_ANALYSIS_FREEZE.json"
def sha(p):
    p = pathlib.Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
FILES = {
 "final_split": "configs/v3_3_2/final_split.json",
 "classifier": "scripts/v3_3_2/final_claim_check_v332.py",
 "absolute_guard": "llm/src/cantor_guard_v332/absolute_guard.py",
 "phase_state": "llm/src/cantor_guard_v332/phase_state.py",
 "phase_generation": "llm/src/cantor_guard_v332/phase_generation.py",
 "phase_residuals": "llm/src/cantor_guard_v332/phase_residuals.py",
 "calibration": "llm/src/cantor_guard_v332/calibration.py",
 "uncertainty": "llm/src/cantor_guard_v332/uncertainty.py",
 "phase_calibration_result": "results/v3_3_2/tables/phase_calibration_qwen2.5-0.5b-instruct.json",
 "theory_gate": "results/v3_3_2/tables/theory_gate.json",
}
payload = {
 "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "git_sha": subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
 "cantor_band": [1/3-0.03, 1/3+0.03], "sesoi": 0.03,
 "target_C_rms": 0.02, "q_cap": 0.05,
 "rho_grid": [0.20,0.24,0.28,1/3,0.36,0.40,0.44],
 "depth_grid": [2,3,5],
 "ORDERING_DISCLOSURE": (
   "The System B real-coordinate run on D_final executed BEFORE this seal was "
   "written. Its inputs (delta_abs, the guard classifier, the rho grid) were "
   "all fixed beforehand and no rule was changed after seeing it, but the "
   "ordering discipline was not followed, so the System B result is reported "
   "as CONFIRMATORY-WITH-DISCLOSURE rather than fully sealed. The System A run "
   "executes strictly after this seal."),
 "files": {k: sha(v) for k,v in FILES.items()}, "file_paths": FILES,
 "classifier_sha256": sha(FILES["classifier"]),
}
F.write_text(json.dumps(payload, indent=2))
print(f"SEALED -> {F}\n  classifier {payload['classifier_sha256'][:16]}")
print("  ORDERING DISCLOSURE recorded")
