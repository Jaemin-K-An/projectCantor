import json, hashlib, pathlib, sys, datetime, subprocess
ROOT = pathlib.Path("."); F = ROOT/"configs/v3_3_5/PRE_ANALYSIS_FREEZE.json"
def sha(p):
    p = pathlib.Path(p); return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
FILES = {"splits":"configs/v3_3_5/splits.json",
 "classifier":"scripts/v3_3_5/final_claim_check_v335.py",
 "affine_coordinate":"llm/src/cantor_guard_v335/affine_coordinate.py",
 "certificate":"llm/src/cantor_guard_v335/certificate.py",
 "affine_policy":"llm/src/cantor_guard_v335/affine_guarded_policy.py",
 "g1_generation":"llm/src/cantor_guard_v335/g1_only_generation.py",
 "g1_boundary_dev":"results/v3_3_5/tables/g1_boundary_D_beh_g1_dev.json",
 "g1_boundary_confirm":"results/v3_3_5/tables/g1_boundary_D_beh_g1_confirm.json",
 "certificate_result":"results/v3_3_5/tables/certificate_summary.json"}
p = {"frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "git_sha": subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
 "files": {k: sha(v) for k,v in FILES.items()}, "file_paths": FILES,
 "classifier_sha256": sha(FILES["classifier"]),
 "W_rule": "W = 1.05 * Q_0.99(|z_G1 - anchor|) on D_window_cal only",
 "rho_grid": [0.25,0.28,0.30,1/3,0.36,0.40,0.44], "depth": 3,
 "q_G1_target": 0.02, "q_cap": 0.05, "budget_tolerance_final": 0.03,
 "sesoi_auc": 0.02, "bootstrap_reps": 20000, "multiplicity": "max-T simultaneous",
 "CHRONOLOGY": {
   "before_seal": [
     "Theorem CP proved; affine certificate derived (exact argmax 1/3 at n=3)",
     "G1-only hook implemented and phase trace verified (prefill 1, G1 1, G2+ 46)",
     "G1 dose bracket searched on D_beh_g1_dev ONLY; grid then frozen",
     "confirmatory fit on D_beh_g1_confirm reproduced DEV: TAU_G1_UNIDENTIFIABLE",
     "certificate validated with REAL forward attacks on D_attack_dev, using a "
     "GEOMETRIC anchor explicitly labelled as not behavioural",
     "V3.3.4 negative findings preserved unchanged"],
   "D_final_335_touched": False,
   "reason_final_not_run": ("tau_G1 unidentifiable; substituting tau_mid or the "
                            "V3.3.3 global tau_beh is forbidden, so no admissible "
                            "behavioural centre exists and the final set was "
                            "deliberately left unspent")}}
F.write_text(json.dumps(p, indent=2))
print(f"SEALED -> {F}\n  D_final_335 touched: {p['CHRONOLOGY']['D_final_335_touched']}")
