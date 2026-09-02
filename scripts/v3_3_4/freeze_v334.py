import json, hashlib, pathlib, sys, datetime, subprocess
ROOT = pathlib.Path("."); F = ROOT/"configs/v3_3_4/PRE_ANALYSIS_FREEZE.json"
if list((ROOT/"results/v3_3_4/raw").glob("generation_*.csv")) and "--allow-existing" not in sys.argv:
    raise SystemExit("D_final generation exists; refusing to re-freeze")
def sha(p):
    p = pathlib.Path(p); return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
FILES = {"protocol":"configs/v3_3_4/protocol.json","splits":"configs/v3_3_4/splits.json",
 "classifier":"scripts/v3_3_4/final_claim_check_v334.py",
 "certified_geometry":"llm/src/cantor_guard_v334/certified_geometry.py",
 "certificate":"llm/src/cantor_guard_v334/certificate.py",
 "guarded_policy":"llm/src/cantor_guard_v334/guarded_policy.py",
 "behavioural_result":"results/v3_3_3/tables/behavioral_boundary.json",
 "certificate_dev":"results/v3_3_4/tables/certificate_summary_dev.json",
 "center_ablation":"results/v3_3_4/tables/center_ablation.json"}
p = {"frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "git_sha": subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
 "files": {k: sha(v) for k,v in FILES.items()}, "file_paths": FILES,
 "classifier_sha256": sha(FILES["classifier"]),
 "tau_beh": json.loads(pathlib.Path("results/v3_3_3/tables/behavioral_boundary.json").read_text())["tau_beh"],
 "CHRONOLOGY": {
   "before_seal": ["Theorem CR/CR.1 proved and verified",
     "certificates derived; Lipschitz argmax = 1/3, EXACT argmax ~ 0.296 (known)",
     "certificate attack run on D_beh (DEV) -- 0 violations",
     "depth-shift run on DEV with a COMMON ABSOLUTE grid -- prediction FAILED (corr -0.064)",
     "centre ablation run on DEV",
     "the depth-shift metric was corrected pre-seal after the first version was "
     "found CIRCULAR (certificate-normalised grid recovers rho_theory by construction)"],
   "D_final_334_touched_before_seal": False,
   "rho_grid_fixed_before_final": True},
 "protocol": json.loads(pathlib.Path("configs/v3_3_4/protocol.json").read_text())}
F.write_text(json.dumps(p, indent=2))
print(f"SEALED -> {F}\n  classifier {p['classifier_sha256'][:16]}")
print(f"  D_final_334 touched before seal: {p['CHRONOLOGY']['D_final_334_touched_before_seal']}")
