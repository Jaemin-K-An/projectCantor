import json, hashlib, pathlib, datetime, subprocess
ROOT = pathlib.Path("."); F = ROOT/"configs/v3_3_5a/PRE_ANALYSIS_FREEZE.json"
def sha(p):
    p = pathlib.Path(p); return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
FILES = {"splits":"configs/v3_3_5a/splits.json",
 "classifier":"scripts/v3_3_5a/final_claim_check_v335a.py",
 "p0_residual":"llm/src/cantor_guard_v335a/p0_residual.py",
 "p0_direction":"results/v3_3_5a/tables/p0_direction.json",
 "p0_boundary_dev":"results/v3_3_5a/tables/p0_boundary_D_beh_P0_dev.json",
 "p0_boundary_confirm":"results/v3_3_5a/tables/p0_boundary_D_beh_P0_confirm.json",
 "phase_causality":"results/v3_3_5a/tables/phase_causality.json"}
p = {"frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "git_sha": subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
 "files": {k: sha(v) for k,v in FILES.items()}, "file_paths": FILES,
 "classifier_sha256": sha(FILES["classifier"]),
 "math_frozen_from": "V3.3.5: eps(rho)=2W rho^2(1-2rho), argmax 1/3, eps_C=2W/27",
 "identifiability_gate": {"beta_std_min": 0.10, "tau_ci_width_max_sigma": 3.0,
                          "note": "scale-normalized, fixing V3.3.5's unit-dependent 0.05"},
 "CHRONOLOGY": {
   "before_seal": [
     "padding-safe P0 extraction implemented and tested on BOTH padding sides",
     "P0 direction re-estimated on a fresh block (never used as an endpoint)",
     "PHASE 3 passed: a +4 sigma P0 dose moves first-token logits (max|dlogit| 2.59, top-1 flip 6.2%)",
     "PHASE 4 narrow +-2 sigma gate FAILED with zero refusal variation",
     "DEV dose-response then run at the SAME amplitude already used for G1 in "
     "V3.3.5, which sections 13/37 require for the phase comparison -- same "
     "direction, token and layer, larger amplitude only",
     "confirmatory split reproduced DEV: B2_P0_CAUSAL_BUT_BOUNDARY_IMPRECISE",
     "phase comparison computed across P0 / G1 / GLOBAL"],
   "D_final_P0_touched": False,
   "reason_final_not_run": ("section 29: the P0 behavioural gate failed, so the "
                            "final set was deliberately left unspent rather than "
                            "run to obtain a number")}}
F.write_text(json.dumps(p, indent=2))
print(f"SEALED -> {F}\n  D_final_P0 touched: {p['CHRONOLOGY']['D_final_P0_touched']}")
