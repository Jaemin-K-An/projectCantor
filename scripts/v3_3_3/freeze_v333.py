"""V3.3.3 -- seal BEFORE any D_final generation, with full chronology."""
import json, hashlib, pathlib, sys, datetime, subprocess
ROOT = pathlib.Path("."); F = ROOT / "configs/v3_3_3/PRE_ANALYSIS_FREEZE.json"
if list((ROOT / "results/v3_3_3/raw").glob("systemA_*.csv")) and "--allow-existing" not in sys.argv:
    raise SystemExit("D_final output exists; refusing to re-freeze")
def sha(p):
    p = pathlib.Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
FILES = {
 "protocol": "configs/v3_3_3/protocol.json",
 "splits": "configs/v3_3_3/splits.json",
 "behavioral_boundary_module": "llm/src/cantor_guard_v333/behavioral_boundary.py",
 "adversarial_crossing": "llm/src/cantor_guard_v333/adversarial_crossing.py",
 "absolute_guard": "llm/src/cantor_guard_v332/absolute_guard.py",
 "phase_generation": "llm/src/cantor_guard_v332/phase_generation.py",
 "classifier": "scripts/v3_3_3/final_claim_check_v333.py",
 "behavioral_result": "results/v3_3_3/tables/behavioral_boundary.json",
 "quantile_result": "results/v3_3_3/tables/quantile_sensitivity.csv",
}
payload = {
 "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "git_sha": subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
 "branch": subprocess.run(["git","rev-parse","--abbrev-ref","HEAD"],capture_output=True,text=True).stdout.strip(),
 "files": {k: sha(v) for k, v in FILES.items()}, "file_paths": FILES,
 "classifier_sha256": sha(FILES["classifier"]),
 "CHRONOLOGY_DISCLOSURE": {
   "exploratory_before_seal": [
     "V3.3.2 U_EST_mid quantiles (q50=0.02528 .. q95=0.07218) were already known",
     "V3.3.2 System B random-perturbation result was already known",
     "V3.3.3 behavioural dose-response on D_beh was run before this seal, and its "
     "first frozen grid (+-3 sigma) failed to bracket the transition; the grid was "
     "extended to -10 sigma on the SAME calibration block and refit ONCE",
     "the identifiability gate was STRENGTHENED after seeing the first fit "
     "(transition_observed, ci_width_reasonable) -- both make identification harder",
     "adversarial crossing was validated on synthetic geometry, not on D_final"],
   "rho_grid_chosen": "before seeing any D_final data; brackets 1/3 symmetrically "
                      "and spans the V3.3.2-derived region, which WAS known",
   "v332_system_B_ran_before_its_own_seal": True,
   "D_final_touched_before_this_seal": False},
 "protocol": json.loads(pathlib.Path("configs/v3_3_3/protocol.json").read_text()),
 "verdict_categories": ["A_CANTOR_GENERATION_POSITIVE","B_OTHER_RHO_GENERATION_POSITIVE",
   "C_PRACTICALLY_EQUIVALENT","D_CANTOR_INFERIOR","E_METRIC_OR_MODEL_DEPENDENT",
   "F_INCONCLUSIVE","G_BEHAVIORAL_BOUNDARY_UNIDENTIFIABLE"],
}
F.write_text(json.dumps(payload, indent=2))
print(f"SEALED -> {F}")
print(f"  git {payload['git_sha'][:12]}  classifier {payload['classifier_sha256'][:16] if payload['classifier_sha256'] else 'PENDING'}")
print(f"  D_final touched before seal: {payload['CHRONOLOGY_DISCLOSURE']['D_final_touched_before_this_seal']}")
