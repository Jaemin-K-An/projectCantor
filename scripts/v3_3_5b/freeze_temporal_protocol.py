import json, hashlib, pathlib, sys, datetime, subprocess
ROOT = pathlib.Path("."); F = ROOT/"configs/v3_3_5b/FREEZE_STAGE_A.json"
if pathlib.Path("results/v3_3_5b/raw/temporal_D_temporal_confirm.csv").exists() \
        and "--allow-existing" not in sys.argv:
    raise SystemExit("confirm output exists; refusing to re-freeze")
def sha(p):
    p = pathlib.Path(p); return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
FILES = {"protocol":"configs/v3_3_5b/protocol_stageA.json",
 "splits":"configs/v3_3_5b/splits.json",
 "temporal_budget":"llm/src/cantor_guard_v335b/temporal_budget.py",
 "temporal_generation":"llm/src/cantor_guard_v335b/temporal_generation.py",
 "runner":"scripts/v3_3_5b/run_matched_budget_temporal.py",
 "regime_gap":"results/v3_3_5b/tables/regime_gap.json",
 "dev_result":"results/v3_3_5b/raw/temporal_D_temporal_dev.csv"}
p = {"frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "git_sha": subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
 "files": {k: sha(v) for k,v in FILES.items()}, "file_paths": FILES,
 "protocol": json.loads(pathlib.Path("configs/v3_3_5b/protocol_stageA.json").read_text()),
 "CHRONOLOGY": {
  "before_seal": [
    "matched-budget machinery implemented; B2 identical across schedules by construction",
    "pre-declared q<=0.05 grid run on DEV: ZERO variation in all 35 cells",
    "regime gap measured on DEV: effects begin at q~0.10; historical global dose q~0.50",
    "grid extended to span the effect regime, cap change disclosed in the protocol",
    "second DEV run showed concentration >= distribution at every matched B2"],
  "D_temporal_confirm_touched": False,
  "D_final_traj_touched": False}}
F.write_text(json.dumps(p, indent=2))
print(f"SEALED -> {F}\n  confirm touched: {p['CHRONOLOGY']['D_temporal_confirm_touched']}")
