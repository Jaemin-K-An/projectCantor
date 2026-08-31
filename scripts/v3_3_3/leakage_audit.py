"""V3.3.3 -- hard failure if any D_final prompt was used for ANY fitting."""
import sys, json, pathlib
sys.path.insert(0, "llm/src")
import pandas as pd
TAB = pathlib.Path("results/v3_3_3/tables")
S = json.loads(pathlib.Path("configs/v3_3_3/splits.json").read_text())
final = set(S["D_final"]); beh = set(S["D_beh"])
reg = pd.read_csv(TAB / "global_prompt_usage_registry.csv")
rows = []
# every prior tuning role
tuning = reg[reg.ever_used_for_tuning].prompt_id.tolist()
prior_final = reg[(reg.ever_used_for_final) &
                  (reg.split != "D_final_v333")].prompt_id.tolist()
for name, ids in (("ever_used_for_tuning", tuning),
                  ("prior_version_final", prior_final),
                  ("D_beh", list(beh))):
    hit = final & set(ids)
    rows.append({"check": f"D_final ∩ {name}", "n_violations": len(hit),
                 "examples": sorted(hit)[:5]})
# D_final must not appear in any fitted artefact
for f, label in (("configs/v3_2/split.json", "direction/calibration/budget split"),
                 ("configs/v3_3_2/final_split.json", "V3.3.2 final"),
                 ("results/v3_3_3/raw/behavioral_dose_response.csv", "behavioral fit")):
    p = pathlib.Path(f)
    if not p.exists():
        continue
    txt = p.read_text()
    hit = [x for x in final if x in txt]
    rows.append({"check": f"D_final ids in {label}", "n_violations": len(hit),
                 "examples": hit[:5]})
df = pd.DataFrame(rows); df.to_csv(TAB / "leakage_audit.csv", index=False)
print(df.to_string(index=False))
tot = int(df.n_violations.sum())
print(f"\nTOTAL VIOLATIONS: {tot}")
if tot:
    raise SystemExit("LEAKAGE DETECTED")
