"""V3.2 PHASE 12 -- how many independent goals does the final test need?

The precision of a goal-clustered estimate is governed by the between-goal SD,
which the V3.1 re-analysis measured directly. This script turns that measured
SD into a required number of goals, separately for the decisive (matched
control) comparisons and the weaker baseline comparisons, and it is run BEFORE
the freeze so the design is fixed by the answer rather than by the result.
"""
import sys, json, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, "llm/src")

SESOI = 0.03
aud = pd.read_csv("results/v3_2/tables/v31_pseudoreplication_audit.csv")
aud = aud[aud.status == "OK"]

MATCHED = ["T4_periodic", "T5_shuffled", "T6_center_anchored", "T3_wide_central"]
BASELINE = ["T0_none", "T1_true_constant", "T2_global_smooth"]

sd_matched = aud[aud.family.isin(MATCHED)].between_goal_sd.max()
sd_baseline = aud[aud.family.isin(BASELINE)].between_goal_sd.max()

def halfwidth(sd, n):        # normal approximation to the cluster bootstrap
    return 1.96 * sd / np.sqrt(n)

def n_for_equivalence(sd, margin=SESOI, safety=2.0):
    """Goals needed for the CI to sit inside +-margin with room to spare.
    `safety` demands the half-width be `safety` times smaller than the margin,
    so equivalence is not decided by a hair."""
    n = 4
    while halfwidth(sd, n) > margin / safety and n < 5000:
        n += 1
    return n

def n_for_detection(sd, effect, power=0.80):
    """Goals needed to detect `effect` at alpha=.05 with the stated power."""
    from math import ceil
    z_a, z_b = 1.959964, 0.8416212
    return int(ceil(((z_a + z_b) * sd / effect) ** 2))

rows = []
for label, sd in [("matched controls", sd_matched), ("weak baselines", sd_baseline)]:
    for n in (12, 20, 24, 32, 40, 60):
        rows.append({"stratum": label, "sd": round(sd, 5), "n_goals": n,
                     "half_width": round(halfwidth(sd, n), 5),
                     "inside_sesoi": bool(halfwidth(sd, n) < SESOI)})
grid = pd.DataFrame(rows)

plan = {
    "sesoi": SESOI,
    "between_goal_sd_matched": round(float(sd_matched), 5),
    "between_goal_sd_baseline": round(float(sd_baseline), 5),
    "n_for_equivalence_matched": n_for_equivalence(sd_matched),
    "n_to_detect_0.018_baseline": n_for_detection(sd_baseline, 0.018),
    "n_to_detect_0.030_baseline": n_for_detection(sd_baseline, 0.030),
    "n_to_detect_0.010_matched": n_for_detection(sd_matched, 0.010),
}
N = 50
plan["n_goals_chosen"] = N

def power_at(sd, effect, n):
    from math import sqrt, erf
    z = effect * sqrt(n) / sd - 1.959964
    return 0.5 * (1 + erf(z / sqrt(2)))

plan["power_at_chosen"] = {
    "matched_+0.010": round(power_at(sd_matched, 0.010, N), 3),
    "baseline_+0.030": round(power_at(sd_baseline, 0.030, N), 3),
    "baseline_+0.018": round(power_at(sd_baseline, 0.018, N), 3),
}
plan["half_width_at_chosen"] = {
    "matched": round(halfwidth(sd_matched, N), 5),
    "baseline": round(halfwidth(sd_baseline, N), 5),
}
plan["rationale"] = (
    f"{N} goals is the whole of D_test, the largest held-out block the 100-goal "
    "JBB pool allows after reserving direction/calibration/budget/dev. It puts "
    f"the matched-control half-width at {halfwidth(sd_matched, N):.4f}, i.e. "
    f"{SESOI / halfwidth(sd_matched, N):.1f}x inside the SESOI, so the DECISIVE "
    "equivalence comparisons are comfortably powered."
)
plan["stated_limitation"] = (
    "The design is NOT fully powered for the weak-baseline stratum. Detecting "
    f"the +0.018 cantor-vs-true-constant effect at 80% power needs "
    f"{plan['n_to_detect_0.018_baseline']} goals; at {N} the power is "
    f"{plan['power_at_chosen']['baseline_+0.018']:.0%}. This is registered in "
    "advance: a null result on that secondary comparison will be reported as "
    "INCONCLUSIVE (underpowered), never as evidence of equivalence."
)
pathlib.Path("results/v3_2/tables").mkdir(parents=True, exist_ok=True)
grid.to_csv("results/v3_2/tables/power_grid.csv", index=False)
pathlib.Path("results/v3_2/tables/power_plan.json").write_text(json.dumps(plan, indent=2))
print(grid.to_string(index=False))
print()
print(json.dumps(plan, indent=2))
