"""V3.3.1 POST-HOC correction -- NOT part of the sealed pre-analysis.

Labelled post-hoc, run after the sealed verdict, and it does not replace it.

THE GAP IT CLOSES. The sealed classifier promotes to
B_CANTOR_BALANCED_OPTIMUM_EMPIRICALLY_SUPPORTED whenever the empirical argmax
falls within 0.03 of 1/3. It never asks whether that argmax is DISTINGUISHABLE
from its neighbours. At the feasible depth the argmax is indeed rho = 1/3, but:

  * the whole spread across rho is 0.0058, a fifth of the SESOI
  * every goal-clustered CI contains zero
  * benign utility is IDENTICAL to four decimals at every rho

An argmax on a flat surface is not evidence. Reading it as support for
kappa = 1 would be exactly the "find an objective that makes Cantor win" move
this programme exists to avoid, so the honest verdict is the weaker one.
"""
import sys, json, pathlib
sys.path.insert(0, "scripts/v3_3_1")
import pandas as pd
from final_claim_check_v331 import classify

TAB = pathlib.Path("results/v3_3_1/tables")
theory = json.loads((TAB / "theory_gate.json").read_text())
emp = json.loads((TAB / "empirical_gate.json").read_text())
cmp_ = pd.read_csv(TAB / "rho_cluster_comparisons.csv")
sealed = json.loads((TAB / "verdict_v331.json").read_text())

any_sig = bool(cmp_.significant.any())
max_spread = float(max(v["spread"] for v in emp["per_depth"].values()))
ut = pd.read_csv("results/v3_3_1/raw/v331_llm_utility_qwen2.5-0.5b-instruct.csv")
util_spread = float(ut.groupby("rho").false_refusal.mean().max()
                    - ut.groupby("rho").false_refusal.mean().min())

print(f"any rho comparison significant : {any_sig}")
print(f"max spread across rho          : {max_spread:.4f}  (SESOI 0.03)")
print(f"benign false-refusal spread    : {util_spread:.4f}")

emp2 = dict(emp)
if not any_sig:
    # The empirical optimum is not identifiable; withhold it.
    emp2["rho_empirical_optimum"] = None
    emp2["withheld_reason"] = (
        "no goal-clustered comparison between rho values excludes zero, and the "
        f"entire spread is {max_spread:.4f} against a SESOI of 0.03, so the "
        "argmax is not distinguishable from any other rho")
v2, why2, detail2 = classify(theory, emp2)

print(f"\nsealed verdict   : {sealed['verdict']}")
print(f"post-hoc verdict : {v2}")
print(f"  {why2}")
out = {"POSTHOC": True, "sealed_verdict": sealed["verdict"],
       "posthoc_verdict": v2, "reason": why2,
       "any_significant": any_sig, "max_spread": max_spread,
       "utility_spread": util_spread, "detail": detail2,
       "correction": ("sealed rule promoted on argmax proximity alone, without "
                      "requiring the optimum to be distinguishable")}
(TAB / "posthoc_significance.json").write_text(json.dumps(out, indent=2))
print("\nPOST-HOC. Does not replace the sealed verdict.")
