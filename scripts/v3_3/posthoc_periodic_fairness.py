"""V3.3 POST-HOC fairness correction -- NOT part of the sealed pre-analysis.

Labelled post-hoc, run after the sealed verdict, and it does not replace it.

THE ERROR IT CORRECTS. The sealed certificate model put `periodic_procedural`
in the enumerative branch, giving it Theta(2^n) proof obligations, while
simultaneously crediting it with an O(1) point query. That is inconsistent, and
it under-credits a CONTROL -- i.e. it errs in Cantor's favour, which is the
direction that must never go unchecked.

`L3_periodic` orders the gaps by (level, position) and lays them out with a
fixed survivor spacing. The prefix sum of widths up to level k is a closed form,

    S_k = sum_{i=1..k} 2^(i-1) * 3^-i

so the level block containing r is found by at most n comparisons and the index
within the block by one division. The address map is therefore O(n) closed form
with O(1) storage, exactly like Cantor's ternary descent, and its structural
properties admit the same cheap symbolic treatment.

This script recounts periodic's obligations on that basis and re-runs the
sealed classifier's logic to see whether the verdict survives.
"""
import sys, json, pathlib, copy
sys.path.insert(0, "llm/src")
sys.path.insert(0, "scripts/v3_3")
from final_claim_check_v33 import classify, PRIMARY, STRONG_CONTROLS, CANTOR

BENCH = pathlib.Path("results/v3_3/tables/benchmark_summary.json")
bench = json.loads(BENCH.read_text())
safety = json.loads(pathlib.Path("results/v3_3/tables/safety_gate.json").read_text())

corrected = copy.deepcopy(bench)
for n_s, fams in corrected["metrics"].items():
    n = int(n_s)
    if "periodic_procedural" not in fams:
        continue
    # Same accounting the inductive scheme gets: n per closed-form law,
    # 1 for directionality, 2 for support, n+1 for address soundness.
    # P6 (cross-scale identity) is NOT granted -- periodic does not have it.
    fams["periodic_procedural"]["M2_certificate_assertions"] = 4 * n + 1 + 2 + (n + 1)

v_sealed = json.loads(pathlib.Path("results/v3_3/tables/verdict_v33.json").read_text())
v2, why2, detail2 = classify(corrected, safety)

n = str(bench["deepest_n"])
print(f"--- primary metrics at n={n}, periodic recounted ---")
print(f"{'family':26s} {'M1 bits':>10s} {'M2 sealed':>11s} {'M2 corrected':>13s} {'M3':>8s}")
for f in bench["families"]:
    a = bench["metrics"][n][f]
    b = corrected["metrics"][n][f]
    print(f"{f:26s} {a['M1_canonical_bits']:10d} "
          f"{a['M2_certificate_assertions']:11d} "
          f"{b['M2_certificate_assertions']:13d} "
          f"{a['M3_point_query_words']:8d}")

print(f"\nsealed verdict    : {v_sealed['verdict']}")
print(f"post-hoc verdict  : {v2}")
print(f"  {why2}")
print("\nbeats_all (corrected):", json.dumps(detail2["beats_all_strong_controls"]))

# What actually separates recursive from non-recursive, after the correction?
st = pathlib.Path("results/v3_3/tables/bench_scale_transfer.csv")
if st.exists():
    import pandas as pd
    t = pd.read_csv(st)
    deep = t[t.n == t.n.max()]
    print("\nexact scale transfer at deepest n (the property periodic LACKS):")
    for _, r in t.groupby("family").E_scale_rel.max().items():
        pass
    print(t.groupby("family").agg(
        max_E_scale_rel=("E_scale_rel", "max"),
        new_params=("new_parameters", "max"),
        zero_shot=("zero_shot", "all")).to_string())

out = {"POSTHOC": True, "sealed_verdict": v_sealed["verdict"],
       "posthoc_verdict": v2, "reason": why2, "detail": detail2,
       "correction": "periodic_procedural recounted as closed-form addressable"}
pathlib.Path("results/v3_3/tables/posthoc_periodic_fairness.json").write_text(
    json.dumps(out, indent=2))
print("\nPOST-HOC. Does not replace the sealed verdict.")
