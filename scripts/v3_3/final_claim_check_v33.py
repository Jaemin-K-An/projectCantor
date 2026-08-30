"""V3.3 AUTOMATIC CLAIM CLASSIFIER -- frozen before the benchmarks are run.

The wording of the structural conclusion is decided by these rules, not by
whoever reads the tables. The rules are written to make the WEAK verdicts easy
to reach and the strong one hard, because the failure mode this programme
guards against is dressing a null result as a positive one.

Two gates fire before any structural comparison:

  * SAFETY GATE. If Cantor is not safety-equivalent to the controls within the
    SESOI, no structure-positive claim is available at all (STOP F). V3.2's
    Model A result supplies this, and it is inherited only if the symbolic
    evaluator matches the explicit one (STOP D).
  * CONTROL GATE. Seeded procedural controls and a recursive non-Cantor control
    must both be present (STOP B, STOP C). Without them the comparison cannot
    tell "Cantor is special" from "procedural is special" or "recursion is
    special".

Verdicts:
  A_CANTOR_SPECIFIC_STRUCTURAL_ADVANTAGE
  B_RECURSIVE_SELF_SIMILARITY_ADVANTAGE
  C_DESCRIPTION_ONLY_ADVANTAGE
  D_CERTIFICATION_ONLY_ADVANTAGE
  E_NO_STRUCTURAL_ADVANTAGE
  F_INCONCLUSIVE
"""
from __future__ import annotations
import sys, json, hashlib, pathlib, argparse

sys.path.insert(0, "llm/src")

CANTOR = "cantor_recursive"
RECURSIVE_NON_CANTOR = "recursive_non_cantor"
# Strong controls. Seeded procedural entries are what make a description-length
# claim non-trivial; the recursive non-Cantor entry is what separates A from B.
STRONG_CONTROLS = ["shuffled_seeded", "center_anchored_seeded",
                   "periodic_procedural"]
NON_RECURSIVE = STRONG_CONTROLS + ["shuffled_explicit",
                                   "learned_minimax_explicit"]

# PRIMARY structural metrics (harness section 59). Lower is better for all.
PRIMARY = {
    "M1_canonical_bits": "canonical symbolic description length (bits)",
    "M2_certificate_assertions": "proof obligations to certify the controller",
    "M3_point_query_words": "resident words required for one point query",
}
SECONDARY = ["serialized_bytes", "gzip_bytes", "ast_nodes",
             "construction_seconds", "verification_seconds",
             "materialised_words"]
# A metric only counts as an advantage if it improves by more than this
# relative margin at the deepest benchmarked n (guards against ties).
MARGIN = 0.05


def _strictly_better(a: float, b: float) -> bool:
    return a < b * (1.0 - MARGIN)


def _no_worse(a: float, b: float) -> bool:
    return a <= b * (1.0 + MARGIN)


def classify(bench: dict, safety: dict) -> tuple[str, str, dict]:
    detail: dict = {}

    if not safety.get("equivalent", False):
        return ("F_INCONCLUSIVE",
                "safety equivalence not established, so no performance-neutral "
                "structural claim is available (STOP F)", detail)
    present = set(bench["families"])
    missing_strong = [c for c in STRONG_CONTROLS if c not in present]
    if missing_strong:
        return ("F_INCONCLUSIVE",
                f"strong procedural controls missing: {missing_strong} "
                "(STOP B)", detail)
    if RECURSIVE_NON_CANTOR not in present:
        return ("F_INCONCLUSIVE",
                "recursive non-Cantor control missing; cannot separate a "
                "Cantor effect from a recursion effect (STOP C)", detail)

    n = str(bench["deepest_n"])
    M = bench["metrics"][n]              # family -> metric -> value

    # --- Cantor against each strong control, per primary metric -------------
    wins: dict[str, list[str]] = {}
    losses: dict[str, list[str]] = {}
    for m in PRIMARY:
        wins[m], losses[m] = [], []
        for c in STRONG_CONTROLS:
            if _strictly_better(M[CANTOR][m], M[c][m]):
                wins[m].append(c)
            elif not _no_worse(M[CANTOR][m], M[c][m]):
                losses[m].append(c)
    detail["wins"], detail["losses"] = wins, losses

    beats_all = {m: len(wins[m]) == len(STRONG_CONTROLS) for m in PRIMARY}
    loses_any = {m: len(losses[m]) > 0 for m in PRIMARY}
    detail["beats_all_strong_controls"] = beats_all
    detail["loses_to_some_control"] = loses_any

    # --- is the advantage Cantor's, or recursion's? -------------------------
    rec_matches_cantor = {
        m: _no_worse(M[RECURSIVE_NON_CANTOR][m], M[CANTOR][m])
        and _no_worse(M[CANTOR][m], M[RECURSIVE_NON_CANTOR][m])
        for m in PRIMARY}
    rec_beats_nonrecursive = {
        m: all(_strictly_better(M[RECURSIVE_NON_CANTOR][m], M[c][m])
               for c in STRONG_CONTROLS if c in M)
        for m in PRIMARY}
    detail["recursive_non_cantor_matches_cantor"] = rec_matches_cantor
    detail["recursive_non_cantor_beats_nonrecursive"] = rec_beats_nonrecursive

    # --- does the advantage persist with depth? -----------------------------
    persists = {}
    for m in PRIMARY:
        ok = True
        for nn in bench["depths"][-3:]:
            Mn = bench["metrics"][str(nn)]
            ok &= all(_strictly_better(Mn[CANTOR][m], Mn[c][m])
                      for c in STRONG_CONTROLS)
        persists[m] = ok
    detail["persists_with_depth"] = persists

    any_primary_win = any(beats_all[m] and persists[m] for m in PRIMARY)
    cantor_strictly_better_than_recursive = any(
        _strictly_better(M[CANTOR][m], M[RECURSIVE_NON_CANTOR][m])
        for m in PRIMARY)

    if any_primary_win and cantor_strictly_better_than_recursive \
            and not any(loses_any.values()):
        return ("A_CANTOR_SPECIFIC_STRUCTURAL_ADVANTAGE",
                "Cantor strictly beats every strong procedural control on a "
                "primary structural metric, the advantage persists with depth, "
                "and it is not matched by the recursive non-Cantor control",
                detail)

    both_recursive_dominate = any(
        rec_beats_nonrecursive[m] and beats_all[m] and persists[m]
        for m in PRIMARY)
    if both_recursive_dominate:
        return ("B_RECURSIVE_SELF_SIMILARITY_ADVANTAGE",
                "Cantor and the recursive non-Cantor control both beat the "
                "non-recursive layouts on a primary structural metric, and "
                "Cantor is not separately better -- the advantage belongs to "
                "recursive self-similarity, of which Cantor is one instance",
                detail)

    desc_only = (beats_all["M1_canonical_bits"]
                 and not beats_all["M2_certificate_assertions"])
    cert_only = (beats_all["M2_certificate_assertions"]
                 and not beats_all["M1_canonical_bits"])
    if cert_only:
        return ("D_CERTIFICATION_ONLY_ADVANTAGE",
                "the advantage appears in certification cost only; the "
                "canonical description is not shorter than the seeded "
                "procedural controls", detail)
    if desc_only:
        return ("C_DESCRIPTION_ONLY_ADVANTAGE",
                "the advantage appears in description length only", detail)
    return ("E_NO_STRUCTURAL_ADVANTAGE",
            "no primary structural metric shows a robust Cantor advantage over "
            "the strong procedural controls", detail)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="results/v3_3/tables/benchmark_summary.json")
    ap.add_argument("--safety", default="results/v3_3/tables/safety_gate.json")
    ap.add_argument("--out", default="results/v3_3/tables/verdict_v33.json")
    ap.add_argument("--allow-unsealed", action="store_true")
    a = ap.parse_args()

    me = pathlib.Path(__file__)
    my_sha = hashlib.sha256(me.read_bytes()).hexdigest()
    seal = pathlib.Path("configs/v3_3/PRE_ANALYSIS_FREEZE.json")
    if seal.exists():
        sealed = json.loads(seal.read_text()).get("classifier_sha256")
        if sealed and sealed != my_sha and not a.allow_unsealed:
            raise SystemExit(f"CLASSIFIER MODIFIED SINCE FREEZE\n  sealed {sealed}"
                             f"\n  actual {my_sha}\nRefusing to emit a verdict.")

    bench = json.loads(pathlib.Path(a.bench).read_text())
    safety = json.loads(pathlib.Path(a.safety).read_text())
    v, why, detail = classify(bench, safety)
    out = {"verdict": v, "reason": why, "classifier_sha256": my_sha,
           "safety_gate": safety, "detail": detail,
           "deepest_n": bench["deepest_n"]}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n########  V3.3 VERDICT: {v}  ########\n{why}\n")
    print(json.dumps(detail, indent=2)[:2400])
