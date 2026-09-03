"""Classify Cantor 1/3 versus the equal-budget nonrecursive linear baseline."""
from __future__ import annotations

from _common import CONFIG, RESULTS, read_json, require_confirmatory_freeze, write_json
from patch_claim_classifier import baseline_verdict


def main() -> None:
    require_confirmatory_freeze()
    effects = read_json(RESULTS / "tables/controller_effect.json")
    sesoi = float(read_json(CONFIG / "statistics.json")["SESOI"])
    by_family = {}
    for family, row in effects["by_family"].items():
        contrast = next((x for x in row["max_t"]["contrasts"]
                         if x["arm"] == "1/3" and x["reference"] == "LINEAR"), None)
        verdict = baseline_verdict(
            interval_lo=None if contrast is None else contrast["simultaneous_lo"],
            interval_hi=None if contrast is None else contrast["simultaneous_hi"],
            sesoi=sesoi,
        )
        by_family[family] = {"contrast": contrast, "verdict": verdict}
    values = {row["verdict"] for row in by_family.values()}
    verdict = values.pop() if len(values) == 1 else "BASE4_INCONCLUSIVE"
    write_json(RESULTS / "tables/linear_baseline.json", {
        "sesoi": sesoi,
        "by_family": by_family,
        "verdict": verdict,
    })
    print(verdict)


if __name__ == "__main__":
    main()
