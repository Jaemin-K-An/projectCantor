import json
import pathlib

RESULTS = pathlib.Path(__file__).resolve().parents[2] / "results/v3_4_0"


def test_orientation_and_prediction_stability_are_reported():
    s = json.loads((RESULTS / "tables" / "sensor_stability.json").read_text())
    for key in ("cosine_to_full_fit", "heldout_auroc", "decision_agreement_with_full_fit"):
        assert 0.0 <= s[key]["mean"] <= 1.0
        assert s[key]["ci95"][0] <= s[key]["mean"] <= s[key]["ci95"][1]


def test_per_coefficient_stability_is_not_claimed():
    """n=180 in 896 dimensions cannot identify individual weights (section 17)."""
    s = json.loads((RESULTS / "tables" / "sensor_stability.json").read_text())
    assert "per_coefficient" not in json.dumps(s).lower().replace("not reported", "")
    assert "note" in s and "unidentified" in s["note"]


def test_bootstrap_actually_ran():
    s = json.loads((RESULTS / "tables" / "sensor_stability.json").read_text())
    assert s["n_boot_effective"] >= 1000
