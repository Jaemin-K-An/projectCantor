"""V3.4.0 is a historical experiment. Nothing in V3.4.0R may alter it."""
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "configs/v3_4_0r/V340_IMMUTABLE_MANIFEST.json").read_text())


def test_every_recorded_v340_artifact_is_byte_identical():
    changed = []
    for rel, digest in MANIFEST["sha256"].items():
        path = ROOT / rel
        assert path.exists(), f"V3.4.0 artifact disappeared: {rel}"
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            changed.append(rel)
    assert not changed, f"V3.4.0 artifacts modified: {changed}"


def test_manifest_covers_the_verdict_and_freeze():
    keys = set(MANIFEST["sha256"])
    assert "results/v3_4_0/tables/final_verdict.json" in keys
    assert "configs/v3_4_0/PRE_ANALYSIS_FREEZE.json" in keys
    assert "results/v3_4_0/tables/final_budget_audit.json" in keys
    assert MANIFEST["n_files"] >= 15


def test_v340_overall_verdict_still_reads_inconclusive():
    verdict = json.loads((ROOT / "results/v3_4_0/tables/final_verdict.json").read_text())
    assert verdict["OVERALL"] == "G_INCONCLUSIVE"


def test_the_erratum_exists_and_is_additive():
    doc = (ROOT / "docs/v3_4_0/V340_POSTHOC_AUDIT.md").read_text()
    assert "GEN6_EQUAL_BUDGET_COMPARISON_BLOCKED" in doc
    assert "V340_POSTHOC_AUDIT.md" not in MANIFEST["sha256"]
