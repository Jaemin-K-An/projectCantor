import sys

sys.path.insert(0, "llm/src")
from cantor_guard_v335c.semantic_eval import RefusalProxyOnly, validate_predictions


def test_balanced_accuracy_gate_and_confusion_matrix():
    metrics = validate_predictions(
        [0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 0],
        threshold=0.85,
    )
    assert metrics.confusion_matrix == [[2, 1], [1, 2]]
    assert metrics.balanced_accuracy == 2 / 3
    assert not metrics.valid


def test_refusal_proxy_cannot_authorize_semantic_claim():
    evaluator = RefusalProxyOnly(lambda text: text.startswith("no"))
    assert evaluator.status() == "S2_PROXY_ONLY"
    assert not evaluator.is_semantic_harmful_compliance_evaluator
