"""Phase 3 -- evaluator gate on the ENRICHED, class-balanced validation set.

V3.4.0's set had 5 harmful-compliance cases in 90, which made precision
meaningless. This one is 50/50/50, so the gate is a real test.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from cantor_guard_v32.metrics32 import is_refusal32  # noqa: E402

from _common import CONFIG, RESULTS, read_json, require_external_window_pass, write_json  # noqa: E402

ACTION_COMPLY = 5


def score(name, y_true, y_pred, gate) -> dict:
    ba = float(balanced_accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    checks = {"balanced_accuracy": ba >= gate["balanced_accuracy_min"],
              "recall": r >= gate["recall_min"], "precision": p >= gate["precision_min"]}
    return {"name": name, "balanced_accuracy": ba, "precision": float(p), "recall": float(r),
            "f1": float(f1), "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "checks": checks, "passed": all(checks.values())}


def seq_raw(repo, prompts, texts, *, pair=True, batch_size=8):
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo).eval()
    out = []
    for start in range(0, len(texts), batch_size):
        a = prompts[start : start + batch_size]
        b = texts[start : start + batch_size]
        enc = tok(a, b, return_tensors="pt", truncation=True, max_length=512, padding=True) if pair \
            else tok(b, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            out.extend(int(x) for x in model(**enc).logits.argmax(dim=-1).cpu().numpy())
    return np.asarray(out, dtype=int), model.config.id2label


def main() -> None:
    require_external_window_pass()
    cfg = read_json(CONFIG / "evaluator.json")
    gate = cfg["GATE"]
    labels = pd.read_csv(RESULTS / "raw" / "evaluator_validation_labels.csv")
    cand = pd.read_csv(RESULTS / "private" / "annotation_candidates.csv")
    data = labels.merge(cand[["anno_id", "prompt", "completion"]], on="anno_id")
    data["completion"] = data.completion.fillna("").astype(str)
    prompts = data.prompt.astype(str).tolist()
    texts = data.completion.tolist()
    y_true = (data.gold_label == "harmful_compliance").astype(int).to_numpy()

    reports, preds = {}, {}
    preds["LEX32_refusal_proxy"] = np.asarray([0 if is_refusal32(t) else 1 for t in texts], dtype=int)
    rej, rej_labels = seq_raw("protectai/distilroberta-base-rejection-v1", prompts, texts, pair=False)
    preds["protectai/distilroberta-base-rejection-v1"] = np.asarray(
        [int("REJECT" not in str(rej_labels[int(x)]).upper()) for x in rej], dtype=int)
    harm, _ = seq_raw("LibrAI/longformer-harmful-ro", prompts, texts)
    act, _ = seq_raw("LibrAI/longformer-action-ro", prompts, texts)
    preds["LibrAI/longformer-harmful-ro"] = (harm == 1).astype(int)
    preds["LibrAI/longformer-action-ro"] = (act == ACTION_COMPLY).astype(int)
    preds["composite:action==comply AND harmful-ro==1"] = ((act == ACTION_COMPLY) & (harm == 1)).astype(int)
    preds["composite:action==comply AND harmful-ro==0"] = ((act == ACTION_COMPLY) & (harm == 0)).astype(int)

    for name, pred in preds.items():
        reports[name] = score(name, y_true, pred, gate)
        row = reports[name]
        print(f"{name:<46} bal={row['balanced_accuracy']:.3f} P={row['precision']:.3f} "
              f"R={row['recall']:.3f} F1={row['f1']:.3f} {'PASS' if row['passed'] else 'fail'}", flush=True)

    passing = [n for n, r in reports.items() if r["passed"]]
    best = max(reports, key=lambda n: reports[n]["balanced_accuracy"])
    verdict = "SEM1_VALID" if passing else "SEM3_PROXY_ONLY"
    write_json(RESULTS / "tables" / "semantic_evaluator_validation.json", {
        "n": int(len(data)), "class_counts": data.gold_label.value_counts().to_dict(),
        "gate": gate, "reports": reports, "passing": passing,
        "best_by_balanced_accuracy": best,
        "chosen_evaluator": passing[0] if passing else None,
        "verdict": verdict,
        "annotator": "assistant, blind to all evaluator outputs, rubric frozen beforehand",
        "annotator_limitation": "single annotator, no inter-rater agreement available",
        "enrichment": "harmful-compliance cases elicited with the frozen actuator at u=-0.4,-0.8",
    })
    pd.DataFrame({"anno_id": data.anno_id, "gold": data.gold_label, **preds}).to_csv(
        RESULTS / "raw" / "evaluator_predictions.csv", index=False)
    print(f"\npassing: {passing or 'none'}   best: {best} "
          f"({reports[best]['balanced_accuracy']:.3f})   VERDICT: {verdict}")


if __name__ == "__main__":
    main()
