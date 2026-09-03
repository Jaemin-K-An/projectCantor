"""Audit/validate a semantic harmful-compliance evaluator before its use."""
from __future__ import annotations

import argparse

import pandas as pd

import sys
sys.path.insert(0, "llm/src")
from cantor_guard_v335c.semantic_eval import validate_predictions  # noqa: E402

from _common import CONFIG, RESULTS, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-csv", help="Independent labelled CSV with y_true and y_pred")
    parser.add_argument("--name", help="Evaluator identifier/revision for supplied predictions")
    args = parser.parse_args()
    config = read_json(CONFIG / "evaluator.json")
    prior = read_json("results/v3_2/tables/evaluator_validation.json")
    audit = {
        "options_audited": config["preferred_hierarchy"],
        "existing_repository_evidence": {
            "LEX32": {
                "accuracy": prior["accuracy"]["LEX32"],
                "limitation": "in-sample lexical refusal control set; not harmful-compliance semantics",
            },
            "protectai/distilroberta-base-rejection-v1": {
                "accuracy": prior["accuracy"]["EXT"],
                "limitation": "below 0.85 and refusal-only",
            },
        },
    }
    if args.validation_csv:
        if not args.name:
            raise SystemExit("--name is required with --validation-csv")
        frame = pd.read_csv(args.validation_csv)
        required = {"y_true", "y_pred", "class_name"}
        if not required <= set(frame.columns):
            raise ValueError(f"validation CSV must contain {sorted(required)}")
        needed_classes = set(config["validation_classes"])
        if not needed_classes <= set(frame.class_name):
            raise ValueError("validation set lacks one or more required semantic classes")
        metrics = validate_predictions(
            frame.y_true, frame.y_pred,
            threshold=float(config["minimum_balanced_accuracy"]),
        )
        audit.update({"evaluator": args.name, "validation": metrics.as_dict()})
        if metrics.valid:
            config.update({"selected": args.name, "status": "S1_VALID", "reason": "independent semantic validation gate passed"})
        else:
            config.update({"selected": args.name, "status": "S2_PROXY_ONLY", "reason": "independent semantic validation gate failed"})
        write_json(CONFIG / "evaluator.json", config)
    else:
        audit.update({
            "evaluator": None,
            "validation": None,
            "verdict": "S2_PROXY_ONLY",
            "reason": config["reason"],
        })
    audit["verdict"] = config["status"]
    write_json(RESULTS / "tables/semantic_evaluator_validation.json", audit)
    print(audit["verdict"])
    print(config["reason"])


if __name__ == "__main__":
    main()
