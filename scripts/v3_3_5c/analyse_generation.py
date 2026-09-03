"""Prompt-level robustness AUC and shared-index max-T rho comparisons."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from _common import CONFIG, RESULTS, behavioral_protocol, read_json, rho_key, write_json


def paired_max_t(differences: np.ndarray, *, n_boot: int, seed: int) -> dict:
    """One IDX matrix for every contrast; rows are prompts, columns contrasts."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("differences must be [prompt, contrast]")
    idx = np.random.default_rng(seed).integers(
        0, values.shape[0], size=(n_boot, values.shape[0]), dtype=np.int32
    )
    estimates = values.mean(axis=0)
    bootstrap = values[idx].mean(axis=1)
    se = np.std(bootstrap - estimates[None, :], axis=0, ddof=1)
    studentized = (bootstrap - estimates[None, :]) / np.where(se > 0, se, np.inf)
    critical = float(np.quantile(np.max(np.abs(studentized), axis=1), 0.95))
    return {
        "estimates": estimates, "se": se, "critical": critical,
        "lo": estimates - critical * se, "hi": estimates + critical * se,
        "indices": idx, "bootstrap": bootstrap,
    }


def prompt_auc(frame: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    rows = []
    for (pid, rho), group in frame.groupby(["pid", "rho"]):
        group = group.sort_values("epsilon")
        x = group.epsilon.to_numpy(float)
        if x[-1] <= x[0]:
            raise ValueError("attack grid must contain positive range")
        x = (x - x[0]) / (x[-1] - x[0])
        harmful_auc = float(np.trapezoid(group[endpoint].to_numpy(float), x))
        rows.append({"pid": pid, "rho": float(rho), "harmful_auc": harmful_auc, "robustness_auc": 1 - harmful_auc})
    return pd.DataFrame(rows)


def main() -> None:
    protocol = behavioral_protocol()
    controller = read_json(CONFIG / "controller.json")
    evaluator = read_json(CONFIG / "evaluator.json")
    budgets = read_json(RESULTS / "tables/final_budget_audit.json")
    utility_path = RESULTS / "tables/utility.json"
    utility = read_json(utility_path) if utility_path.exists() else {"verdict": "U3_NOT_RUN"}
    frame = pd.read_csv(RESULTS / "tables/final_generation_scored.csv")
    endpoint = "semantic_harmful_compliance" if evaluator["status"] == "S1_VALID" else "harmful_compliance_proxy"
    auc = prompt_auc(frame, endpoint)
    auc.to_csv(RESULTS / "tables/prompt_robustness_auc.csv", index=False)
    pivot = auc.pivot(index="pid", columns="rho", values="robustness_auc")
    cantor_col = min(pivot.columns, key=lambda value: abs(value - 1 / 3))
    comparators = [float(x) for x in controller["primary_comparators"] + controller["secondary_comparators"]]
    cols = [min(pivot.columns, key=lambda value: abs(value - rho)) for rho in comparators]
    pivot = pivot.dropna(subset=[cantor_col, *cols])
    differences = np.column_stack([(pivot[cantor_col] - pivot[col]).to_numpy() for col in cols])
    result = paired_max_t(
        differences,
        n_boot=int(protocol["bootstrap_replicates"]),
        seed=int(protocol["bootstrap_seed"]),
    )
    rows = []
    for i, (requested, actual) in enumerate(zip(comparators, cols)):
        rows.append({
            "contrast": f"Cantor - rho={rho_key(requested)}",
            "comparator": requested, "actual_column": float(actual),
            "mean_diff": float(result["estimates"][i]),
            "bootstrap_se": float(result["se"][i]),
            "simult_lo": float(result["lo"][i]), "simult_hi": float(result["hi"][i]),
            "primary": requested in [float(x) for x in controller["primary_comparators"]],
        })
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(RESULTS / "tables/generation_maxT_contrasts.csv", index=False)
    sesoi = float(controller["semantic_auc_sesoi"])
    primary = contrasts[contrasts.primary]
    if evaluator["status"] != "S1_VALID":
        verdict = "G4_PROXY_ONLY"
    elif budgets["status"] != "BUDGET_MATCHED" or utility["verdict"] != "U1_PASS":
        verdict = "G5_INCONCLUSIVE"
    elif len(primary) and (primary.simult_lo > sesoi).all():
        verdict = "G1_CANTOR_SEMANTIC_GAIN"
    elif len(primary) and (primary.simult_hi < -sesoi).any():
        verdict = "G3_OTHER_RHO_BETTER"
    elif len(primary) and (primary.simult_lo >= -sesoi).all() and (primary.simult_hi <= sesoi).all():
        verdict = "G2_RHO_FAMILY_PRACTICALLY_EQUIVALENT"
    else:
        verdict = "G5_INCONCLUSIVE"
    index_hash = hashlib.sha256(result["indices"].tobytes()).hexdigest()
    write_json(RESULTS / "tables/generation_analysis.json", {
        "verdict": verdict, "endpoint": endpoint,
        "n_prompts": int(len(pivot)), "n_boot": int(protocol["bootstrap_replicates"]),
        "bootstrap_seed": int(protocol["bootstrap_seed"]),
        "shared_prompt_index_matrix_sha256": index_hash,
        "same_indices_all_rho_contrasts": True,
        "maxT_critical_value": result["critical"], "sesoi": sesoi,
        "contrasts": rows,
    })
    print(contrasts.to_string(index=False))
    print(verdict)


if __name__ == "__main__":
    main()
