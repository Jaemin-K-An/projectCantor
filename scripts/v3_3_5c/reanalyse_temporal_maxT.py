"""Correct the V3.3.5b prompt-paired max-T analysis without regeneration.

The V3.3.5b script sampled prompts separately inside each B2 loop.  Here one
IDX[bootstrap replicate, prompt] matrix is constructed and reused for every
budget and contrast.  A bootstrap replicate consequently represents the same
resampled prompt cluster throughout the repeated-measures family.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fix_temporal_classifier import classify_temporal  # noqa: E402


DEFAULT_DATA = pathlib.Path(
    "results/v3_3_5b/raw/temporal_D_temporal_confirm.csv"
)
DEFAULT_PROTOCOL = pathlib.Path("configs/v3_3_5b/protocol_stageA.json")
DEFAULT_TABLE = pathlib.Path("results/v3_3_5c/tables/temporal_contrasts_corrected.csv")
DEFAULT_RESULT = pathlib.Path("results/v3_3_5c/tables/temporal_correction.json")


@dataclass(frozen=True)
class MaxTResult:
    table: pd.DataFrame
    critical_value: float
    bootstrap_indices: np.ndarray
    bootstrap_statistics: np.ndarray
    prompt_ids: tuple[str, ...]


def shared_prompt_index_matrix(n_boot: int, n_prompts: int, seed: int) -> np.ndarray:
    """Return the one and only prompt resampling matrix for the family."""
    if n_boot < 1 or n_prompts < 2:
        raise ValueError("n_boot >= 1 and n_prompts >= 2 are required")
    return np.random.default_rng(seed).integers(
        0, n_prompts, size=(n_boot, n_prompts), dtype=np.int32
    )


def _complete_prompt_ids(df: pd.DataFrame, budgets: list[float], contrasts: list[list[str]]) -> tuple[str, ...]:
    required = {(float(b), s) for b in budgets for pair in contrasts for s in pair}
    by_prompt = df.groupby("pid", sort=True)
    complete: list[str] = []
    for pid, frame in by_prompt:
        observed = set(zip(frame.B2_target.astype(float), frame.schedule.astype(str)))
        if required <= observed:
            complete.append(str(pid))
    if not complete:
        raise ValueError("no prompt has all required budget/contrast measurements")
    return tuple(complete)


def corrected_max_t(
    df: pd.DataFrame,
    *,
    contrasts: list[list[str]],
    sesoi: float,
    n_boot: int = 20_000,
    seed: int = 7,
) -> MaxTResult:
    """Compute simultaneous paired intervals with shared prompt resampling.

    Effects are refusal reductions relative to baseline.  For a contrast
    distributed-minus-single the baseline cancels, leaving
    refusal(single) - refusal(distributed), evaluated within prompt.
    """
    positive = df[df.B2_target > 0].copy()
    budgets = sorted(float(x) for x in positive.B2_target.unique())
    prompt_ids = _complete_prompt_ids(positive, budgets, contrasts)
    positive = positive[positive.pid.astype(str).isin(prompt_ids)]
    wide = positive.pivot_table(
        index="pid", columns=["B2_target", "schedule"], values="refusal", aggfunc="first"
    ).reindex(prompt_ids)

    idx = shared_prompt_index_matrix(n_boot, len(prompt_ids), seed)
    records: list[dict] = []
    boot_columns: list[np.ndarray] = []
    for budget in budgets:
        for distributed, single in contrasts:
            d = (
                wide[(budget, single)].to_numpy(float)
                - wide[(budget, distributed)].to_numpy(float)
            )
            if not np.isfinite(d).all():
                raise ValueError(f"missing paired observation at B2={budget}: {distributed}/{single}")
            boot = d[idx].mean(axis=1)
            estimate = float(d.mean())
            se = float(np.std(boot - estimate, ddof=1))
            records.append(
                {
                    "B2": budget,
                    "distributed": distributed,
                    "single": single,
                    "contrast": f"{distributed} - {single}",
                    "mean_diff": estimate,
                    "pointwise_lo": float(np.quantile(boot, 0.025)),
                    "pointwise_hi": float(np.quantile(boot, 0.975)),
                    "bootstrap_se": se,
                }
            )
            boot_columns.append(boot)

    boot_matrix = np.column_stack(boot_columns)
    estimates = np.asarray([r["mean_diff"] for r in records])
    ses = np.asarray([r["bootstrap_se"] for r in records])
    safe_ses = np.where(ses > 0, ses, np.inf)
    studentized = (boot_matrix - estimates[None, :]) / safe_ses[None, :]
    max_abs_t = np.max(np.abs(studentized), axis=1)
    critical = float(np.quantile(max_abs_t, 0.95))

    table = pd.DataFrame(records)
    table["simult_lo"] = table.mean_diff - critical * table.bootstrap_se
    table["simult_hi"] = table.mean_diff + critical * table.bootstrap_se
    table["significant"] = (table.simult_lo > 0) | (table.simult_hi < 0)
    table["material_positive"] = table.simult_lo > float(sesoi)
    table["material_negative"] = table.simult_hi < -float(sesoi)
    return MaxTResult(table, critical, idx, boot_matrix, prompt_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=pathlib.Path, default=DEFAULT_DATA)
    parser.add_argument("--protocol", type=pathlib.Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--table", type=pathlib.Path, default=DEFAULT_TABLE)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_RESULT)
    parser.add_argument("--n-boot", type=int, default=20_000)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    raw = pd.read_csv(args.data)
    analysed = raw[raw.coherence >= float(protocol["coherence_gate"])].copy()
    bad = analysed[
        (analysed.B2_target > 0)
        & (np.abs(analysed.B2_realised / analysed.B2_target - 1) > 0.03)
    ]
    result = corrected_max_t(
        analysed,
        contrasts=protocol["primary_contrasts"],
        sesoi=float(protocol["sesoi_dP"]),
        n_boot=args.n_boot,
        seed=int(protocol["seeds"]["bootstrap"]),
    )
    verdict = classify_temporal(result.table, sesoi=float(protocol["sesoi_dP"]))

    args.table.parent.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(args.table, index=False)
    payload = {
        "source_data": str(args.data),
        "regeneration_performed": False,
        "n_prompts": len(result.prompt_ids),
        "n_boot": args.n_boot,
        "bootstrap_seed": int(protocol["seeds"]["bootstrap"]),
        "shared_index_shape": list(result.bootstrap_indices.shape),
        "shared_indices_across_budgets": True,
        "shared_indices_across_contrasts": True,
        "budget_mismatch_rows": int(len(bad)),
        "maxT_critical_value": result.critical_value,
        "sesoi": float(protocol["sesoi_dP"]),
        **verdict,
        "contrasts": result.table.to_dict(orient="records"),
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"prompts={len(result.prompt_ids)} bootstrap={args.n_boot}")
    print(f"shared IDX shape={result.bootstrap_indices.shape}")
    print(f"corrected max-T critical value={result.critical_value:.6f}")
    print(result.table[["B2", "contrast", "mean_diff", "simult_lo", "simult_hi"]].to_string(index=False))
    print(f"TEMPORAL_CORRECTION={verdict['verdict']}")


if __name__ == "__main__":
    main()
