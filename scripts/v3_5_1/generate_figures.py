"""Generate compact V3.5.1 calibration, budget, efficacy and utility figures."""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from cantor_guard_v351.one_sided_cantor import epsilon_r  # noqa: E402
from _common import FIGURES, RESULTS, read_json, write_json  # noqa: E402

BLUE, ORANGE, RED, GREEN, GREY = "#2563eb", "#f59e0b", "#dc2626", "#16a34a", "#64748b"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    cal = read_json(RESULTS / "tables/risk_conditional_window.json")
    W_R = float(cal["W_R"])
    risk = pd.read_csv(RESULTS / "cache/D_risk_cal_v351.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(risk.x_risk, bins=28, color=BLUE, alpha=.78)
    ax.axvline(W_R, color=RED, ls="--", lw=2, label=f"W_R={W_R:.3f}")
    ax.set(title="Risk-conditional conformal window (d<0 only)", xlabel="x=-d", ylabel="count")
    ax.legend(); save(fig, "F351-01-risk-conditional-window.png")

    rho = np.linspace(.20, .49, 500)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rho, [epsilon_r(r, W_R) for r in rho], color=BLUE, lw=3)
    ax.axvline(1/3, color=RED, ls="--", label="unique maximum rho=1/3")
    ax.set(title="Direct terminal risk-policy-switch certificate", xlabel="rho", ylabel="epsilon_R(rho)")
    ax.legend(); save(fig, "F351-02-certificate-rho.png")

    budget = read_json(RESULTS / "tables/budget_calibration.json")["per_arm"]
    names = list(budget)
    risk_rms = [budget[name]["risk_q_rms"] for name in names]
    global_rms = [budget[name]["global_q_rms"] for name in names]
    x = np.arange(len(names)); fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x-.2, risk_rms, .4, label="risk-conditional RMS", color=BLUE)
    ax.bar(x+.2, global_rms, .4, label="global RMS", color=ORANGE)
    ax.axhline(.03, color=RED, ls="--", label="frozen risk target")
    ax.set_xticks(x, names, rotation=35, ha="right")
    ax.set(title="Matched risk intensity vs sparse global burden", ylabel="q RMS")
    ax.legend(); save(fig, "F351-03-risk-vs-global-budget.png")

    final = pd.read_csv(RESULTS / "raw/final_D_final_v351_harmful.csv")
    primary = final[(final.family == "ATTACK_V") & final.arm.isin(["ATTACK_ONLY", "LINEAR", "CANTOR_1/3"])]
    curves = primary.groupby(["arm", "epsilon"], sort=True).y_safe.mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for arm, group in curves.groupby("arm", sort=False):
        ax.plot(group.epsilon, group.y_safe, marker="o", ms=3, label=arm)
    ax.set_xscale("symlog", linthresh=.03)
    ax.set(title="ATTACK_V frozen refusal response", xlabel="epsilon", ylabel="refusal rate", ylim=(-.02, 1.02))
    ax.legend(); save(fig, "F351-04-primary-refusal-curves.png")

    contrast_rows = []
    for filename in ("controller_effect.json", "linear_comparison.json", "rho_family.json"):
        table = read_json(RESULTS / f"tables/{filename}")
        contrast_rows.extend(table["max_t"]["contrasts"])
    contrast_rows = [row for row in contrast_rows if row["arm"].startswith("ATTACK_V:")]
    fig, ax = plt.subplots(figsize=(9, max(4, .55*len(contrast_rows))))
    y = np.arange(len(contrast_rows))
    means = np.array([row["mean_difference"] for row in contrast_rows])
    lo = np.array([row["simultaneous_lo"] for row in contrast_rows])
    hi = np.array([row["simultaneous_hi"] for row in contrast_rows])
    labels = [f"{row['arm'].split(':',1)[1]} − {row['reference'].split(':',1)[1]}" for row in contrast_rows]
    ax.errorbar(means, y, xerr=[means-lo, hi-means], fmt="o", color=BLUE, capsize=3)
    ax.axvline(0, color=GREY); ax.axvline(.03, color=RED, ls="--", label="SESOI +.03")
    ax.set_yticks(y, labels); ax.set(title="ATTACK_V max-T simultaneous intervals", xlabel="AUC difference")
    ax.legend(); save(fig, "F351-05-primary-simultaneous-intervals.png")

    utility = read_json(RESULTS / "tables/utility.json")
    arms = list(utility["per_arm"])
    increases = [utility["per_arm"][arm]["false_refusal_increase"] for arm in arms]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(arms, increases, color=[GREEN if x <= .05 else RED for x in increases])
    ax.axhline(.05, color=RED, ls="--", label="utility gate")
    ax.tick_params(axis="x", rotation=35)
    ax.set(title="Benign false-refusal increase", ylabel="increase vs no controller")
    ax.legend(); save(fig, "F351-06-benign-utility.png")

    diag = read_json(RESULTS / "tables/diagnostics.json")["cantor_cell_and_control_occupancy"]
    cantor = {k: v for k, v in diag.items() if k.startswith("CANTOR_")}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(list(cantor), [v["guard_fraction"] for v in cantor.values()], label="guard", color=ORANGE)
    ax.bar(list(cantor), [v["outside_risk_window_fraction"] for v in cantor.values()],
           bottom=[v["guard_fraction"] for v in cantor.values()], label="outside", color=RED)
    ax.tick_params(axis="x", rotation=35)
    ax.set(title="Cantor guard and outside-window occupancy", ylabel="fraction")
    ax.legend(); save(fig, "F351-07-policy-occupancy.png")

    captions = {
        "F351-01": "First 200 d<0 clean states; safe zero mass excluded.",
        "F351-02": "Residual-L2 direct terminal risk-policy-switch certificate only.",
        "F351-03": "Risk RMS is matched; global RMS reports sparse deployment burden.",
        "F351-04": "Frozen external refusal endpoint; not semantic safety.",
        "F351-05": "Shared 20,000 prompt bootstrap with max-T simultaneous intervals.",
        "F351-06": "Benign behavioral-refusal utility diagnostic.",
        "F351-07": "Post-confirmatory occupancy diagnostic; not used for tuning.",
    }
    write_json(FIGURES / "CAPTIONS.json", captions)
    print("generated 7 V3.5.1 figures")


if __name__ == "__main__":
    main()
