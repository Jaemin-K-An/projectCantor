"""Generate the complete R01-R19 record without reading invalidated finals."""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
sys.path.insert(0, str(ROOT / "scripts/v3_4_0r"))
from _common import RESULTS, V340, frozen_sensor, read_json  # noqa: E402

FIG = ROOT / "figures/v3_4_0r"
BLUE, RED, GREEN, GREY, AMBER = "#2471a3", "#c0392b", "#239b56", "#7f8c8d", "#d68910"


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print(name)


def stopped(number: int, slug: str, title: str, detail: str = "ST3_WINDOW_SHIFT: confirmatory downstream stage not run"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((.05, .18), .90, .64, fc="#fdf2e9", ec=AMBER, lw=2))
    ax.text(.5, .62, title, ha="center", va="center", fontsize=16, weight="bold")
    ax.text(.5, .43, detail, ha="center", va="center", fontsize=11, color="#6e2c00", wrap=True)
    ax.text(.5, .27, "No q=.025 final result is plotted or analysed.", ha="center", fontsize=10)
    save(fig, f"R{number:02d}_{slug}.png")


def r01():
    old = read_json(V340 / "tables/sensor_vs_old_projection.json")
    transfer = read_json(RESULTS / "tables/sensor_transfer.json")
    vals = [old["old_projection"]["auroc"], old["new_sensor"]["auroc"], transfer["auroc"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(["old actuator\nprojection", "V3.4.0 sensor\nHarmfulQA", "frozen sensor\nexternal"],
           vals, color=[GREY, BLUE, GREEN])
    lo, hi = transfer["auroc_ci95"]
    ax.errorbar([2], [vals[2]], yerr=[[vals[2]-lo], [hi-vals[2]]], fmt="none", color="k", capsize=5)
    ax.set_ylim(.5, 1); ax.set_ylabel("AUROC"); ax.grid(axis="y", alpha=.3)
    ax.set_title("Historical V3.4.0 sensor and external discrimination")
    save(fig, "R01_historical_v340_sensor.png")


def r02():
    d = pd.read_csv(RESULTS / "raw/clean_D_sensor_transfer_r.csv")
    y, score = d.y_safe.to_numpy(int), d.d_clean.to_numpy(float)
    fpr, tpr, _ = roc_curve(y, score)
    precision, recall, _ = precision_recall_curve(y, score)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(fpr, tpr, color=BLUE, lw=2); axes[0].plot([0,1],[0,1], "--", color=GREY)
    axes[0].set(xlabel="false-positive rate", ylabel="true-positive rate", title="ROC (AUROC 0.899)")
    axes[1].plot(recall, precision, color=GREEN, lw=2)
    axes[1].set(xlabel="recall", ylabel="precision", title="PR (AUC 0.982)")
    for ax in axes: ax.grid(alpha=.3); ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    save(fig, "R02_external_sensor_roc_pr.png")


def r03():
    sensor = frozen_sensor()
    old_h = np.load(V340 / "cache/h_D_sensor_train.npy")
    old_d = np.asarray(sensor.distance(old_h), float)
    new_d = pd.read_csv(RESULTS / "raw/clean_D_sensor_transfer_r.csv").d_clean.to_numpy(float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(min(old_d.min(), new_d.min()), max(old_d.max(), new_d.max()), 30)
    ax.hist(old_d, bins=bins, density=True, alpha=.45, color=BLUE, label="HarmfulQA train")
    ax.hist(new_d, bins=bins, density=True, alpha=.45, color=GREEN, label="external")
    ax.axvline(0, color="k", ls="--", label="classifier hyperplane")
    ax.set(xlabel="signed sensor distance d0", ylabel="density", title="Sensor-score population shift")
    ax.legend(); ax.grid(alpha=.25)
    save(fig, "R03_sensor_score_shift.png")


def r04():
    d = pd.read_csv(RESULTS / "raw/clean_D_sensor_transfer_r.csv").d_clean.to_numpy(float)
    W = 2.2805212277347544
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(d, bins=25, color=BLUE, alpha=.75)
    ax.axvspan(-W, W, color=GREEN, alpha=.15, label="frozen window")
    ax.axvline(-W, color=GREEN, ls="--"); ax.axvline(W, color=GREEN, ls="--")
    ax.set(xlabel="d0", ylabel="prompts", title="Fixed-W coverage: 130/150 = 0.8667 < 0.90")
    ax.legend(); ax.grid(alpha=.25)
    save(fig, "R04_external_fixed_W_coverage.png")


def r05():
    old = read_json(V340 / "tables/final_budget_audit.json")
    arms = list(old["per_rho"])
    values = [old["per_rho"][a]["q_rms"] for a in arms]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(arms, values, color=GREY, label="V3.4.0 clean-calibrated final")
    ax.axhline(.03, color=RED, ls="--", label="frozen target .03")
    ax.text(.5, .92, "V3.4.0R attacked-state calibration not opened: ST3", transform=ax.transAxes,
            ha="center", color="#6e2c00", bbox=dict(fc="#fdf2e9", ec=AMBER))
    ax.set(xlabel="rho", ylabel="q_rms", title="Historical budget defect and stopped repair")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    save(fig, "R05_clean_vs_attacked_budget.png")


def r08():
    raw = np.linspace(0, .1, 400)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(raw, np.minimum(raw, .05), color=BLUE, lw=2)
    ax.axhline(.05, color=RED, ls="--", label="hard q_cap")
    ax.set(xlabel="q_raw", ylabel="q_ctrl", title="Runtime hard-cap law")
    ax.legend(); ax.grid(alpha=.3)
    save(fig, "R08_hard_q_cap_audit.png")


def r19():
    fig, ax = plt.subplots(figsize=(11, 4.8)); ax.axis("off")
    steps = [
        ("Frozen sensor\nexternal discrimination", "ST1 PASS", GREEN),
        ("Frozen-W\napplicability", "ST3 FAIL\n0.8667 < 0.90", RED),
        ("Attacked budget", "NOT RUN", GREY),
        ("Controller efficacy", "NOT RUN", GREY),
        ("Cantor specificity", "NOT RUN", GREY),
    ]
    xs = np.linspace(.09, .91, len(steps))
    for i, (label, status, color) in enumerate(steps):
        if i: ax.annotate("", (xs[i]-.075,.52), (xs[i-1]+.075,.52), arrowprops=dict(arrowstyle="->", lw=2))
        ax.add_patch(plt.Circle((xs[i], .52), .07, fc=color, alpha=.85))
        ax.text(xs[i], .76, label, ha="center", va="center", fontsize=10)
        ax.text(xs[i], .52, status, ha="center", va="center", color="white", fontsize=8, weight="bold")
    ax.text(.5, .13, "OVERALL: E_EXTERNAL_SENSOR_TRANSPORT_FAILURE", ha="center", fontsize=13, weight="bold")
    ax.set_title("V3.4.0R confirmatory evidence chain")
    save(fig, "R19_evidence_chain.png")


def main():
    r01(); r02(); r03(); r04(); r05()
    stopped(6, "final_q_rms", "Final q_rms by rho")
    stopped(7, "q_by_attack", "q by attack family / epsilon")
    r08()
    stopped(9, "attack_w_arms", "ATTACK-W: no controller vs linear vs Cantor")
    stopped(10, "attack_v_arms", "ATTACK-V: no controller vs linear vs Cantor")
    stopped(11, "cantor_vs_attack_only", "Cantor 1/3 vs attacked no-controller")
    stopped(12, "cantor_vs_linear", "Cantor 1/3 vs linear")
    stopped(13, "rho_maxt", "rho-family max-T intervals")
    stopped(14, "certificate_validation", "Direct terminal-policy certificate validation")
    stopped(15, "first_failure", "First-failure and censoring diagnostics")
    stopped(16, "reversion", "Non-monotonic reversion analysis")
    stopped(17, "benign_utility", "Benign false refusal / utility")
    stopped(18, "benign_window_shift", "Benign window distribution shift")
    r19()


if __name__ == "__main__":
    main()
