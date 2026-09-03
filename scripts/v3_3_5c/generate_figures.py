"""Generate the sixteen preregistered V3.3.5c figures from available evidence."""
from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "llm/src")
from cantor_guard_v335c.cantor_geometry import epsilon_z, partition  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/v3_3_5c"
OUT = ROOT / "figures/v3_3_5c"
OUT.mkdir(parents=True, exist_ok=True)


def placeholder(ax, title: str, reason: str = "Not run: upstream gate or data unavailable"):
    ax.set_title(title)
    ax.text(0.5, 0.5, reason, ha="center", va="center", wrap=True, transform=ax.transAxes)
    ax.set_axis_off()


def save(fig, number: int, slug: str):
    fig.tight_layout()
    fig.savefig(OUT / f"F335c-{number:02d}_{slug}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def figure_01():
    data = pd.read_csv(ROOT / "results/v3_3_5b/raw/temporal_D_temporal_confirm.csv")
    base = data[data.B2_target == 0].refusal.mean()
    table = data[data.B2_target > 0].groupby(["schedule", "B2_target"]).refusal.mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 4))
    for schedule in ("S1_P0_ONLY", "S2_G1_ONLY", "S4_EARLY_4", "S5_EARLY_8"):
        ax.plot(table.columns, base - table.loc[schedule], marker="o", label=schedule)
    ax.set(title="Corrected V3.3.5b matched-B2 comparison", xlabel="Matched trajectory B2", ylabel="Refusal reduction")
    ax.legend(fontsize=8)
    save(fig, 1, "corrected_temporal")


def dose_figures():
    path = RESULTS / "raw/symmetric_D_beh_P0_dev_335c.csv"
    for number, slug, title in ((2, "symmetric_dose", "P0 symmetric normalized dose-response"), (3, "outcome_vs_z", "Refusal/compliance vs realised z"), (4, "boundary_fit", "Logistic and isotonic P0 boundary")):
        fig, ax = plt.subplots(figsize=(6, 4))
        if not path.exists():
            placeholder(ax, title)
        else:
            data = pd.read_csv(path)
            if number == 2:
                summary = data.groupby("u").agg(refusal=("refusal_proxy", "mean"), coherence=("coherence", "mean"))
                ax.plot(summary.index, summary.refusal, marker="o", label="refusal")
                ax.plot(summary.index, summary.coherence, marker="s", label="coherence")
                ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
                ax.set(xlabel="signed relative dose u", ylabel="rate", title=title); ax.legend()
            elif number == 3:
                ax.scatter(data.z_after, data.refusal_proxy, s=8, alpha=0.25)
                bins = pd.qcut(data.z_after, min(20, data.z_after.nunique()), duplicates="drop")
                summary = data.groupby(bins, observed=True).agg(z=("z_after", "mean"), y=("refusal_proxy", "mean"))
                ax.plot(summary.z, summary.y, color="black", marker="o")
                ax.set(xlabel="realised z_after", ylabel="refusal probability", title=title)
            else:
                boundary_path = RESULTS / "tables/p0_behavioral_boundary.json"
                if not boundary_path.exists():
                    placeholder(ax, title)
                else:
                    boundary = json.loads(boundary_path.read_text())
                    fit = boundary["confirm"]
                    z = np.linspace(data.z_after.min(), data.z_after.max(), 400)
                    p = 1 / (1 + np.exp(-(fit["intercept"] + fit["slope"] * z)))
                    ax.plot(z, p, label="logistic")
                    if fit["tau_isotonic"] is not None:
                        ax.axvline(fit["tau_isotonic"], label="isotonic crossing", linestyle=":")
                    if fit["tau_logistic"] is not None:
                        ax.axvline(fit["tau_logistic"], label="logistic crossing", linestyle="--")
                    ax.set(xlabel="realised z_after", ylabel="P(refusal)", title=title); ax.legend()
        save(fig, number, slug)


def figure_05():
    fig, ax = plt.subplots(figsize=(6, 4))
    path = RESULTS / "raw/window_calibration.csv"
    config_path = ROOT / "configs/v3_3_5c/controller.json"
    config = json.loads(config_path.read_text())
    if not path.exists() or config["tau"] is None:
        placeholder(ax, "Clean z distribution, tau, and affine W")
    else:
        data = pd.read_csv(path); tau, W = config["tau"], config["W"]
        ax.hist(data.z_clean, bins=18, alpha=0.7)
        ax.axvline(tau, color="black", label="tau")
        ax.axvspan(tau - W, tau + W, alpha=0.15, label="window")
        ax.set(title="Clean z distribution, tau, and affine W", xlabel="z_clean", ylabel="count"); ax.legend()
    save(fig, 5, "window")


def figure_06():
    fig, ax = plt.subplots(figsize=(8, 2.2))
    leaves, guards = partition(1 / 3, 3)
    for leaf in leaves:
        ax.axvspan(leaf.lo, leaf.hi, color="#4c78a8", alpha=0.8)
    for guard in guards:
        ax.axvspan(guard.lo, guard.hi, color="#e45756", alpha=0.25 + 0.12 * (4 - guard.level))
    ax.axvline(0.5, color="black", linestyle="--", label="behavioral boundary")
    ax.set(xlim=(0, 1), yticks=[], xlabel="affine risk coordinate r", title="Actual depth-3 middle-third policy partition")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, 6, "cantor_partition")


def figure_07():
    fig, ax = plt.subplots(figsize=(6, 4))
    rhos = np.linspace(0.001, 0.499, 1000)
    config = json.loads((ROOT / "configs/v3_3_5c/controller.json").read_text())
    W = config["W"] or 1.0
    ax.plot(rhos, epsilon_z(rhos, W))
    ax.axvline(1 / 3, color="black", linestyle="--", label="rho=1/3")
    ax.set(title="Exact P0 residual certificate", xlabel="rho", ylabel="epsilon_z(rho)"); ax.legend()
    save(fig, 7, "certificate_curve")


def downstream_fig(number: int, slug: str, title: str, path: str, plotter):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    target = RESULTS / path
    if not target.exists():
        placeholder(ax, title)
    else:
        plotter(ax, pd.read_csv(target))
        ax.set_title(title)
    save(fig, number, slug)


def main() -> None:
    figure_01(); dose_figures(); figure_05(); figure_06(); figure_07()
    downstream_fig(8, "occupancy", "Guard/leaf/outside occupancy", "tables/final_budget_audit.csv",
                    lambda ax, d: d.set_index("rho")[["guard_frequency", "leaf_frequency", "outside_window_frequency"]].plot.bar(ax=ax))
    downstream_fig(9, "certificate_states", "Leaf/guard/direct crossing under attack", "raw/certificate_validation.csv",
                    lambda ax, d: d.groupby(["lambda", "attacked_cell_kind"]).size().unstack(fill_value=0).plot(ax=ax))
    downstream_fig(10, "first_token", "First-token divergence and restoration", "tables/first_token_analysis.csv",
                    lambda ax, d: d.groupby("epsilon")[["kl_clean_attacked", "kl_clean_corrected"]].mean().plot(ax=ax))
    downstream_fig(11, "semantic_curves", "Harmful-compliance attack curves", "tables/final_generation_scored.csv",
                    lambda ax, d: d.assign(endpoint=d.get("semantic_harmful_compliance", d.get("harmful_compliance_proxy"))).groupby(["epsilon", "rho"]).endpoint.mean().unstack().plot(ax=ax))
    downstream_fig(12, "auc", "Robustness AUC simultaneous intervals", "tables/generation_maxT_contrasts.csv",
                    lambda ax, d: ax.errorbar(np.arange(len(d)), d.mean_diff, yerr=[d.mean_diff-d.simult_lo, d.simult_hi-d.mean_diff], fmt="o"))
    downstream_fig(13, "failure_threshold", "Empirical failure vs analytic certificate", "tables/failure_threshold_summary.csv",
                    lambda ax, d: ax.scatter(d.epsilon_cert, d.median_epsilon_fail, c=d.rho))
    downstream_fig(14, "final_budget", "D_final controller budget by rho", "tables/final_budget_audit.csv",
                    lambda ax, d: ax.plot(d.rho, d.q_rms, marker="o"))
    downstream_fig(15, "benign_utility", "Independent benign utility", "tables/utility_summary.csv",
                    lambda ax, d: d.set_index("condition")[["false_refusal", "coherence", "degeneration_rate"]].plot.bar(ax=ax))
    fig, ax = plt.subplots(figsize=(11, 2.8))
    stages = ["V1", "V2–V3.2", "V3.3–.2", "V3.3.3", "V3.3.4", "V3.3.5", "V3.3.5a", "V3.3.5b", "V3.3.5c"]
    notes = ["confounds", "matched null", "1/3 theorem", "tau differs", "warp shifts z optimum", "affine; G1 fails", "P0; absolute dose", "P0 concentration", "behavioral closure"]
    ax.plot(range(len(stages)), np.zeros(len(stages)), marker="o")
    for i, note in enumerate(notes): ax.text(i, 0.04 if i % 2 == 0 else -0.04, note, rotation=35, ha="left", va="bottom" if i % 2 == 0 else "top", fontsize=8)
    ax.set_xticks(range(len(stages)), stages); ax.set_yticks([]); ax.set_ylim(-0.22, 0.22); ax.set_title("V1 → V3.3.5c evidence chain")
    save(fig, 16, "evidence_chain")
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
