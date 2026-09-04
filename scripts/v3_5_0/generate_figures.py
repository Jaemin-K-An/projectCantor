"""Phase 22: V3.5.0 figures, with post-budget panels explicitly blocked."""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from cantor_guard_v350.one_sided_cantor import epsilon_r, partition  # noqa: E402
from _common import FIGURES, RESULTS, read_json, write_json  # noqa: E402

BLUE, ORANGE, RED, GREY = "#2563eb", "#f59e0b", "#dc2626", "#64748b"


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight"); plt.close(fig)


def blocked(name, title):
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.axis("off")
    ax.text(.5, .62, title, ha="center", va="center", fontsize=15, weight="bold")
    ax.text(.5, .42, "NOT RUN — BUD2_MISMATCH\nq_target=.03 is unattainable under q_cap=.05\nFinal data remained untouched.",
            ha="center", va="center", fontsize=12, color=RED, linespacing=1.5)
    save(fig, name)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    cal = read_json(RESULTS / "tables/risk_window_calibration.json"); W = float(cal["W_R"])
    old = pd.read_csv(ROOT / "results/v3_4_0r/raw/clean_D_sensor_transfer_r.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.hist(old.d_clean, bins=25, color=BLUE, alpha=.75)
    old_W = 2.2805212277347544; ax.axvline(-old_W, color=RED, ls="--"); ax.axvline(old_W, color=RED, ls="--")
    ax.set(title="V3.4.0R external signed sensor distance", xlabel="d (larger = safer)", ylabel="count")
    ax.text(old_W, ax.get_ylim()[1]*.9, "safe-side overflow", ha="left", color=RED)
    save(fig, "F350-01-v340r-safe-side-overflow.png")

    d = np.linspace(-3.5, 4, 500); x = np.maximum(0, -d)
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.plot(d, x, lw=3, color=BLUE); ax.axvline(0, color=GREY, ls="--")
    ax.fill_between(d[d >= 0], 0, x[d >= 0], color=BLUE, alpha=.1)
    ax.set(title="One-sided risk transform", xlabel="signed sensor distance d", ylabel="x=max(0,-d)")
    save(fig, "F350-02-one-sided-transform.png")

    risk = pd.read_csv(RESULTS / "raw/risk_window_calibration.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.hist(risk.x_risk, bins=30, color=BLUE, alpha=.75); ax.axvline(W, color=RED, lw=2, ls="--", label=f"W_R={W:.3f}")
    ax.set(title="Split-conformal risk-window calibration", xlabel="x", ylabel="count"); ax.legend()
    save(fig, "F350-03-conformal-risk-window.png")

    leaves, guards = partition(1/3, 3); fig, ax = plt.subplots(figsize=(9, 2.8))
    for leaf in leaves: ax.axvspan(leaf.lo, leaf.hi, color=BLUE, alpha=.75); ax.text((leaf.lo+leaf.hi)/2, .5, str(leaf.index), ha="center", color="white", weight="bold")
    for guard in guards: ax.axvspan(guard.lo, guard.hi, color=ORANGE, alpha=.65)
    ax.set(xlim=(0,1), ylim=(0,1), yticks=[], xlabel="r_R=x/W_R", title="Depth-3 middle-third Cantor risk policy")
    save(fig, "F350-04-depth3-risk-cantor.png")

    rho = np.linspace(.2, .49, 500); eps = np.array([epsilon_r(r, W) for r in rho])
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.plot(rho, eps, color=BLUE, lw=3); ax.axvline(1/3, color=RED, ls="--", label="unique maximum rho=1/3")
    ax.set(title="One-sided certificate ε_R(ρ)", xlabel="rho", ylabel="W_R rho²(1−2rho)"); ax.legend()
    save(fig, "F350-05-certificate-rho.png")

    budget = read_json(RESULTS / "tables/budget_calibration.json")
    names = list(budget["per_arm"]); maxima = [budget["per_arm"][x]["maximum_attainable_q_rms"] for x in names]
    fig, ax = plt.subplots(figsize=(9, 4.8)); ax.bar(names, maxima, color=ORANGE); ax.axhline(.03, color=RED, ls="--", lw=2, label="frozen target .03")
    ax.set(title="Attacked-state budget feasibility (hard stop)", ylabel="maximum attainable q RMS", ylim=(0,.033)); ax.tick_params(axis="x", rotation=35); ax.legend()
    save(fig, "F350-06-equal-budget-impossible.png")

    blocked("F350-07-attack-v-no-controller-vs-cantor.png", "ATTACK-V: attack-only vs Cantor")
    blocked("F350-08-attack-w-no-controller-vs-cantor.png", "ATTACK-W: attack-only vs Cantor")
    blocked("F350-09-cantor-vs-linear.png", "Cantor vs linear")
    blocked("F350-10-rho-family-curves.png", "Rho-family curves")
    blocked("F350-11-simultaneous-cis.png", "Simultaneous confidence intervals")

    cert = read_json(RESULTS / "tables/certificate_validation.json")
    labels = list(cert["per_rho"]); violations = [cert["per_rho"][x]["violations"] for x in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.bar(labels, violations, color=BLUE); ax.set(title="Certificate implementation validation", xlabel="rho", ylabel="direct leaf-switch violations", ylim=(0,1))
    save(fig, "F350-12-certificate-validation.png")
    blocked("F350-13-benign-intervention-frequency.png", "Benign intervention frequency")

    fig, ax = plt.subplots(figsize=(11, 3.2)); ax.axis("off")
    labels = ["P0 residual", "frozen sensor d", "x=max(0,-d)", "Cantor cell", "v_safe action", "generation"]
    xs = np.linspace(.08,.92,len(labels))
    for i,(xv,label) in enumerate(zip(xs,labels)):
        ax.text(xv,.55,label,ha="center",va="center",bbox=dict(boxstyle="round,pad=.5",fc="#eff6ff",ec=BLUE),fontsize=10)
        if i < len(labels)-1: ax.annotate("",xy=(xs[i+1]-.07,.55),xytext=(xv+.07,.55),arrowprops=dict(arrowstyle="->",color=GREY,lw=1.5))
    ax.text(.5,.16,"Execution halted before final generation: frozen q budget infeasible",ha="center",color=RED,weight="bold")
    save(fig, "F350-14-full-pipeline.png")
    captions = {f"F350-{i:02d}": ("post-budget endpoint not run" if i in {7,8,9,10,11,13} else "generated from V3.5.0 preregistered or calibration evidence") for i in range(1,15)}
    write_json(FIGURES / "CAPTIONS.json", captions)
    print("generated 14 figures")


if __name__ == "__main__": main()
