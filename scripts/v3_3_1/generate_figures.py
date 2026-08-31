"""V3.3.1 figures."""
import sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cantor_guard_v331.guard_geometry import (
    guard_width, retention, hausdorff_dim, alpha_field, alpha_sensitivity,
    rho_star, bottleneck, new_guard_measure, new_coverage_argmax, RHO_CANTOR)

FIG = pathlib.Path("figures/v3_3_1"); FIG.mkdir(parents=True, exist_ok=True)
TAB = pathlib.Path("results/v3_3_1/tables")
plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
CAN = "#d93025"
r = np.linspace(0.005, 0.495, 2000)


def mark(ax, y=None):
    ax.axvline(1/3, color=CAN, ls="--", lw=1.1)
    ax.annotate("Cantor 1/3", (1/3, ax.get_ylim()[1]), color=CAN, fontsize=7.5,
                ha="center", va="top", xytext=(0, -4), textcoords="offset points")


# F1/F2/F3 -- geometry, crossing, bottleneck
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].bar([0, 1, 2], [1/3, 1/3, 1/3], color=["#1a73e8", "#34a853", "#1a73e8"])
ax[0].set_xticks([0, 1, 2]); ax[0].set_xticklabels(["child\nrho", "GUARD\ng", "child\nrho"])
ax[0].set_ylabel("width"); ax[0].set_title("F1 — Cantor: 1 : 1 : 1", fontsize=9)
ax[1].plot(r, r, label=r"refinement $\rho$", color="#1a73e8")
ax[1].plot(r, 1 - 2 * r, label=r"guard $g=1-2\rho$", color="#34a853")
ax[1].plot(1/3, 1/3, "o", color=CAN, ms=8)
ax[1].set_xlabel(r"$\rho$"); ax[1].legend(fontsize=8)
ax[1].set_title(r"F2 — they cross at $\rho=1/3$", fontsize=9); mark(ax[1])
for k, c in zip((0.5, 1.0, 2.0), ("#9aa0a6", "#1a73e8", "#5f6368")):
    ax[2].plot(r, [bottleneck(x, k) for x in r], color=c, label=f"$\\kappa$={k}")
    ax[2].plot(rho_star(k), 1/(2+k), "o", color=c, ms=5)
ax[2].set_xlabel(r"$\rho$"); ax[2].set_ylabel(r"$B_\kappa=\min(\rho,g/\kappa)$")
ax[2].legend(fontsize=8); ax[2].set_title("F3 — unique maximum", fontsize=9); mark(ax[2])
fig.tight_layout(); fig.savefig(FIG / "F1_F2_F3_geometry.png"); plt.close(fig)

# F4 -- rho*(kappa)
fig, ax = plt.subplots(figsize=(5.6, 3.6))
kk = np.linspace(0.2, 3.0, 400)
ax.plot(kk, 1 / (2 + kk), color="#1a73e8")
ax.plot(1.0, 1/3, "o", color=CAN, ms=9)
ax.annotate(r"$\kappa=1 \Rightarrow \rho^*=1/3$", (1.0, 1/3),
            xytext=(14, 14), textcoords="offset points", color=CAN, fontsize=8.5)
ax.set_xlabel(r"guard weight $\kappa$"); ax.set_ylabel(r"$\rho^*(\kappa)=1/(2+\kappa)$")
ax.set_title("F4 — Cantor is the balanced case, not a universal optimum", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "F4_rho_star_kappa.png"); plt.close(fig)

# F5-F8 -- the four objectives
fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
for a_, f, t in zip(ax, (retention, hausdorff_dim, alpha_field, alpha_sensitivity),
                    (r"retention $2\rho$", r"$d_H$", r"$A_f=1/(2\rho)$",
                     r"$A_s=1/(2\rho^2)$")):
    a_.plot(r, [f(x) for x in r], color="#1a73e8")
    a_.plot(1/3, f(1/3), "o", color=CAN, ms=7)
    a_.axvspan(1/3, 0.5, color="#d93025", alpha=.07)
    a_.set_xlabel(r"$\rho$"); a_.set_title(t, fontsize=9)
ax[0].set_ylabel("value")
fig.suptitle("F5–F8 — all four improve with rho; the shaded region is INFEASIBLE "
             "under g>=rho, so the optimum sits at the boundary", fontsize=9.5)
fig.tight_layout(); fig.savefig(FIG / "F5_F8_objectives.png"); plt.close(fig)

# F9 -- new coverage, the honest counterexample
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for n_, c in zip((1, 2, 3, 5, 10), plt.cm.viridis(np.linspace(0, .85, 5))):
    ax.plot(r, [new_guard_measure(x, n_) for x in r], color=c, label=f"n={n_}")
    ax.plot(new_coverage_argmax(n_), new_guard_measure(new_coverage_argmax(n_), n_),
            "o", color=c, ms=5)
ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$F_n(\rho)=(1-2\rho)(2\rho)^n$")
ax.legend(fontsize=7.5)
ax.set_title(r"F9 — new-coverage argmax $=n/(2(n+1))$, equals 1/3 ONLY at n=2",
             fontsize=9); mark(ax)
fig.tight_layout(); fig.savefig(FIG / "F9_new_coverage.png"); plt.close(fig)

# F12 -- synthetic: empirical boundary vs 1/(2+beta)
sb = pd.read_csv(TAB / "synthetic_beta_prediction.csv")
sb = sb[sb.perturbation == "worst_case"].groupby("beta").agg(
    emp=("rho_emp_opt", "median"), p1=("rho_star_pred", "first"),
    p2=("rho_star_pred_two_sided", "first")).reset_index()
fig, ax = plt.subplots(figsize=(6.2, 4.0))
ax.plot(sb.beta, sb.p1, "-", color="#1a73e8", label=r"theory $1/(2+\beta)$")
ax.plot(sb.beta, sb.p2, "--", color="#9aa0a6", label=r"rejected $1/(2+2\beta)$")
ax.plot(sb.beta, sb.emp, "o", color=CAN, ms=7, label="measured boundary")
ax.axhline(1/3, color="k", ls=":", lw=.8)
ax.set_xlabel(r"$\beta$ = uncertainty / leaf width"); ax.set_ylabel(r"$\rho$")
ax.legend(fontsize=8)
ax.set_title(r"F12 — synthetic boundary tracks $1/(2+\beta)$ (corr 0.999);"
             "\n" r"Cantor corresponds to $\beta=1$", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "F12_synthetic_beta.png"); plt.close(fig)

# F14 -- LLM: rho sweep with the noise floor
llm = pd.read_csv("results/v3_3_1/raw/v331_llm_rho_qwen2.5-0.5b-instruct.csv")
cal = json.loads((TAB / "phase_calibration_qwen2.5-0.5b-instruct.json").read_text())
nmax = cal["max_useful_depth"]["q50"]["n_max_over_all_rho"]
fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.9))
for d, c in zip(sorted(llm.depth.unique()), ("#1a73e8", "#9aa0a6")):
    s = llm[llm.depth == d].groupby("rho").safe.agg(["mean", "sem"])
    lab = f"n={d}" + (" (feasible)" if d <= nmax else " (below noise floor)")
    ax[0].errorbar(s.index, s["mean"], yerr=s["sem"], fmt="o-", ms=5,
                   color=c, capsize=3, label=lab)
ax[0].axvline(1/3, color=CAN, ls="--", lw=1.1)
ax[0].set_xlabel(r"$\rho$"); ax[0].set_ylabel("safety (goal mean)")
ax[0].legend(fontsize=8)
ax[0].set_title("F14 — no rho effect at either depth\n(spread 0.006 vs SESOI 0.03)",
                fontsize=9)
gw = [x ** (nn - 1) * (1 - 2 * x) for nn in range(1, 8) for x in [1/3]]
ax[1].semilogy(range(1, 8), gw, "o-", color="#1a73e8", label=r"level-$n$ guard, $\rho$=1/3")
for q, c in zip(("q50", "q95"), ("#34a853", "#d93025")):
    ax[1].axhline(cal["eps_cal_quantiles"][q], color=c, ls="--", lw=1,
                  label=f"calibration uncertainty {q}")
ax[1].axvline(5, color="k", ls=":", lw=.9)
ax[1].annotate("V3.1/V3.2 ran n=5", (5, 1e-3), fontsize=7.5, rotation=90,
               va="bottom", ha="right")
ax[1].set_xlabel("depth n"); ax[1].set_ylabel("width (threat coordinate)")
ax[1].legend(fontsize=7.5)
ax[1].set_title("F14b — the guard falls below the noise floor after n≈2", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "F14_llm_rho_sweep.png"); plt.close(fig)
print("figures ->", FIG)
for p in sorted(FIG.glob("*.png")): print("  ", p.name)
