"""V3.3.2 figures."""
import sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cantor_guard_v332.absolute_guard import (G_n, G_n_max, rho_guard_max,
                                              rho_abs_star, RHO_CANTOR)

FIG = pathlib.Path("figures/v3_3_2"); FIG.mkdir(parents=True, exist_ok=True)
TAB = pathlib.Path("results/v3_3_2/tables")
CAL = json.loads((TAB / "phase_calibration_qwen2.5-0.5b-instruct.json").read_text())
PAR = json.loads((TAB / "systemB_pareto.json").read_text())
sb = pd.read_csv("results/v3_3_2/raw/systemB_real_coordinate.csv")
plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
CAN = "#d93025"
DQ = CAL["U_EST"]["delta_abs_quantiles"]
r = np.linspace(0.005, 0.495, 2000)

# F332-01/02/03 -- phase projections and calibrations
fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.6))
pj = CAL["projections"]
for i, ph in enumerate(("P", "G1")):
    ax[i].hist(pj[ph]["harmful"], bins=8, alpha=.6, color="#d93025", label="harmful")
    ax[i].hist(pj[ph]["harmless"], bins=8, alpha=.6, color="#1a73e8", label="harmless")
    c = CAL["calibrations"][ph]
    ax[i].axvline(c["tau"], color="k", ls="--", lw=1.2)
    ax[i].set_title(f"F332-0{i+1} — phase {ph}: tau={c['tau']:+.3f} "
                    f"sigma={c['sigma']:.3f}", fontsize=8.5)
    ax[i].set_xlabel("projection z"); ax[i].legend(fontsize=7)
cals = CAL["calibrations"]
names = list(cals)
ax[2].bar(range(len(names)), [cals[k]["sigma"] for k in names], color="#34a853")
ax[2].set_xticks(range(len(names))); ax[2].set_xticklabels(names, fontsize=8)
ax[2].set_ylabel(r"$\sigma$"); ax[2].set_title("F332-03 — phase scales", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "F332-01_03_phases.png"); plt.close(fig)

# F332-06/07 -- G_n with the measured uncertainty bands
fig, ax = plt.subplots(figsize=(7.4, 4.4))
for n, c in zip((2, 3, 5), ("#1a73e8", "#34a853", "#9aa0a6")):
    ax.plot(r, G_n(r, n), color=c, label=f"$G_{{{n}}}(\\rho)$")
    ax.plot(rho_guard_max(n), G_n_max(n), "o", color=c, ms=5)
for k, c, ls in (("q50", "#d93025", "--"), ("q95", "#f9ab00", ":")):
    ax.axhline(DQ[k], color=c, ls=ls, lw=1.2,
               label=f"measured $\\delta_{{abs}}$ {k} = {DQ[k]:.4f}")
ax.axvline(RHO_CANTOR, color=CAN, ls="--", lw=1)
ax.annotate("Cantor 1/3", (RHO_CANTOR, 0.13), color=CAN, fontsize=7.5, ha="center")
ax.set_yscale("log"); ax.set_ylim(1e-4, 0.2)
ax.set_xlabel(r"$\rho$"); ax.set_ylabel("finest guard width $G_n$")
ax.legend(fontsize=7.5)
ax.set_title("F332-06/07 — measured uncertainty against the achievable guard\n"
             r"($\rho^*$ is where a curve crosses the band, on the RIGHT branch)",
             fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "F332-06_07_guard_vs_delta.png"); plt.close(fig)

# F332-08 -- predicted rho bootstrap vs the Cantor band
fig, ax = plt.subplots(figsize=(6.6, 3.8))
rb = CAL["rho_pred_bootstrap"]
ys, labs = [], []
for i, (n, v) in enumerate(rb.items()):
    if v["rho_pred_ci95"] is None:
        continue
    lo, hi = v["rho_pred_ci95"]
    ax.plot([lo, hi], [i, i], lw=3, color="#1a73e8")
    ax.plot(v["rho_pred_median"], i, "o", color="#1a73e8", ms=7)
    ys.append(i); labs.append(f"n={n}  ({v['feasible_fraction']:.0%} feasible)")
ax.axvspan(RHO_CANTOR - .03, RHO_CANTOR + .03, color=CAN, alpha=.18)
ax.axvline(RHO_CANTOR, color=CAN, ls="--", lw=1.2)
ax.annotate("pre-registered\nCantor band", (RHO_CANTOR, max(ys) + .35),
            color=CAN, fontsize=7.5, ha="center")
ax.set_yticks(ys); ax.set_yticklabels(labs, fontsize=8)
ax.set_xlabel(r"predicted $\rho^*_{abs}$"); ax.set_xlim(0.25, 0.52)
ax.set_title("F332-08 — the measured uncertainty predicts $\\rho\\approx0.46$,\n"
             "outside the Cantor band at every depth", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "F332-08_rho_pred.png"); plt.close(fig)

# F332-11/12/13 -- real-coordinate System B
s = sb.groupby(["depth", "rho"]).agg(regression=("regression", "mean"),
                                     abstention=("abstention", "mean")).reset_index()
fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
for n, c in zip((2, 3, 5), ("#1a73e8", "#34a853", "#9aa0a6")):
    g = s[s.depth == n]
    ax[0].plot(g.rho, g.regression, "o-", ms=4, color=c, label=f"n={n}")
    ax[1].plot(g.rho, g.abstention, "o-", ms=4, color=c, label=f"n={n}")
    ax[2].plot(g.regression, g.abstention, "o-", ms=4, color=c, label=f"n={n}")
    cc = g[np.isclose(g.rho, RHO_CANTOR)]
    ax[2].plot(cc.regression, cc.abstention, "*", color=CAN, ms=14)
for a_ in ax[:2]:
    a_.axvline(RHO_CANTOR, color=CAN, ls="--", lw=1); a_.set_xlabel(r"$\rho$")
    a_.legend(fontsize=7.5)
ax[0].set_ylabel("policy-switch regression"); ax[0].set_title("F332-11", fontsize=9)
ax[1].set_ylabel("abstention"); ax[1].set_title("F332-12", fontsize=9)
ax[2].set_xlabel("regression"); ax[2].set_ylabel("abstention")
ax[2].set_title("F332-13 — Pareto (star = Cantor);\nlower-left is better", fontsize=9)
ax[2].legend(fontsize=7.5)
fig.suptitle("System B on REAL D_final residual coordinates "
             f"($\\delta_{{abs}}$ = {DQ['q50']:.4f}, measured independently of "
             r"$\rho$)", fontsize=9.5)
fig.tight_layout(); fig.savefig(FIG / "F332-11_13_systemB.png"); plt.close(fig)

# F332-18 -- evidence matrix
fig, ax = plt.subplots(figsize=(8.6, 2.9))
claims = ["balanced optimum\n(kappa=1)", "n=3 guard max\n= 1/3",
          "n=2, delta=1/9\n-> 1/3", "measured delta\nselects 1/3",
          "Cantor on real\nPareto front"]
vals = [1, 1, 1, 0, 0]
ax.imshow(np.array(vals).reshape(1, -1), cmap="RdYlGn", vmin=-0.4, vmax=1.4,
          aspect="auto")
ax.set_xticks(range(len(claims))); ax.set_xticklabels(claims, fontsize=8)
ax.set_yticks([])
for i, v in enumerate(vals):
    ax.text(i, 0, "PROVED" if v else "NO", ha="center", va="center",
            fontsize=9, weight="bold")
ax.set_title("F332-18 — mathematics holds; the measured LLM uncertainty does not "
             "select 1/3", fontsize=9.5)
fig.tight_layout(); fig.savefig(FIG / "F332-18_evidence_matrix.png"); plt.close(fig)
print("figures ->", FIG)
for p in sorted(FIG.glob("*.png")): print("  ", p.name)
