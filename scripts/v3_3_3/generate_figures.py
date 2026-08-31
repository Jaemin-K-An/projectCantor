"""V3.3.3 figures."""
import sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cantor_guard_v332.absolute_guard import G_n, G_n_max, rho_guard_max
FIG = pathlib.Path("figures/v3_3_3"); FIG.mkdir(parents=True, exist_ok=True)
TAB = pathlib.Path("results/v3_3_3/tables")
plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
CAN = "#d93025"; RC = 1/3
r = np.linspace(0.005, 0.495, 2000)
BEH = json.loads((TAB/"behavioral_boundary.json").read_text())
QS = pd.read_csv(TAB/"quantile_sensitivity.csv")

# F333-01: G_n and argmax
fig, ax = plt.subplots(figsize=(6.6, 4.0))
for n, c in zip((2, 3, 5), ("#9aa0a6", "#1a73e8", "#34a853")):
    ax.plot(r, G_n(r, n), color=c, label=f"$G_{{{n}}}$")
    ax.plot(rho_guard_max(n), G_n_max(n), "o", color=c, ms=6)
    ax.annotate(f"{rho_guard_max(n):.3f}", (rho_guard_max(n), G_n_max(n)),
                xytext=(4, 5), textcoords="offset points", fontsize=7, color=c)
ax.axvline(RC, color=CAN, ls="--", lw=1.2)
ax.annotate("1/3", (RC, 0.115), color=CAN, fontsize=8, ha="center")
ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$G_n(\rho)=\rho^{n-1}(1-2\rho)$")
ax.legend(fontsize=8)
ax.set_title(r"F333-01 — $\rho_{max}(n)=(n-1)/(2n)$; equals 1/3 ONLY at $n=3$",
             fontsize=9)
fig.tight_layout(); fig.savefig(FIG/"F333-01_Gn.png"); plt.close(fig)

# F333-02: n=3 with all four quantiles, both uncertainty definitions
fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.0))
for k, (name, sub) in enumerate(QS[QS.n == 3].groupby("uncertainty")):
    ax[k].plot(r, G_n(r, 3), color="#1a73e8", lw=1.6)
    ax[k].axvline(RC, color=CAN, ls="--", lw=1.2)
    for _, row in sub.iterrows():
        c = {"q50": "#34a853", "q75": "#f9ab00", "q90": "#e8710a",
             "q95": "#d93025"}[row["quantile"]]
        ax[k].axhline(row.delta, color=c, ls=":", lw=1.2,
                      label=f"{row['quantile']}={row.delta:.4f}"
                            + ("" if row.feasible else "  INFEASIBLE"))
        if row.feasible:
            ax[k].plot(row.rho_right, row.delta, "v", color=c, ms=7)
    ax[k].set_xlabel(r"$\rho$"); ax[k].set_ylabel(r"$G_3(\rho)$")
    ax[k].set_title(f"F333-02 — n=3, {name}", fontsize=9)
    ax[k].legend(fontsize=7)
fig.suptitle("the V3.3.2 point estimate depends on the quantile: with U_EST_mid "
             "only q50 is feasible; with U_EST_beh none is", fontsize=9.5)
fig.tight_layout(); fig.savefig(FIG/"F333-02_quantiles.png"); plt.close(fig)

# F333-03/04: dose-response and tau_mid vs tau_beh
dr = pd.read_csv("results/v3_3_3/raw/behavioral_dose_response.csv")
fig, ax = plt.subplots(1, 2, figsize=(11.4, 3.9))
g = dr.groupby("dose_sigma").agg(refusal=("refusal", "mean"), z=("z", "mean"))
ax[0].plot(g.z, g.refusal, "o-", color="#1a73e8", ms=5)
ax[0].axhline(0.5, color="k", ls=":", lw=.9)
tb = BEH["tau_beh"]; ci = BEH["bootstrap"]["tau_ci95"]
ax[0].axvspan(ci[0], ci[1], color=CAN, alpha=.15)
ax[0].axvline(tb, color=CAN, ls="--", lw=1.4)
ax[0].annotate(r"$\tau_{beh}$", (tb, .55), color=CAN, fontsize=9, ha="right")
ax[0].set_xlabel("realised projection z"); ax[0].set_ylabel("refusal rate")
ax[0].set_title(r"F333-03 — behavioural dose-response, $\tau_{beh}$ CI",
                fontsize=9)
tm = BEH["tau_mid_G1"]; sg = BEH["sigma_G1"]
ax[1].barh([0, 1], [tm, tb], color=["#9aa0a6", CAN])
ax[1].set_yticks([0, 1]); ax[1].set_yticklabels([r"$\tau_{mid}$", r"$\tau_{beh}$"])
ax[1].axvline(0, color="k", lw=.8)
ax[1].set_xlabel("projection")
ax[1].set_title(f"F333-04 — the midpoint is biased by "
                f"{abs(BEH['gap_tau_mid_minus_tau_beh_sigma']):.2f}$\\sigma$",
                fontsize=9)
fig.tight_layout(); fig.savefig(FIG/"F333-03_04_behavior.png"); plt.close(fig)

# F333-05: adversarial crossing vs analytic guard
sb = pd.read_csv("results/v3_3_3/raw/systemB_adversarial.csv").dropna(subset=["min_d_cross"])
fig, ax = plt.subplots(figsize=(6.2, 4.0))
for n, c in zip((2, 3, 5), ("#9aa0a6", "#1a73e8", "#34a853")):
    s = sb[(sb.n == n) & (sb.coord == "tau_mid")]
    ax.scatter(s.G_n, s.min_d_cross, color=c, label=f"n={n}", s=28)
lim = [sb.G_n.min()*.8, sb.min_d_cross.max()*1.1]
ax.plot(lim, lim, "k--", lw=1, label=r"$d_{cross}=G_n$ (bound)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"analytic guard $G_n(\rho)$")
ax.set_ylabel(r"measured minimum crossing distance")
ax.legend(fontsize=8)
ax.set_title("F333-05 — adversarial crossing never violates the bound\n"
             "(0/39 violations; tight to 0.95% under dense coverage)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG/"F333-05_adversarial.png"); plt.close(fig)

# F333-06/07/08: System A
auc = pd.read_csv(TAB/"systemA_auc_per_prompt.csv")
cmp_ = pd.read_csv(TAB/"systemA_comparisons.csv")
ut = pd.read_csv("results/v3_3_3/raw/systemA_utility_qwen2.5-0.5b-instruct.csv")
meta = json.loads((TAB/"systemA_meta.json").read_text())
P = json.loads(pathlib.Path("configs/v3_3_3/protocol.json").read_text())
fig, ax = plt.subplots(1, 3, figsize=(14.2, 3.9))
s = auc.groupby("rho").auc.agg(["mean", "sem"])
bad = [float(k) for k, v in meta["gains"].items() if not v["matched"]]
cols = [CAN if abs(x-RC) < 1e-9 else ("#9aa0a6" if x in bad else "#1a73e8")
        for x in s.index]
ax[0].errorbar(s.index, s["mean"], yerr=s["sem"], fmt="o", ms=6, capsize=3,
               ecolor="#666", ls="none")
ax[0].scatter(s.index, s["mean"], c=cols, s=60, zorder=3)
ax[0].axvline(RC, color=CAN, ls="--", lw=1)
ax[0].set_xlabel(r"$\rho$"); ax[0].set_ylabel("robust safety AUC")
ax[0].set_title("F333-06 — AUC vs rho (grey = budget-excluded)", fontsize=9)
y = np.arange(len(cmp_))
ax[1].errorbar(cmp_.mean_diff, y,
               xerr=[cmp_.mean_diff-cmp_.simult_lo, cmp_.simult_hi-cmp_.mean_diff],
               fmt="o", ms=5, color="#1a73e8", capsize=3)
ax[1].axvline(0, color="k", lw=.8)
ax[1].axvspan(-P["sesoi_auc"], P["sesoi_auc"], color="#34a853", alpha=.12)
ax[1].set_yticks(y); ax[1].set_yticklabels([f"1/3 vs {x:.4f}" for x in cmp_.rho_other],
                                           fontsize=8)
ax[1].set_xlabel("AUC difference"); 
ax[1].set_title("F333-07 — max-T simultaneous bands\n(green = SESOI ±0.02)",
                fontsize=9)
ax[2].scatter(s["mean"], ut.set_index("rho").loc[s.index].false_refusal,
              c=cols, s=60)
ax[2].set_xlabel("robust safety AUC"); ax[2].set_ylabel("benign false refusal")
ax[2].set_title("F333-08 — safety/utility Pareto\n(utility identical at every rho)",
                fontsize=9)
fig.tight_layout(); fig.savefig(FIG/"F333-06_08_systemA.png"); plt.close(fig)

# F333-10: evidence matrix
V = json.loads((TAB/"verdict_v333.json").read_text())
fig, ax = plt.subplots(figsize=(9.2, 2.6))
labs = ["M1 math\n(n=3 -> 1/3)", "M2 behavioural\nboundary",
        "M3 uncertainty\nbridge", "M4 adversarial\ncrossing",
        "M5 generation\nsafety"]
vals = [1, 1, 0, 1, 0]
txt = ["PROVED", "IDENTIFIED", "FRAGILE", "VALIDATED", "INCONCLUSIVE"]
ax.imshow(np.array(vals).reshape(1, -1), cmap="RdYlGn", vmin=-.4, vmax=1.4,
          aspect="auto")
ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=8); ax.set_yticks([])
for i, t in enumerate(txt):
    ax.text(i, 0, t, ha="center", va="center", fontsize=9, weight="bold")
ax.set_title("F333-10 — the mathematics and the mechanism hold; the bridge to "
             "generation does not close", fontsize=9.5)
fig.tight_layout(); fig.savefig(FIG/"F333-10_evidence.png"); plt.close(fig)
print("figures ->", FIG)
for p in sorted(FIG.glob("*.png")): print("  ", p.name)
