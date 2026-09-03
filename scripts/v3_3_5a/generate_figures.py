import sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
FIG = pathlib.Path("figures/v3_3_5a"); FIG.mkdir(parents=True, exist_ok=True)
TAB = pathlib.Path("results/v3_3_5a/tables")
plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
CAN = "#d93025"

# F1: timing diagram
fig, ax = plt.subplots(figsize=(9.4, 2.4))
ax.axis("off")
xs = [0.06, 0.26, 0.46, 0.66, 0.86]
lab = ["prompt\ntokens", "P0\n(last prompt tok)", "token 1\nchosen",
       "G1\n(first decode)", "token 2"]
col = ["#9aa0a6", "#1a73e8", "#34a853", "#d93025", "#9aa0a6"]
for x, l, c in zip(xs, lab, col):
    ax.add_patch(plt.Rectangle((x-0.075, 0.42), 0.15, 0.3, color=c, alpha=.85))
    ax.text(x, 0.57, l, ha="center", va="center", fontsize=8, color="white")
for i in range(4):
    ax.annotate("", (xs[i+1]-0.08, 0.57), (xs[i]+0.08, 0.57),
                arrowprops=dict(arrowstyle="->", lw=1.3))
ax.annotate("P0 CAN change token 1", (xs[1], 0.30), ha="center", fontsize=8.5,
            color="#1a73e8")
ax.annotate("G1 CANNOT — token 1 already chosen", (xs[3], 0.30), ha="center",
            fontsize=8.5, color=CAN)
ax.set_xlim(0, 1); ax.set_ylim(0.15, 0.85)
ax.set_title("F335a-01 — why placement matters", fontsize=10)
fig.tight_layout(); fig.savefig(FIG/"F335a-01_timing.png"); plt.close(fig)

# F2: P0 vs G1 dose-response
p = pd.read_csv("results/v3_3_5a/raw/p0_dose_D_beh_P0_confirm.csv")
g = pd.read_csv("results/v3_3_5/raw/g1_dose_D_beh_g1_confirm.csv")
b3 = pd.read_csv("results/v3_3_3/raw/behavioral_dose_response.csv")
D = json.loads((TAB/"p0_direction.json").read_text())
fig, ax = plt.subplots(1, 2, figsize=(11.4, 3.9))
gp = p.groupby("dose").agg(r=("refusal","mean"), c=("coherence","mean"))
gg = g.groupby("dose").agg(r=("refusal","mean"), c=("coherence","mean"))
ax[0].plot(gp.index/D["sigma_P0"], gp.r, "o-", ms=5, color="#1a73e8", label="P0-only")
ax[0].plot(gg.index/0.5613, gg.r, "o-", ms=5, color="#34a853", label="G1-only")
ax[0].axhline(0.5, color="k", ls=":", lw=.9)
ax[0].set_xlabel(r"dose ($\sigma_{phase}$)"); ax[0].set_ylabel("refusal rate")
ax[0].legend(fontsize=8)
ax[0].set_title("F335a-02 — both single-state doses are weak;\n"
                "P0 is also NON-MONOTONE", fontsize=9)
ax[1].plot(gp.index/D["sigma_P0"], gp.c, "o-", ms=5, color="#1a73e8", label="P0 coherence")
ax[1].axhline(0.6, color=CAN, ls="--", lw=1)
ax[1].set_xlabel(r"dose ($\sigma_{P0}$)"); ax[1].set_ylabel("coherence")
ax[1].legend(fontsize=8)
ax[1].set_title("F335a-03 — the largest doses destroy the output\n"
                "(coherence 0.00 at −153σ)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG/"F335a-02_03_dose.png"); plt.close(fig)

# F3: phase leverage — the headline
ph = pd.DataFrame(json.loads((TAB/"phase_causality.json").read_text())["rows"])
fig, ax = plt.subplots(1, 2, figsize=(11.2, 3.9))
c = ["#1a73e8", "#34a853", "#d93025"]
ax[0].bar(range(3), ph.beta_std, color=c)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(
    [x.split(" (")[0] for x in ph.phase], fontsize=9)
ax[0].set_ylabel(r"$\beta_{std}$ (log-odds per $\sigma$)")
for i, vv in enumerate(ph.beta_std):
    ax[0].text(i, vv, f"{vv:.4f}", ha="center", va="bottom", fontsize=8.5)
ax[0].set_title("F335a-04 — standardized causal leverage", fontsize=9)
ax[1].bar(range(3), ph.dP_2sigma, color=c)
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(
    [x.split(" (")[0] for x in ph.phase], fontsize=9)
ax[1].set_ylabel(r"$\Delta P$(refusal) over $\pm 1\sigma$")
for i, vv in enumerate(ph.dP_2sigma):
    ax[1].text(i, vv, f"{vv:+.4f}", ha="center", va="bottom", fontsize=8.5)
ax[1].set_title("F335a-05 — all-forward intervention carries\n"
                "~19x the leverage of either single state", fontsize=9)
fig.tight_layout(); fig.savefig(FIG/"F335a-04_05_phase.png"); plt.close(fig)

# F4: evidence chain
V = json.loads((TAB/"verdict_v335a.json").read_text())
fig, ax = plt.subplots(figsize=(10.6, 2.7))
labs = ["Cantor certificate\n(V3.3.5)", "P0 reaches\ntoken-1 logits",
        "P0 ±2σ\nbehavioural gate", "P0 boundary\nidentifiable",
        "final generation"]
vals = [1, 1, 0, 0, 0]
txt = ["VALID", "YES (2.59)", "ZERO SPAN", "IMPRECISE", "NOT RUN"]
ax.imshow(np.array(vals).reshape(1, -1), cmap="RdYlGn", vmin=-.4, vmax=1.4, aspect="auto")
ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=8); ax.set_yticks([])
for i, t in enumerate(txt):
    ax.text(i, 0, t, ha="center", va="center", fontsize=8.5, weight="bold")
ax.set_title("F335a-06 — the placement is mechanically correct; the behavioural "
             "anchor is still missing", fontsize=9.5)
fig.tight_layout(); fig.savefig(FIG/"F335a-06_evidence.png"); plt.close(fig)
print("figures ->", FIG)
for q in sorted(FIG.glob("*.png")): print("  ", q.name)
