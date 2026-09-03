import sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
FIG = pathlib.Path("figures/v3_3_5b"); FIG.mkdir(parents=True, exist_ok=True)
TAB = pathlib.Path("results/v3_3_5b/tables")
plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
CAN = "#d93025"
d = pd.read_csv("results/v3_3_5b/raw/temporal_D_temporal_confirm.csv")
base = float(d[d.B2_target == 0].refusal.mean())
c = pd.read_csv(TAB/"temporal_contrasts.csv")
rg = json.loads((TAB/"regime_gap.json").read_text())

# F1: the confound, with warning
fig, ax = plt.subplots(1, 2, figsize=(11.4, 3.8))
ph = pd.DataFrame(json.loads(pathlib.Path(
    "results/v3_3_5a/tables/phase_causality.json").read_text())["rows"])
ax[0].bar(range(3), ph.beta_std, color=["#1a73e8","#34a853","#9aa0a6"])
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(
    [x.split(" (")[0] for x in ph.phase], fontsize=9)
ax[0].set_ylabel(r"$\beta_{std}$")
ax[0].set_title("F1 — V3.3.5a apparent phase leverage\n"
                "UNMATCHED BUDGETS: GLOBAL touched 48 forwards", fontsize=9,
                color=CAN)
q = [rg["historical_global_q_estimate"][k] for k in sorted(rg["historical_global_q_estimate"])]
ax[1].axhline(rg["cap"], color=CAN, ls="--", lw=1.4, label=f"pre-declared cap {rg['cap']}")
ax[1].axhline(rg["first_effective_q_nondegenerate"], color="#34a853", ls=":", lw=1.4,
              label=f"first effect q={rg['first_effective_q_nondegenerate']}")
ax[1].bar(range(len(q)), sorted(q), color="#1a73e8")
ax[1].set_xticks(range(len(q))); ax[1].set_xticklabels(
    sorted(rg["historical_global_q_estimate"]), fontsize=8)
ax[1].set_ylabel("q = |dh| / ||h||"); ax[1].legend(fontsize=7.5)
ax[1].set_title("F2 — the historical dose sat 3–10x above the cap;\n"
                "the pre-declared grid was below the effect threshold", fontsize=9)
fig.tight_layout(); fig.savefig(FIG/"F1_F2_confound.png"); plt.close(fig)

# F3: matched-budget response
fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.0))
t = d[d.B2_target > 0].pivot_table(index="B2_target", columns="schedule",
                                   values="refusal")
eff = base - t
cols = {"S1_P0_ONLY": "#d93025", "S2_G1_ONLY": "#f9ab00", "S3_EARLY_2": "#e8710a",
        "S4_EARLY_4": "#1a73e8", "S5_EARLY_8": "#34a853", "S6_ALL_K": "#9aa0a6",
        "S7_LATE_4": "#5f6368"}
for s in eff.columns:
    ax[0].plot(eff.index, eff[s], "o-", ms=5, color=cols[s],
               lw=2.2 if s == "S1_P0_ONLY" else 1.3, label=s.replace("_", " "))
ax[0].set_xlabel(r"matched trajectory budget $B_2$")
ax[0].set_ylabel(r"causal effect $|\Delta P|$ from baseline")
ax[0].legend(fontsize=7)
ax[0].set_title("F3 — at EQUAL energy, concentrating at P0 wins;\n"
                "spreading dilutes", fontsize=9)
sub = c[c.B2 == 0.8]
y = np.arange(len(sub))
ax[1].errorbar(sub.mean_diff, y,
               xerr=[sub.mean_diff-sub.simult_lo, sub.simult_hi-sub.mean_diff],
               fmt="o", ms=6, color="#1a73e8", capsize=3)
ax[1].axvline(0, color="k", lw=.9)
ax[1].axvspan(-0.03, 0.03, color="#34a853", alpha=.12)
ax[1].set_yticks(y); ax[1].set_yticklabels(
    [x.replace("_ONLY","").replace("S4_","").replace("S5_","").replace("S1_","").replace("S2_","")
     for x in sub.contrast], fontsize=8)
ax[1].set_xlabel("distributed − single (effect scale)")
ax[1].set_title(r"F4/F5 — max-T simultaneous at $B_2=0.8$;"
                "\nnegative = distribution is WORSE", fontsize=9)
fig.tight_layout(); fig.savefig(FIG/"F3_F5_matched.png"); plt.close(fig)

# F6: evidence chain
V = json.loads((TAB/"verdict_v335b.json").read_text())
fig, ax = plt.subplots(figsize=(10.6, 2.7))
labs = ["Cantor math\n(frozen)", "budget matched\nexactly", "distributed >\nsingle-state",
        "trajectory\ncoordinate", "final generation"]
vals = [1, 1, 0, 0, 0]
txt = ["VALID", "0/1680 off", "NO — WORSE", "NOT BUILT", "NOT RUN"]
ax.imshow(np.array(vals).reshape(1, -1), cmap="RdYlGn", vmin=-.4, vmax=1.4, aspect="auto")
ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=8); ax.set_yticks([])
for i, tt in enumerate(txt):
    ax.text(i, 0, tt, ha="center", va="center", fontsize=8.5, weight="bold")
ax.set_title("F14 — the historical GLOBAL advantage was accumulation, "
             "not temporal distribution", fontsize=9.5)
fig.tight_layout(); fig.savefig(FIG/"F14_evidence.png"); plt.close(fig)
print("figures ->", FIG)
for p in sorted(FIG.glob("*.png")): print("  ", p.name)
