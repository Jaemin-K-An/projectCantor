"""V3.2 figures. Every interval plotted is a GOAL-CLUSTERED interval."""
import sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cantor_guard_v32.cluster_stats import (cluster_bootstrap_by_goal,
                                            naive_cell_bootstrap)

FIG = pathlib.Path("figures/v3_2"); FIG.mkdir(parents=True, exist_ok=True)
SESOI = 0.03
KEYS = ["attack", "delta", "eps", "pid"]
CANTOR = "T7_cantor"
MATCHED = ["T5_shuffled", "T6_center_anchored", "T4_periodic", "T3_wide_central"]
BASE = ["T0_none", "T1_true_constant", "T2_global_smooth", "T8_minimax"]
plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def paired(df, a, b, score):
    g = df.groupby(["family"] + KEYS, as_index=False)[score].mean()
    piv = g.pivot_table(index=KEYS, columns="family", values=score)
    if a not in piv or b not in piv:
        return pd.DataFrame()
    return piv[[a, b]].dropna().reset_index().rename(
        columns={a: "score_a", b: "score_b"})


# ---- FIG 1: the pseudoreplication correction, on V3.1's own data ----------
aud = pd.read_csv("results/v3_2/tables/v31_pseudoreplication_audit.csv")
aud = aud[aud.status == "OK"].sort_values("mean_diff")
fig, ax = plt.subplots(figsize=(7.4, 4.0))
y = np.arange(len(aud))
ax.errorbar(aud.mean_diff, y - 0.16,
            xerr=[aud.mean_diff - aud.naive_lo, aud.naive_hi - aud.mean_diff],
            fmt="o", ms=4, color="#9aa0a6", capsize=3, label="V3.1 naive (cells)")
ax.errorbar(aud.mean_diff, y + 0.16,
            xerr=[aud.mean_diff - aud.cluster_lo, aud.cluster_hi - aud.mean_diff],
            fmt="s", ms=4, color="#1a73e8", capsize=3, label="goal-clustered")
ax.axvline(0, color="k", lw=.8)
ax.axvspan(-SESOI, SESOI, color="#34a853", alpha=.10, label=f"SESOI ±{SESOI}")
ax.set_yticks(y); ax.set_yticklabels(aud.family)
ax.set_xlabel("mean safety difference (Cantor − control)")
ax.set_title("V3.1 re-analysed: clustering by goal widens the weak-baseline\n"
             "intervals and narrows the matched-control ones", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(FIG / "fig1_pseudoreplication.png"); plt.close(fig)

# ---- the remaining figures need the final test ---------------------------
for model in ("qwen2.5-0.5b-instruct", "olmo2-1b-instruct"):
    src = pathlib.Path(f"results/v3_2/raw/v32_final_{model}.csv")
    if not src.exists():
        print(f"[skip] {src} absent"); continue
    df = pd.read_csv(src)
    scores = [c for c in ("safe_lex32", "safe_ext") if c in df.columns]

    # FIG 2: forest plot of every comparison, both scorers
    fig, axes = plt.subplots(1, len(scores), figsize=(5.2 * len(scores), 4.4),
                             squeeze=False)
    for ax, sc in zip(axes[0], scores):
        rows = []
        for alt in MATCHED + BASE:
            m = paired(df, CANTOR, alt, sc)
            if m.empty:
                continue
            st = cluster_bootstrap_by_goal(m, "score_a", "score_b", n_boot=6000)
            rows.append((alt, st["mean_diff"], st["ci_lo"], st["ci_hi"],
                         alt in MATCHED))
        rows.sort(key=lambda r: r[1])
        yy = np.arange(len(rows))
        for i, (nm, d, lo, hi, ism) in enumerate(rows):
            c = "#1a73e8" if ism else "#9aa0a6"
            ax.plot([lo, hi], [i, i], color=c, lw=2)
            ax.plot(d, i, "o", color=c, ms=5)
        ax.axvline(0, color="k", lw=.8)
        ax.axvspan(-SESOI, SESOI, color="#34a853", alpha=.10)
        ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
        ax.set_xlabel(f"Cantor − control  ({sc})")
        ax.set_title(sc, fontsize=9)
    fig.suptitle(f"{model}: goal-clustered 95% CIs (blue = width/energy-matched)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / f"fig2_forest_{model}.png"); plt.close(fig)

    # FIG 3: per-goal effects -- shows WHY clustering matters
    sc = scores[0]
    m = paired(df, CANTOR, "T5_shuffled", sc)
    if not m.empty:
        per = (m.assign(d=m.score_a - m.score_b).groupby("pid").d.mean()
                 .sort_values())
        fig, ax = plt.subplots(figsize=(7.4, 3.4))
        ax.bar(range(len(per)), per.values,
               color=["#d93025" if v < 0 else "#1a73e8" for v in per.values])
        ax.axhline(0, color="k", lw=.8)
        ax.axhline(per.mean(), color="#34a853", ls="--", lw=1.2,
                   label=f"mean {per.mean():+.4f}")
        ax.set_xlabel("held-out goal (sorted)")
        ax.set_ylabel("Cantor − shuffled")
        ax.set_title(f"{model}: per-goal effect, between-goal SD = "
                     f"{per.std(ddof=1):.4f}", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(FIG / f"fig3_per_goal_{model}.png")
        plt.close(fig)

    # FIG 4: realised budget on D_test vs the frozen target (generalisation)
    cfg = json.loads(pathlib.Path(f"configs/v3_2/frozen_{model}.json").read_text())
    ach = df.groupby("family").C_rms.mean().sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh(range(len(ach)), ach.values, color="#1a73e8")
    ax.axvline(cfg["target_C_rms"], color="k", ls="--", lw=1,
               label=f"target {cfg['target_C_rms']}")
    for f in (0.97, 1.03):
        ax.axvline(cfg["target_C_rms"] * f, color="#d93025", ls=":", lw=.9)
    ax.set_yticks(range(len(ach))); ax.set_yticklabels(ach.index, fontsize=8)
    ax.set_xlabel("realised $C_{rms}$ on D_test (generalisation, NOT retuned)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / f"fig4_budget_{model}.png"); plt.close(fig)

    # FIG 5: scorer agreement
    if len(scores) == 2:
        fam = df.groupby("family")[scores].mean()
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        ax.scatter(fam[scores[0]], fam[scores[1]], color="#1a73e8")
        for n, r in fam.iterrows():
            ax.annotate(n.replace("T", ""), (r[scores[0]], r[scores[1]]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        lo = float(min(fam.min())); hi = float(max(fam.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=.8)
        ax.set_xlabel(scores[0]); ax.set_ylabel(scores[1])
        ax.set_title(f"{model}: scorer agreement by family", fontsize=9)
        fig.tight_layout(); fig.savefig(FIG / f"fig5_scorers_{model}.png")
        plt.close(fig)
    print(f"[figures] {model} done")

print("figures ->", FIG)
