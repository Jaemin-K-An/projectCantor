"""V3.3 figures."""
import sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = pathlib.Path("results/v3_3/tables"); FIG = pathlib.Path("figures/v3_3")
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"cantor_recursive": "#1a73e8", "recursive_non_cantor": "#8430ce",
     "periodic_procedural": "#34a853", "shuffled_seeded": "#d93025",
     "center_anchored_seeded": "#f9ab00", "shuffled_explicit": "#9aa0a6",
     "learned_minimax_explicit": "#5f6368"}
LBL = {k: k.replace("_", " ") for k in C}

st = pd.read_csv(T / "bench_structure.csv")
ce = pd.read_csv(T / "bench_certification.csv")
ev = pd.read_csv(T / "bench_symbolic_eval.csv")
tr = pd.read_csv(T / "bench_scale_transfer_all.csv")
gi = pd.read_csv(T / "bench_general_ifs.csv")
post = json.loads((T / "posthoc_periodic_fairness.json").read_text())

def _line(ax, df, x, y, logy=True):
    for f, g in df.groupby("family"):
        ax.plot(g[x], g[y], "o-", ms=3, lw=1.4, color=C.get(f, "k"),
                label=LBL.get(f, f))
    if logy: ax.set_yscale("log")

# V33-04/05/06 component count, description bits, materialised memory
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
axes[0].semilogy(st[st.family == "cantor_recursive"].n,
                 st[st.family == "cantor_recursive"].n_components,
                 "o-", ms=3, color="k")
axes[0].set_xlabel("depth n"); axes[0].set_ylabel("materialised components")
axes[0].set_title("$2^n-1$ gaps", fontsize=9)
_line(axes[1], st, "n", "M1_canonical_bits")
axes[1].set_xlabel("depth n"); axes[1].set_ylabel("canonical description (bits)")
axes[1].set_title("M1 — Cantor is NOT the shortest", fontsize=9)
_line(axes[2], st, "n", "M3_point_query_words")
axes[2].set_xlabel("depth n"); axes[2].set_ylabel("point-query resident words")
axes[2].set_title("M3 — closed-form vs materialised", fontsize=9)
axes[2].legend(fontsize=6.5, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "V33-04_05_06_structure.png"); plt.close(fig)

# V33-08 certificate obligations, sealed vs post-hoc
fig, ax = plt.subplots(figsize=(7.2, 4.2))
_line(ax, ce, "n", "assertions")
n_last = int(st.n.max())
corr = 4 * n_last + 1 + 2 + (n_last + 1)
ax.plot([n_last], [corr], "*", ms=15, color="#34a853")
ax.annotate("periodic, post-hoc\ncorrected accounting",
            (n_last, corr), textcoords="offset points", xytext=(-95, 18),
            fontsize=7.5, color="#34a853")
ax.set_xlabel("depth n"); ax.set_ylabel("proof obligations")
ax.set_title("M2 — certification cost (sealed accounting; star = fairness fix)",
             fontsize=9)
ax.legend(fontsize=7, loc="center left")
fig.tight_layout(); fig.savefig(FIG / "V33-08_certificate_obligations.png"); plt.close(fig)

# V33-07 point-query time and memory, symbolic vs explicit
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
axes[0].semilogy(ev.n, ev.symbolic_seconds, "o-", ms=3, label="symbolic", color="#1a73e8")
axes[0].semilogy(ev.n, ev.explicit_seconds, "s-", ms=3, label="explicit", color="#d93025")
axes[0].set_xlabel("depth n"); axes[0].set_ylabel("seconds / 2000 queries")
axes[0].legend(fontsize=8); axes[0].set_title("point-query time", fontsize=9)
axes[1].semilogy(ev.n, ev.symbolic_peak_bytes, "o-", ms=3, label="symbolic", color="#1a73e8")
axes[1].semilogy(ev.n, ev.explicit_peak_bytes, "s-", ms=3, label="explicit", color="#d93025")
axes[1].set_xlabel("depth n"); axes[1].set_ylabel("peak bytes")
axes[1].legend(fontsize=8); axes[1].set_title("peak memory", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "V33-07_symbolic_vs_explicit.png"); plt.close(fig)

# V33-10 scale transfer -- the one real separator
fig, ax = plt.subplots(figsize=(7.2, 4.2))
_line(ax, tr, "n", "E_scale_rel")
ax.axhline(1e-5, color="k", ls="--", lw=.9)
ax.annotate("exact-transfer threshold", (2.2, 1.5e-5), fontsize=7.5)
ax.set_xlabel("depth n"); ax.set_ylabel(r"relative $E_{scale}$")
ax.set_title("V33-10 — exact scale transfer holds for RECURSIVE families only\n"
             "(periodic fails at 100% error, like the shuffles)", fontsize=9)
ax.legend(fontsize=7, loc="center right")
fig.tight_layout(); fig.savefig(FIG / "V33-10_scale_transfer.png"); plt.close(fig)

# V33-15/16 (b,rho) map with Cantor located
fig, ax = plt.subplots(figsize=(6.6, 4.6))
sc = ax.scatter(gi.rho, gi.alpha_sensitivity, c=gi.b, cmap="viridis", s=14)
cb = fig.colorbar(sc, ax=ax); cb.set_label("branch factor b", fontsize=8)
cp = gi[gi.is_cantor]
ax.plot(cp.rho, cp.alpha_sensitivity, "*", ms=20, color="#d93025")
ax.annotate("Cantor (2, 1/3)", (float(cp.rho.iloc[0]), float(cp.alpha_sensitivity.iloc[0])),
            textcoords="offset points", xytext=(12, 6), fontsize=8, color="#d93025")
ax.set_yscale("log"); ax.set_xlabel(r"contraction ratio $\rho$")
ax.set_ylabel(r"sensitivity scale $1/(b\rho^2)$")
ax.set_title("V33-15/16 — Cantor's position in the recursive family", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "V33-15_16_general_ifs.png"); plt.close(fig)

# V33-17 evidence matrix
fig, ax = plt.subplots(figsize=(8.4, 3.4))
fams = ["cantor_recursive", "recursive_non_cantor", "periodic_procedural",
        "shuffled_seeded", "shuffled_explicit"]
crit = ["short description\n(M1)", "cheap certificate\n(M2, corrected)",
        "O(1) point query\n(M3)", "exact scale\ntransfer"]
last = st[st.n == st.n.max()].set_index("family")
tro = tr.groupby("family").E_scale_rel.max()
M = np.zeros((len(fams), len(crit)))
best_m1 = last.M1_canonical_bits.min()
for i, f in enumerate(fams):
    M[i, 0] = 1 if last.loc[f, "M1_canonical_bits"] <= best_m1 * 2 else 0
    M[i, 1] = 1 if f in ("cantor_recursive", "recursive_non_cantor",
                         "periodic_procedural") else 0
    M[i, 2] = 1 if last.loc[f, "M3_point_query_words"] < 100 else 0
    M[i, 3] = 1 if tro.get(f, 1e9) < 1e-5 else 0
ax.imshow(M, cmap="Greens", vmin=0, vmax=1.6, aspect="auto")
ax.set_xticks(range(len(crit))); ax.set_xticklabels(crit, fontsize=8)
ax.set_yticks(range(len(fams)))
ax.set_yticklabels([LBL[f] for f in fams], fontsize=8)
for i in range(len(fams)):
    for j in range(len(crit)):
        ax.text(j, i, "YES" if M[i, j] else "no", ha="center", va="center",
                fontsize=8, color="#0b3d16" if M[i, j] else "#777")
ax.set_title("V33-17 — evidence matrix: only the last column separates,\n"
             "and it separates RECURSION, not Cantor", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "V33-17_evidence_matrix.png"); plt.close(fig)
print("figures ->", FIG)
for p in sorted(FIG.glob("*.png")): print("  ", p.name)
