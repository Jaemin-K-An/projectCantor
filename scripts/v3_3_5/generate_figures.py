import sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cantor_guard_v335.affine_coordinate import AffineCoordinate
from cantor_guard_v335.certificate import eps_z_affine, logistic_exact
from cantor_guard_v334.certified_geometry import M_n, rho_max
FIG = pathlib.Path("figures/v3_3_5"); FIG.mkdir(parents=True, exist_ok=True)
TAB = pathlib.Path("results/v3_3_5/tables")
plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
CAN = "#d93025"; RC = 1/3
cert = json.loads((TAB/"certificate_summary.json").read_text())
W, ANCH = cert["W"], cert["anchor"]

# F335-01/02: coordinate maps and slopes
fig, ax = plt.subplots(1, 2, figsize=(11.4, 3.9))
z = np.linspace(ANCH - W, ANCH + W, 600)
C = AffineCoordinate(ANCH, W)
g, s = 0.7, 0.9258
rl = 1/(1+np.exp(np.clip(g*(z-ANCH)/s, -60, 60)))
ax[0].plot(z, C.r(z), color="#1a73e8", lw=2, label="affine (V3.3.5)")
ax[0].plot(z, 1-rl, color="#9aa0a6", lw=2, ls="--", label="logistic (V3.3.4)")
ax[0].set_xlabel("residual projection z"); ax[0].set_ylabel("safety coordinate r")
ax[0].legend(fontsize=8); ax[0].set_title("F335-01 — coordinate maps", fontsize=9)
da = np.abs(np.diff(C.r(z)))/np.abs(np.diff(z))
dl = np.abs(np.diff(1-rl))/np.abs(np.diff(z))
ax[1].plot(z[:-1], da, color="#1a73e8", lw=2, label=f"affine = 1/(2W) = {1/(2*W):.4f}")
ax[1].plot(z[:-1], dl, color="#9aa0a6", lw=2, ls="--", label="logistic (varies)")
ax[1].set_xlabel("z"); ax[1].set_ylabel("|dr/dz|"); ax[1].legend(fontsize=8)
ax[1].set_title("F335-02 — Theorem CP: only the affine map has\n"
                "a position-independent slope", fontsize=9)
fig.tight_layout(); fig.savefig(FIG/"F335-01_02_coordinate.png"); plt.close(fig)

# F335-03/04/05: M_3, affine certificate, and the logistic control
fig, ax = plt.subplots(1, 3, figsize=(14.2, 3.8))
r = np.linspace(0.05, 0.49, 3000)
ax[0].plot(r, M_n(r, 3), color="#1a73e8", lw=1.8)
ax[0].plot(RC, 1/27, "o", color=CAN, ms=8)
ax[0].axvline(RC, color=CAN, ls="--", lw=1)
ax[0].set_xlabel(r"$\rho$"); ax[0].set_ylabel(r"$M_3=\rho^2(1-2\rho)$")
ax[0].set_title("F335-03 — unique max at 1/3", fontsize=9)
for n, c in zip((2, 3, 5), ("#9aa0a6", "#1a73e8", "#34a853")):
    e = np.array([eps_z_affine(x, n, W) for x in r])
    ax[1].plot(r, e/e.max(), color=c, label=f"n={n}")
    ax[1].axvline(rho_max(n), color=c, ls=":", lw=1)
ax[1].axvline(RC, color=CAN, ls="--", lw=1)
ax[1].set_xlabel(r"$\rho$"); ax[1].set_ylabel(r"$\epsilon_z^A$ (normalised)")
ax[1].legend(fontsize=8)
ax[1].set_title(r"F335-04 — exact affine certificate follows"
                "\n" r"$(n-1)/(2n)$: 1/4, 1/3, 2/5", fontsize=9)
A = np.array([eps_z_affine(x, 3, W) for x in r])
L = np.array([logistic_exact(x, 3, s, g) for x in r])
ax[2].plot(r, A/A.max(), color="#1a73e8", lw=2, label=f"affine, argmax {r[A.argmax()]:.3f}")
ax[2].plot(r, L/L.max(), color="#9aa0a6", lw=2, ls="--",
           label=f"logistic, argmax {r[L.argmax()]:.3f}")
ax[2].axvline(RC, color=CAN, ls="--", lw=1)
ax[2].set_xlabel(r"$\rho$"); ax[2].legend(fontsize=8)
ax[2].set_title("F335-05 — the coordinate warp is what\nmoved V3.3.4's optimum",
                fontsize=9)
fig.tight_layout(); fig.savefig(FIG/"F335-03_05_certificate.png"); plt.close(fig)

# F335-06/07: G1 dose-response, global vs G1-only
dev = pd.read_csv("results/v3_3_5/raw/g1_dose_D_beh_g1_dev.csv")
con = pd.read_csv("results/v3_3_5/raw/g1_dose_D_beh_g1_confirm.csv")
b333 = pd.read_csv("results/v3_3_3/raw/behavioral_dose_response.csv")
fig, ax = plt.subplots(1, 2, figsize=(11.4, 3.9))
for d, lab, c in ((dev, "G1-only DEV", "#9aa0a6"), (con, "G1-only CONFIRM", "#1a73e8")):
    gg = d.groupby("z_g1").refusal.mean()
    ax[0].plot(gg.index, gg.values, "o-", ms=4, color=c, label=lab)
ax[0].axhline(0.5, color="k", ls=":", lw=.9)
ax[0].set_xlabel(r"realised $z_{G1}$"); ax[0].set_ylabel("refusal rate")
ax[0].legend(fontsize=8)
ax[0].set_title("F335-06 — G1-only dose barely moves refusal\n"
                "(slope 0.035, CI 24.7σ → UNIDENTIFIABLE)", fontsize=9)
gA = b333.groupby("z").refusal.mean(); gB = con.groupby("z_g1").refusal.mean()
ax[1].plot(gA.index, gA.values, "o-", ms=3, color="#34a853", label="V3.3.3 GLOBAL dose")
ax[1].plot(gB.index, gB.values, "o-", ms=3, color="#1a73e8", label="V3.3.5 G1-ONLY dose")
ax[1].axhline(0.5, color="k", ls=":", lw=.9)
ax[1].set_xlabel("realised projection"); ax[1].set_ylabel("refusal rate")
ax[1].legend(fontsize=8)
ax[1].set_title("F335-07 — the boundary V3.3.3 found needed\n"
                "intervention at EVERY forward", fontsize=9)
fig.tight_layout(); fig.savefig(FIG/"F335-06_07_g1_dose.png"); plt.close(fig)

# F335-10/11: certificate attack
ca = pd.read_csv("results/v3_3_5/raw/certificate_attack.csv")
fig, ax = plt.subplots(1, 2, figsize=(11.4, 3.9))
g3 = ca.groupby("lam").agg(sw=("direct_switch", "sum"), gc=("guard_capture", "sum"),
                           oc=("outside_capture", "sum"), ni=("n_interior", "sum"))
ax[0].plot(g3.index, g3.sw/g3.ni, "o-", color="#d93025", ms=5, label="direct switch")
ax[0].plot(g3.index, g3.gc/g3.ni, "o-", color="#1a73e8", ms=5, label="guard capture")
ax[0].plot(g3.index, g3.oc/g3.ni, "o-", color="#9aa0a6", ms=5, label="outside")
ax[0].axvline(1.0, color="k", ls="--", lw=1.2)
ax[0].annotate("certificate", (1.0, 0.5), rotation=90, fontsize=8, ha="right")
ax[0].set_xlabel(r"$\epsilon/\epsilon_{cert}$"); ax[0].set_ylabel("rate")
ax[0].legend(fontsize=8)
ax[0].set_title("F335-10/11 — 0 direct switches below the certificate;\n"
                "just above, the guard catches them", fontsize=9)
ec = ca.groupby("rho").eps_cert.first()
cols = [CAN if abs(x-RC) < 1e-9 else "#1a73e8" for x in ec.index]
ax[1].bar(range(len(ec)), ec.values, color=cols)
ax[1].set_xticks(range(len(ec))); ax[1].set_xticklabels([f"{x:.3f}" for x in ec.index],
                                                        fontsize=8)
ax[1].set_xlabel(r"$\rho$"); ax[1].set_ylabel(r"$\epsilon_{cert}=2W\,M_3(\rho)$")
ax[1].set_title("F335-04b — Cantor has the largest exact\ncertified radius", fontsize=9)
fig.tight_layout(); fig.savefig(FIG/"F335-10_11_attack.png"); plt.close(fig)

# F335-16: evidence chain
fig, ax = plt.subplots(figsize=(10.6, 2.7))
labs = ["Theorem CP\naffine forced", "exact certificate\nargmax = 1/3",
        "certificate\n0 violations", "G1 causal\nboundary", "generation\nvalidation"]
vals = [1, 1, 1, 0, 0]
txt = ["PROVED", "PROVED", "VALIDATED", "UNIDENTIFIABLE", "NOT RUN"]
ax.imshow(np.array(vals).reshape(1, -1), cmap="RdYlGn", vmin=-.4, vmax=1.4, aspect="auto")
ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=8); ax.set_yticks([])
for i, t in enumerate(txt):
    ax.text(i, 0, t, ha="center", va="center", fontsize=8.5, weight="bold")
ax.set_title("F335-16 — the geometry and its certificate close; the causal "
             "anchor does not", fontsize=9.5)
fig.tight_layout(); fig.savefig(FIG/"F335-16_evidence.png"); plt.close(fig)
print("figures ->", FIG)
for p in sorted(FIG.glob("*.png")): print("  ", p.name)
