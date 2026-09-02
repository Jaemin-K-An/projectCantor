import sys, json, pathlib
sys.path.insert(0, "llm/src")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cantor_guard_v334.certified_geometry import M_n, rho_max, M_n_max
from cantor_guard_v334.certificate import eps_z_lipschitz, eps_z_exact
FIG = pathlib.Path("figures/v3_3_4"); FIG.mkdir(parents=True, exist_ok=True)
TAB = pathlib.Path("results/v3_3_4/tables")
B = json.loads(pathlib.Path("results/v3_3_3/tables/behavioral_boundary.json").read_text())
V = json.loads(pathlib.Path("results/v3_3_2/tables/phase_calibration_qwen2.5-0.5b-instruct.json").read_text())
SIG, GAM, TAU = B["sigma_G1"], V["gamma"], B["tau_beh"]
plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
CAN = "#d93025"; RC = 1/3; r = np.linspace(0.005, 0.495, 3000)

# F334-01/04: M_3 and the depth law
fig, ax = plt.subplots(1, 2, figsize=(11.4, 3.9))
ax[0].plot(r, M_n(r, 3), color="#1a73e8", lw=1.8)
ax[0].plot(RC, 1/27, "o", color=CAN, ms=9)
ax[0].annotate(r"$\rho=1/3$, $M_3=1/27$", (RC, 1/27), xytext=(12, -14),
               textcoords="offset points", color=CAN, fontsize=8.5)
ax[0].set_xlabel(r"$\rho$"); ax[0].set_ylabel(r"$M_3(\rho)=\rho^2(1-2\rho)$")
ax[0].set_title(r"F334-01 — $M_3'=2\rho(1-3\rho)$: unique maximum at 1/3", fontsize=9)
for n, c in zip((2, 3, 5), ("#9aa0a6", "#1a73e8", "#34a853")):
    ax[1].plot(r, M_n(r, n)/M_n_max(n), color=c, label=f"n={n}")
    ax[1].axvline(rho_max(n), color=c, ls=":", lw=1)
    ax[1].annotate(f"{rho_max(n):.2f}", (rho_max(n), 1.02), color=c, fontsize=8,
                   ha="center")
ax[1].set_xlabel(r"$\rho$"); ax[1].set_ylabel(r"$M_n/\max M_n$")
ax[1].legend(fontsize=8)
ax[1].set_title(r"F334-04 — depth law $\rho_{max}(n)=(n-1)/(2n)$:"
                "\n1/4, 1/3, 2/5", fontsize=9)
fig.tight_layout(); fig.savefig(FIG/"F334-01_04_theorem.png"); plt.close(fig)

# F334-02/03: certificates
fig, ax = plt.subplots(figsize=(6.6, 4.0))
L = np.array([eps_z_lipschitz(x, 3, SIG, GAM) for x in r])
E = np.array([eps_z_exact(x, 3, TAU, SIG, GAM) for x in r])
ax.plot(r, L, color="#1a73e8", label=r"$\epsilon_{cert}$ Lipschitz")
ax.plot(r, E, color="#34a853", label=r"$\epsilon_{cert}$ exact (inverse-logit)")
ax.plot(r[L.argmax()], L.max(), "o", color="#1a73e8", ms=8)
ax.plot(r[E.argmax()], E.max(), "o", color="#34a853", ms=8)
ax.axvline(RC, color=CAN, ls="--", lw=1.2)
ax.annotate(f"Lipschitz argmax\n= {r[L.argmax()]:.3f}", (r[L.argmax()], L.max()),
            xytext=(8, -28), textcoords="offset points", fontsize=7.5, color="#1a73e8")
ax.annotate(f"exact argmax\n= {r[E.argmax()]:.3f}", (r[E.argmax()], E.max()),
            xytext=(-56, 6), textcoords="offset points", fontsize=7.5, color="#34a853")
ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"certified radius (projection units)")
ax.legend(fontsize=8)
ax.set_title("F334-02/03 — Cantor maximises the Lipschitz certificate exactly;\n"
             "the logit warp moves the EXACT optimum off 1/3", fontsize=9)
fig.tight_layout(); fig.savefig(FIG/"F334-02_03_certificates.png"); plt.close(fig)

# F334-08/11: policy switch and depth shift
sb = pd.read_csv("results/v3_3_4/raw/certificate_attack_dev.csv")
ds = pd.read_csv("results/v3_3_4/raw/depthshift_common_dev.csv")
fig, ax = plt.subplots(1, 2, figsize=(11.6, 3.9))
g3 = sb[sb.n == 3].groupby("lam").switch_rate.mean()
ax[0].plot(g3.index, g3.values, "o-", color="#1a73e8", ms=5)
ax[0].axvline(1.0, color=CAN, ls="--", lw=1.4)
ax[0].annotate("certificate", (1.0, g3.values.max()*0.7), color=CAN, fontsize=8,
               rotation=90, ha="right")
ax[0].set_xlabel(r"$\epsilon\,/\,\epsilon_{cert}$"); ax[0].set_ylabel("direct policy-switch rate")
ax[0].set_title("F334-08 — 0 violations below the certificate\n(468 configs)",
                fontsize=9)
for n, c in zip((2, 3, 5), ("#9aa0a6", "#1a73e8", "#34a853")):
    s = ds[ds.n == n]
    ax[1].plot(s.rho, s.median_switch_eps, "o-", ms=4, color=c, label=f"n={n}")
    ax[1].axvline(rho_max(n), color=c, ls=":", lw=1)
ax[1].set_xlabel(r"$\rho$"); ax[1].set_ylabel(r"median switch $\epsilon$ (common grid)")
ax[1].legend(fontsize=8)
ax[1].set_title("F334-11 — depth-shift NOT supported\n(corr −0.064 on a "
                "non-circular grid)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG/"F334-08_11_transfer.png"); plt.close(fig)

# F334-12/15/16: generation
auc = pd.read_csv(TAB/"generation_auc.csv")
cmp_ = pd.read_csv(TAB/"generation_comparisons.csv")
meta = json.loads((TAB/"generation_meta.json").read_text())
fig, ax = plt.subplots(1, 3, figsize=(14.2, 3.9))
b = auc[auc.centre == "tau_beh"].groupby("rho").auc.agg(["mean", "sem"])
ax[0].errorbar(b.index, b["mean"], yerr=b["sem"], fmt="o", ms=6, capsize=3,
               color="#1a73e8")
ax[0].axvline(RC, color=CAN, ls="--", lw=1)
ax[0].set_xlabel(r"$\rho$"); ax[0].set_ylabel("robust safety AUC")
ax[0].set_title(r"F334-12 — AUC vs $\rho$ ($\tau_{beh}$-centred)", fontsize=9)
y = np.arange(len(cmp_))
ax[1].errorbar(cmp_.mean_diff, y,
               xerr=[cmp_.mean_diff-cmp_.simult_lo, cmp_.simult_hi-cmp_.mean_diff],
               fmt="o", ms=5, color="#1a73e8", capsize=3)
ax[1].axvline(0, color="k", lw=.8)
ax[1].axvspan(-0.02, 0.02, color="#34a853", alpha=.12)
ax[1].set_yticks(y); ax[1].set_yticklabels([f"1/3 vs {x:.3f}" for x in cmp_.rho_other],
                                           fontsize=8)
ax[1].set_xlabel("AUC difference")
ax[1].set_title("F334-13 — all inside SESOI ±0.02\n(max-T simultaneous)", fontsize=9)
fc = [meta[k]["final_C_rms"] for k in meta]
ax[2].barh(range(len(fc)), fc, color="#1a73e8")
ax[2].axvline(0.02, color="k", ls="--", lw=1)
for f in (0.97, 1.03):
    ax[2].axvline(0.02*f, color=CAN, ls=":", lw=.9)
ax[2].set_yticks(range(len(meta))); ax[2].set_yticklabels(list(meta), fontsize=7)
ax[2].set_xlabel(r"actual $C_{rms}$ on D_final")
ax[2].set_title("F334-15 — D_FINAL budget gate\n(all 6 matched)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG/"F334-12_15_generation.png"); plt.close(fig)

# F334-18: evidence chain
V4 = json.loads((TAB/"verdict_v334.json").read_text())
fig, ax = plt.subplots(figsize=(10.4, 2.6))
labs = ["M1 theorem\n$M_3$ max at 1/3", "certificate\n0 violations",
        "exact cert\nargmax 0.296", "depth law\ntransfer", "generation\ngain"]
vals = [1, 1, 0, 0, 0]
txt = ["PROVED", "VALIDATED", "MOVES OFF 1/3", "NOT SUPPORTED", "EQUIVALENT"]
ax.imshow(np.array(vals).reshape(1, -1), cmap="RdYlGn", vmin=-.4, vmax=1.4, aspect="auto")
ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=8); ax.set_yticks([])
for i, t in enumerate(txt):
    ax.text(i, 0, t, ha="center", va="center", fontsize=8.5, weight="bold")
ax.set_title("F334-18 — the geometry and its certificate hold exactly; "
             "the transfer to behaviour does not", fontsize=9.5)
fig.tight_layout(); fig.savefig(FIG/"F334-18_evidence.png"); plt.close(fig)
print("figures ->", FIG)
for p in sorted(FIG.glob("*.png")): print("  ", p.name)
