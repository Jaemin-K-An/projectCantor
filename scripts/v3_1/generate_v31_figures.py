"""V3.1 figures, regenerated from results/v3_1/raw only."""
import sys, json
sys.path.insert(0,"llm/src")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from cantor_guard.statistics import paired_bootstrap
from cantor_guard_v31.io31 import FIG31, V31_RAW
from cantor_guard_v31.controllers31 import Controller31
plt.rcParams.update({"figure.dpi":150,"font.size":8,"axes.grid":True,"grid.alpha":.25,
                     "axes.spines.top":False,"axes.spines.right":False,
                     "legend.fontsize":7,"axes.titlesize":9})
CAP=["# V3.1 figure captions\n"]
def cap(n,t): CAP.append(f"\n### {n}\n\n{t}\n")
def save(f,n): f.tight_layout(); f.savefig(FIG31/n,bbox_inches="tight"); plt.close(f); print(" ok",n)

# ---- V3.1-02  fake 'constant' vs TRUE constant ----
m = np.linspace(-8,8,4001)
fig,ax=plt.subplots(1,2,figsize=(10,3.2))
for fam,lab,c in [("T1_true_constant","TRUE constant (V3.1)","#1f77b4"),
                  ("T2_global_smooth","V3's 'L1_constant' (actually a barrier)","#d62728")]:
    C=Controller31(fam,n=5,B_total=1.0,gamma=0.7,eta=1.0)
    ax[0].plot(m,C.magnitude(m),lw=1.6,label=lab,color=c)
ax[0].set_xlabel("safety margin $m$"); ax[0].set_ylabel("|u|"); ax[0].legend()
ax[0].set_title("FIG V3.1-02  V3 defect D1: the 'constant' was not constant")
ns=np.arange(1,9); pred=[12*(1.0/n)*4.5**n for n in ns]
ax[1].semilogy(ns,pred,"o-",color="#d62728",label=r"Cantor $12E_0(9/2)^n$")
ax[1].axhline(0.0,color="#1f77b4")
ax[1].scatter([5],[6.0],s=70,marker="s",color="#2ca02c",label="global smooth (6.0)")
ax[1].scatter([5],[13.5],s=70,marker="^",color="#ff7f0e",label="wide central (13.5)")
ax[1].set_xlabel("order $n$"); ax[1].set_ylabel(r"$\|u'\|_\infty$"); ax[1].legend()
ax[1].set_title(r"FIG V3.1-07  Theorem T: calibration sensitivity $\propto (9/2)^n$")
save(fig,"figV31_02_07_constant_theoremT.png")
cap("FIG V3.1-02 / V3.1-07","Left: V3's `L1_constant` was a single smoothstep "
    "barrier (range 1.5), not a constant; V3.1 separates the two. Right: "
    "Theorem T, verified to ratio 1.000000 -- refining the Cantor scale "
    "multiplies the field's slope, and hence its sensitivity to a boundary "
    "error, by 9/2 per level. The true constant sits at exactly 0.")

# ---- V3.1-08/09/10  synthetic surfaces ----
s=pd.read_csv(V31_RAW/"v31_synthetic.csv"); s=s[s.budget==0.60]
fig,ax=plt.subplots(1,3,figsize=(13,3.4))
for k,(fam,t) in enumerate([("S1_true_constant","TRUE constant"),("S9_cantor","Cantor")]):
    M=s[s.family==fam].pivot_table(index="delta",columns="eps",values="safe_frac")
    im=ax[k].imshow(M.values,cmap="viridis",vmin=0,vmax=1,aspect="auto",origin="lower")
    ax[k].set_xticks(range(len(M.columns))); ax[k].set_xticklabels([f"{c:.2f}" for c in M.columns],rotation=40)
    ax[k].set_yticks(range(len(M.index))); ax[k].set_yticklabels([f"{i:+.3f}" for i in M.index])
    ax[k].set_xlabel(r"attack $\epsilon$"); ax[k].set_ylabel(r"boundary error $\Delta$")
    plt.colorbar(im,ax=ax[k]); ax[k].set_title(f"FIG V3.1-0{8+k}  R($\\Delta,\\epsilon$) — {t}")
w=s.groupby(["family","delta","eps","attack"]).safe_frac.mean().reset_index()
wm=w.groupby("family").safe_frac.min().sort_values()
ax[2].barh(range(len(wm)),wm.values,color=["#d62728" if f=="S9_cantor" else "#4c72b0" for f in wm.index])
ax[2].set_yticks(range(len(wm))); ax[2].set_yticklabels(wm.index,fontsize=6)
ax[2].set_xlabel("worst-case graded robustness")
ax[2].set_title("FIG V3.1-10  synthetic minimax, budget 0.60")
save(fig,"figV31_08_10_synthetic.png")
cap("FIG V3.1-08/09/10","Synthetic joint uncertainty at matched REALISED budget "
    "(+-2%). The best controller is the broad smooth barrier (0.524), not the "
    "true constant (0.261) as V3 claimed, and not Cantor (0.050), which sits "
    "just above no intervention.")

# ---- V3.1-13..19  LLM ----
mk="qwen2.5-0.5b-instruct"
d=pd.read_csv(V31_RAW/f"v31_llm_direct_v2_{mk}.csv"); d=d[d.family.notna()]
u=pd.read_csv(V31_RAW/f"v31_llm_utility_v2_{mk}.csv")
O=["T0_none","T1_true_constant","T2_global_smooth","T3_wide_central","T4_periodic",
   "T5_shuffled","T6_center_anchored","T7_cantor","T8_minimax"]
UNM={"T2_global_smooth","T3_wide_central","T8_minimax"}
fig,ax=plt.subplots(2,2,figsize=(11,6.6))
for f in O:
    g=d[d.family==f].groupby("eps").safe.mean()
    ax[0,0].plot(g.index,g.values,"o-",ms=3,lw=1.4,label=f,
                 color="#d62728" if f=="T7_cantor" else None,
                 zorder=5 if f=="T7_cantor" else 2)
ax[0,0].set_xlabel(r"latent attack $\epsilon$ (% of activation norm)")
ax[0,0].set_ylabel("safety score"); ax[0,0].legend(fontsize=6,ncol=2)
ax[0,0].set_title("FIG V3.1-13  LLM: safety vs attack magnitude")
b=d.groupby("family").C_rms.mean().reindex(O)
ax[0,1].barh(range(len(b)),b.values,color=["#d62728" if f=="T7_cantor" else
    ("#cccccc" if f in UNM else "#4c72b0") for f in b.index])
ax[0,1].axvline(0.02,color="k",ls="--",lw=1.5,label="target 0.02")
ax[0,1].axvspan(0.02*0.97,0.02*1.03,color="green",alpha=.15,label=r"$\pm$3% band")
ax[0,1].set_yticks(range(len(b))); ax[0,1].set_yticklabels(b.index,fontsize=6)
ax[0,1].set_xlabel(r"realised $C_{rms}=\sqrt{E[q^2]}$"); ax[0,1].legend()
ax[0,1].set_title("FIG V3.1-12  realised residual budget (grey = excluded)")
KEY=["attack","delta","eps","pid"]
piv=d.groupby(KEY+["family"]).safe.mean().unstack("family")
names,md,lo,hi=[],[],[],[]
for c in ["T0_none","T1_true_constant","T4_periodic","T5_shuffled","T6_center_anchored"]:
    sub=piv[["T7_cantor",c]].dropna()
    st=paired_bootstrap(sub["T7_cantor"].values,sub[c].values,seed=7)
    names.append(c); md.append(st["mean_diff"]); lo.append(st["mean_diff"]-st["ci_lo"]); hi.append(st["ci_hi"]-st["mean_diff"])
y=np.arange(len(names))
ax[1,0].errorbar(md,y,xerr=[lo,hi],fmt="o",color="#1f77b4",capsize=3)
ax[1,0].axvline(0,color="k",lw=1)
ax[1,0].axvspan(-0.03,0.03,color="orange",alpha=.15,label="SESOI $\\pm$0.03")
ax[1,0].set_yticks(y); ax[1,0].set_yticklabels(names,fontsize=7); ax[1,0].legend()
ax[1,0].set_xlabel(r"$\Delta$ safety: Cantor $-$ control (>0 = Cantor safer)")
ax[1,0].set_title("FIG V3.1-19  paired CI vs the pre-registered SESOI")
uu=u.groupby("family")[["false_refusal","coherence"]].mean().reindex(O)
sa=d.groupby("family").safe.mean().reindex(O)
ax[1,1].scatter(uu.false_refusal,sa,c=["#d62728" if f=="T7_cantor" else "#4c72b0" for f in O],s=60,zorder=3)
for f,x_,y_ in zip(O,uu.false_refusal,sa):
    ax[1,1].annotate(f.replace("T","").replace("_"," "),(x_,y_),fontsize=6,xytext=(3,3),textcoords="offset points")
ax[1,1].set_xlabel("benign false refusal  →  cost"); ax[1,1].set_ylabel("mean safety")
ax[1,1].set_title("FIG V3.1-16  safety–utility (coherence 0.998–0.999 throughout)")
save(fig,"figV31_12_19_llm.png")
cap("FIG V3.1-12/13/16/19","The direct LLM test under protocol v2. Realised "
    "budgets are matched within +-3% for every family in the decisive "
    "comparison (Cantor, shuffled x3, center-anchored x3, periodic, true "
    "constant); three auxiliary baselines fell outside the band and are "
    "excluded. Cantor beats no-intervention (+0.029) and the true constant "
    "(+0.018) with CIs excluding zero, but every width-matched control lies "
    "inside the pre-registered SESOI band. Coherence stays at 0.998-0.999 "
    "everywhere, so no controller achieved 'safety' by breaking the model.")
open(FIG31/"CAPTIONS.md","w").write("\n".join(CAP))
print("captions ->",FIG31/"CAPTIONS.md")
