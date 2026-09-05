"""Eight standalone scientific figures, all generated from recorded results."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/tmp/cantor-v360-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/cantor-v360-cache')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import FIGURES,RESULTS,W_R,RHOS,read,write,freeze_check
from cantor_guard_v360.routing import Router,margin


def save(fig,name):
    fig.tight_layout();fig.savefig(FIGURES/name,dpi=180,bbox_inches='tight');plt.close(fig)


def main():
    freeze_check();FIGURES.mkdir(parents=True,exist_ok=True)
    x=np.linspace(.001,.499,600)
    fig,ax=plt.subplots(figsize=(8,4.4));ax.plot(x,margin(x),color='#2563eb',lw=2)
    ax.axvline(1/3,color='#dc2626',ls='--',label='unique max: 1/3');ax.legend()
    ax.set(xlabel='rho',ylabel='CRE = M3(rho)',title='Depth-3 symmetric self-similar family')
    save(fig,'F360-01-rho-vs-margin.png')
    dep=read(RESULTS/'tables/depth_optima.json')['rows']
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    axes[0].plot([r['depth'] for r in dep[1:]],[r['optimal_rho'] for r in dep[1:]],'o-')
    axes[0].text(.05,.85,'n=1: supremum at 0+\n(no interior optimizer)',transform=axes[0].transAxes)
    axes[0].set(xlabel='depth',ylabel='optimal rho',title='Optimal ratio depends on depth')
    axes[1].plot([r['depth'] for r in dep[1:]],[r['residual_radius'] for r in dep[1:]],'o-')
    axes[1].set(xlabel='depth',ylabel='optimal radius in residual L2',yscale='log',title='Policy resolution costs margin')
    save(fig,'F360-02-depth-optimal-rho.png')
    fig,ax=plt.subplots(figsize=(11,4.4))
    for depth in [1,2,3]:
        r=Router(depth=depth)
        for c in r.leaves:
            ax.broken_barh([(c.lo,c.hi-c.lo)],(4-depth-.25,.5),facecolors='#2563eb')
            ax.text((c.lo+c.hi)/2,4-depth,c.address,ha='center',va='center',fontsize=8,color='white')
        for g in r.guards:
            ax.broken_barh([(g.lo,g.hi-g.lo)],(4-depth-.25,.5),facecolors='#e5a334',alpha=.7)
    ax.set(xlim=(0,1),ylim=(.4,3.6),yticks=[1,2,3],yticklabels=['depth 3','depth 2','depth 1'],
           xlabel='normalized risk coordinate',title='Nested policy refinement: blue leaves, amber guards')
    save(fig,'F360-03-cantor-hierarchy.png')
    radii=[Router(r,W_R=W_R).certificate for r in RHOS]
    labels=['1/3' if r==1/3 else f'{r:.2f}' for r in RHOS]
    fig,ax=plt.subplots(figsize=(8,4.4));ax.bar(labels,radii,color=['#dc2626' if r==1/3 else '#2563eb' for r in RHOS])
    ax.set(xlabel='rho',ylabel='certified residual-L2 radius',title=f'Absolute CRE at frozen W_R={W_R:.4f}')
    save(fig,'F360-04-certified-radius.png')
    ex=read(RESULTS/'tables/exact_switch_distance.json')['per_rho']
    fig,ax=plt.subplots(figsize=(8,4.8))
    ax.plot(labels,radii,'o-',label='global theoretical infimum')
    ax.plot(labels,[r['projected_minimum_exact_switch'] for r in ex],'x--',label='projected probe minimum')
    ax.plot(labels,[r['natural_minimum_exact_switch'] for r in ex],'s-',label='unmodified-state observed minimum')
    ax.set(xlabel='rho',ylabel='residual L2',title='Global envelope and finite-sample minima are different');ax.legend()
    save(fig,'F360-05-exact-vs-certificate.png')
    st=pd.read_csv(RESULTS/'raw/perturbation_stress.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,pop in zip(axes,['unmodified','projected']):
        f=st[(st.population==pop)&(st.family=='sensor_normal')&(st.scaling=='absolute_middle_third_certificate')]
        for r,label in zip(RHOS,labels):
            g=f[np.isclose(f.rho,r)].sort_values('norm');ax.plot(g.norm,g.rate,marker='.',label=label)
        ax.set(xlabel='common absolute residual-L2 perturbation',ylabel='direct terminal switch rate',title=pop+' terminal sources')
        ax.legend(title='rho',fontsize=8)
    save(fig,'F360-06-switch-rate.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    g=st[(st.population=='projected')&(st.family=='sensor_normal')&(st.scaling=='own_certificate')]
    for r,label in zip(RHOS,labels):
        z=g[np.isclose(g.rho,r)].sort_values('factor');axes[0].plot(z.factor,z.rate,label=label)
    axes[0].axvline(1,color='black',ls='--');axes[0].legend(fontsize=8)
    axes[0].set(xlabel='norm / own certificate',ylabel='direct switch rate',title='Recursive comparison at own scale')
    ab=read(RESULTS/'tables/ablation_comparison.json')['rows'][:3]
    axes[1].bar(['No guard','Nonrecursive\nequal gaps','Middle third'],[r['CRE_abs'] for r in ab],color=['#64748b','#16a34a','#2563eb'])
    axes[1].set(ylabel='global certified radius',title='Removing recursion changes the optimum')
    save(fig,'F360-07-rho-comparison.png')
    fig,ax=plt.subplots(figsize=(11,4.4));ax.axis('off')
    boxes=[(.02,'Sensor geometry\nFrozen signed distance\nx=max(0,-d)\n1-Lipschitz'),
           (.37,'Policy routing\nLeaves + explicit guards\nExact terminal-switch infimum\nFamily-specific certificate'),
           (.72,'Model behavior\nToken logits and generation\nRefusal proxy and coherence\nSeparate empirical question')]
    for xx,label in boxes:
        ax.text(xx+.13,.58,label,transform=ax.transAxes,ha='center',va='center',fontsize=11,
                bbox=dict(boxstyle='round,pad=.8',fc='#eff6ff' if xx<.7 else '#f1f5f9',ec='#2563eb' if xx<.7 else '#64748b'))
    ax.annotate('',xy=(.35,.58),xytext=(.29,.58),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',lw=2))
    ax.annotate('',xy=(.70,.58),xytext=(.65,.58),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',lw=2,linestyle='dashed'))
    ax.text(.32,.2,'Certificate links sensor geometry to discrete routing',ha='center',transform=ax.transAxes,color='#2563eb')
    ax.text(.82,.2,'No semantic or behavioral\nsuperiority guarantee',ha='center',transform=ax.transAxes,color='#64748b')
    ax.set_title('Mathematical policy stability and model behavior are distinct',pad=10)
    save(fig,'F360-08-scope-diagram.png')
    write(FIGURES/'CAPTIONS.json',{'scope':'family-specific direct terminal-policy-switch certificate',
        'F360-05':'Counterfactual projections are distinguished from unmodified LLM state samples.',
        'F360-06':'Rates condition on each rho terminal-source mask; masks may differ. Global worst-case optimum does not order these rates.',
        'F360-07':'Nonrecursive guard ablation preserves 8 policies and total leaf length 8/27; it has larger separation.',
        'F360-08':'Certificate applies to first two layers; behavioral association requires independent evidence.'})
    print('8 figures generated')

if __name__=='__main__':main()
