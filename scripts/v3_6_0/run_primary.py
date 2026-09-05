"""Preregistered mathematical, natural-state and adversarial routing audit."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd

from _common import ROOT,RESULTS,W_R,RHOS,assets,freeze_check,read,write,save_table
from cantor_guard_v360.routing import Router,margin,derivative,optimal_rho,project_to_x,numerical_switch_search


def theorem_and_depth():
    grid=np.unique(np.r_[np.linspace(.000001,.499999,500000),1/3])
    values=margin(grid)
    old=read(ROOT/'results/v3_3_2/tables/phase_calibration_qwen2.5-0.5b-instruct.json')['U_EST']['delta_abs_quantiles']
    rows=[]
    for n in range(1,9):
        r=optimal_rho(n)
        factor=1. if r is None else float(margin(r,n))
        rows.append(dict(depth=n,policies=2**n,optimal_rho=r,maximum_attained=r is not None,
            optimum_or_supremum=factor,residual_radius=W_R*factor,
            middle_third_margin=float(margin(1/3,n)),middle_third_radius=W_R*float(margin(1/3,n)),
            old_normalized_uncertainty_q50=old['q50'],old_normalized_uncertainty_q95=old['q95'],
            dimensional_illustration_q50_times_W_R=old['q50']*W_R,
            same_sensor_uncertainty_comparison_valid=False))
    write(RESULTS/'tables/depth_optima.json',{'rows':rows,
        'n1_exception':'rho->0+ supremum 1; no maximizer on (0,1/2)',
        'uncertainty_caveat':'historical U_EST uses another coordinate, NOT an uncertainty certificate for this frozen sensor',
        'depth3_design':'eight terminal policies, interpretable guards; a design tradeoff, not empirically optimal depth'})
    save_table('depth_optima.csv',pd.DataFrame(rows))
    ref=[.04289815228026324,.04735406915391874,.049418671426863243,.05084225455438606,
         .04981402079827816,.043927707934989554,.03189151596080242]
    per=[]
    for rho,reference in zip(RHOS,ref):
        rt=Router(rho,W_R=W_R)
        gaps=np.array([b.lo-a.hi for a,b in zip(rt.leaves[:-1],rt.leaves[1:])])*W_R
        per.append(dict(rho=rho,CRE=float(margin(rho)),CRE_abs=rt.certificate,
            implementation_minimum_gap=float(gaps.min()),implementation_error=abs(gaps.min()-rt.certificate),
            reference_value=reference,reference_error=abs(reference-rt.certificate)))
    exact=Fraction(1,3)**2*(1-2*Fraction(1,3))
    payload={'analytic_theorem':'M_n=rho^(n-1)*(1-2rho); derivative=rho^(n-2)*((n-1)-2n*rho), n>=2',
        'proof':'derivative positive before (n-1)/(2n), negative after; M_n vanishes at both ends for n>=2',
        'n1_exception':'M_1=1-2rho, derivative -2, no interior optimizer',
        'M3_exact':str(exact),'depth3_unique_optimizer':1/3,'dense_grid_n':len(grid),
        'dense_grid_argmax':float(grid[values.argmax()]),'dense_grid_unique_max':int(np.sum(values==values.max()))==1,
        'per_rho':per,'reference_match_tolerance':1e-12,
        'max_implementation_error':max(r['implementation_error'] for r in per),
        'PASS':exact==Fraction(1,27) and grid[values.argmax()]==1/3 and
               all(r['reference_error']<1e-12 and r['implementation_error']<1e-12 for r in per),
        'novelty':'general depth formula was already proved in V3.3.2; V3.6.0 contribution is exact routing solver, explicit scope and implementation validation'}
    write(RESULTS/'tables/theorem_validation.json',payload)
    return payload


def lipschitz(sensor):
    rng=np.random.default_rng(36000)
    h=rng.normal(size=(10000,len(sensor.w)))
    delta=rng.normal(size=h.shape)
    x=np.maximum(0,-sensor.distance(h));xp=np.maximum(0,-sensor.distance(h+delta))
    raw=np.abs(xp-x)-np.linalg.norm(delta,axis=1)
    tight_h=project_to_x(h[:100],sensor,2.)
    t=np.linspace(.0001,.5,100)
    tight_delta=-t[:,None]*sensor.w_hat
    err=np.abs(np.maximum(0,-sensor.distance(tight_h+tight_delta))-np.maximum(0,-sensor.distance(tight_h))-t)
    return {'random_trials':len(h),'raw_random_violations':int((raw>0).sum()),
            'maximum_random_excess':float(raw.max()),'tight_trials':100,
            'tight_max_abs_error':float(err.max()),'PASS':not np.any(raw>0) and err.max()<1e-9,
            'proof':'normalized affine sensor is exactly 1-Lipschitz, max(0,-d) is 1-Lipschitz; equality along normal inside risk halfspace'}


def records(h,router,sensor,source,ids):
    d=router.distances(sensor.distance(h))
    frame=pd.DataFrame(d)
    frame['source']=source;frame['source_id']=ids;frame['rho']=router.rho
    frame['certificate']=router.certificate
    depths=np.array([g.depth for g in router.guards])
    frame['nearest_guard_depth']=[int(depths[i]) if i>=0 else -1 for i in d['nearest_guard']]
    return frame


def summarize_solver(h,router,sensor,p):
    analytic=router.distances(sensor.distance(h))['D_terminal_switch']
    sol=numerical_switch_search(h,router,sensor,iterations=p['solver']['iterations'],
                              interior_fraction=p['solver']['witness_interior_fraction'])
    ae=np.abs(sol['estimate']-analytic)
    re=ae/np.maximum(np.abs(analytic),p['solver']['relative_denominator_floor'])
    return analytic,sol,ae,re


def main():
    p=freeze_check()
    if (RESULTS/'tables/final_verdict.json').exists(): raise SystemExit('completed experiment exists')
    theorem=theorem_and_depth()
    sensor,v=assets()
    lip=lipschitz(sensor)
    natural=np.concatenate([np.load(ROOT/x) for x in p['natural_states']])
    anchors=np.load(ROOT/p['anchor_states'])
    assert len(anchors)==200
    rng=np.random.default_rng(p['stress_seed'])
    random=rng.normal(size=anchors.shape);random/=np.linalg.norm(random,axis=1)[:,None]
    orth=random-(random@sensor.w_hat)[:,None]*sensor.w_hat
    orth/=np.linalg.norm(orth,axis=1)[:,None]
    mixed=sensor.w_hat+v;mixed/=np.linalg.norm(mixed)
    natural_frames=[];trials=[];summaries=[];stress=[];raw_viols=[]
    eps_c=W_R/27
    for rho in RHOS:
        router=Router(rho,W_R=W_R)
        f=records(natural,router,sensor,'unmodified_inherited_LLM',np.arange(len(natural)))
        natural_frames.append(f)
        natural_mask=f.kind.eq('leaf').to_numpy()
        if natural_mask.any():
            na,ns,ne,nre=summarize_solver(natural[natural_mask],router,sensor,p)
        else: na=ne=nre=np.array([]);ns={'witness_success':np.array([])}
        pos=np.array(p['leaf_positions'])
        x=np.concatenate([(c.lo+pos*(c.hi-c.lo))*W_R for c in router.leaves])
        base=np.repeat(anchors,len(x),axis=0)
        x=np.tile(x,len(anchors))
        projected=project_to_x(base,sensor,x)
        frame=records(projected,router,sensor,'normal_projected_LLM_component',np.repeat(np.arange(len(anchors)),80))
        if not frame.kind.eq('leaf').all(): raise RuntimeError('projected source not in intended terminal region')
        analytic,sol,ae,re=summarize_solver(projected,router,sensor,p)
        frame['numerical_distance']=sol['estimate'];frame['numeric_abs_error']=ae
        frame['numeric_rel_error']=re;frame['witness_success']=sol['witness_success']
        frame['witness_norm']=sol['witness_norm'];trials.append(frame)
        summaries.append({'rho':rho,'natural_states':len(natural),'natural_terminal_states':int(natural_mask.sum()),
            'natural_route_counts':f.kind.value_counts().to_dict(),
            'natural_minimum_exact_switch':float(np.min(na)) if len(na) else None,
            'natural_max_numeric_abs_error':float(ne.max()) if len(ne) else None,
            'natural_max_numeric_relative_error':float(nre.max()) if len(nre) else None,
            'projected_trials':len(frame),'projected_minimum_exact_switch':float(analytic.min()),
            'certificate':router.certificate,'minimum_slack':float((analytic-router.certificate).min()),
            'mean_abs_error':float(ae.mean()),'max_abs_error':float(ae.max()),'max_relative_error':float(re.max()),
            'failed_witnesses':int((~sol['witness_success']).sum())+int((~ns['witness_success']).sum())})
        # Stress both unmodified and projected sources. The sensor projection
        # of every unit direction is explicitly calculated from its 896-D vector.
        for name,hh,dd in [('unmodified',natural,f),('projected',projected,frame)]:
            source=dd.leaf.to_numpy(); terminal=dd.kind.eq('leaf').to_numpy()
            toward=np.sign(dd.target_x.to_numpy()-dd.x.to_numpy())
            toward=np.nan_to_num(toward,nan=1.)
            if name=='projected':
                isotropic=np.repeat(random@sensor.w_hat,80)
                perpendicular=np.repeat(orth@sensor.w_hat,80)
            else:
                rngn=np.random.default_rng(p['stress_seed']+1)
                rn=rngn.normal(size=hh.shape);rn/=np.linalg.norm(rn,axis=1)[:,None]
                ort=rn-(rn@sensor.w_hat)[:,None]*sensor.w_hat;ort/=np.linalg.norm(ort,axis=1)[:,None]
                isotropic=rn@sensor.w_hat;perpendicular=ort@sensor.w_hat
            slopes={'sensor_normal':-toward,'actuator':-toward*abs(sensor.w_hat@v),
                    'random_isotropic':isotropic,'sensor_orthogonal':perpendicular,
                    'mixed_sensor_actuator':-toward*abs(sensor.w_hat@mixed)}
            for family,slope in slopes.items():
                for scaling in p['stress_scalings']:
                    reference=router.certificate if scaling=='own_certificate' else eps_c
                    for factor in p['stress_factors']:
                        norm=factor*reference
                        k,leaf,_,_=router.classify_x(np.maximum(0,-(dd.d.to_numpy()+norm*slope)))
                        switched=terminal&(k=='leaf')&(source!=leaf)
                        violations=int(switched.sum()) if norm<=router.certificate else 0
                        stress.append(dict(rho=rho,population=name,family=family,scaling=scaling,
                            factor=factor,norm=norm,certificate=router.certificate,
                            all_states=len(dd),terminal_denominator=int(terminal.sum()),
                            switches=int(switched.sum()),rate=float(switched.sum()/terminal.sum()) if terminal.any() else None,
                            guard_entries=int((terminal&(k=='guard')).sum()),
                            nonterminal_exits=int((terminal&(k!='leaf')).sum()),raw_violations=violations))
                        if violations:
                            raw_viols.extend([dict(rho=rho,population=name,family=family,scaling=scaling,factor=factor,
                                source_row=int(i),norm=norm,certificate=router.certificate) for i in np.flatnonzero(switched)])
        print(f'rho={rho:.9g}: {len(frame)} solver trials, natural leaves={natural_mask.sum()}, max relative error={re.max():.3g}',flush=True)
    save_table('natural_state_distances.csv',pd.concat(natural_frames,ignore_index=True))
    save_table('adversarial_solver_trials.csv',pd.concat(trials,ignore_index=True))
    stress_frame=pd.DataFrame(stress);save_table('perturbation_stress.csv',stress_frame)
    write(RESULTS/'tables/exact_switch_distance.json',{'per_rho':summaries,
        'unmodified_total_states':len(natural),'projected_total_trials':sum(x['projected_trials'] for x in summaries),
        'natural_is_not_worst_case_envelope':True,
        'solver_definition':'distance infimum to another terminal leaf; guard endpoints excluded from target terminal regions',
        'projection_disclosure':'112000 trials use counterfactual normal projections of 200 stored LLM states, NOT 112000 independent natural LLM states'})
    write(RESULTS/'tables/perturbation_stress.json',{'rows':stress,'raw_violations':raw_viols,
        'raw_below_or_equal_certificate_violations':len(raw_viols),
        'computational_method':'exact affine residual-sensor projection with 896-D direction dot products and actual Router classification; no model generation',
        'family_sign_rule':'normal, actuator line and mixed line oriented towards nearest alternative; isotropic and orthogonal vectors fixed across rho',
        'primary_denominator':'terminal-source states; ineligible safe/guard/outside counts retained separately'})
    tolerance=p['solver']
    agreement=all(s['max_abs_error']<=tolerance['absolute_tolerance'] and
        s['max_relative_error']<=tolerance['relative_tolerance'] and s['failed_witnesses']==0 and
        (s['natural_max_numeric_abs_error'] is None or s['natural_max_numeric_abs_error']<=tolerance['absolute_tolerance']) and
        (s['natural_max_numeric_relative_error'] is None or s['natural_max_numeric_relative_error']<=tolerance['relative_tolerance']) for s in summaries)
    write(RESULTS/'tables/certificate_validation.json',{'lipschitz':lip,
        'direct_switch_violations_below_or_equal':len(raw_viols),
        'maximum_violation':max((v['certificate']-v['norm'] for v in raw_viols),default=0.),
        'minimum_raw_projected_slack':min(s['minimum_slack'] for s in summaries),
        'analytic_numeric_agreement':agreement,'tolerance':tolerance,
        'maximum_absolute_error':max(s['max_abs_error'] for s in summaries),
        'maximum_relative_error':max(s['max_relative_error'] for s in summaries),
        'total_solver_trials':sum(s['projected_trials']+s['natural_terminal_states'] for s in summaries),
        'certificate_scope':'terminal-to-terminal only; not guard entry, safe/outside transitions, continuous-path no-guard crossing, or semantics',
        'boundary_convention':'guard owns internal endpoints; infimum not necessarily attained',
        'PASS':theorem['PASS'] and lip['PASS'] and agreement and not raw_viols})
    ablations=[]
    for ablation in ['NO_GUARD','NON_RECURSIVE_GUARD',None]:
        r=Router(W_R=W_R,ablation=ablation)
        ablations.append(dict(name=ablation or 'MIDDLE_THIRD_CANTOR',policies=8,
            retained_measure=sum(c.hi-c.lo for c in r.leaves),guards=len(r.guards),
            CRE=r.certificate/W_R,CRE_abs=r.certificate,recursive=ablation is None))
    for rho in RHOS:
        if rho!=1/3:
            r=Router(rho,W_R=W_R);ablations.append(dict(name='RECURSIVE_NONOPTIMAL',rho=rho,policies=8,
                retained_measure=(2*rho)**3,guards=7,CRE=r.certificate/W_R,CRE_abs=r.certificate,recursive=True))
    write(RESULTS/'tables/ablation_comparison.json',{'rows':ablations,
        'nonrecursive_optimality_under_aggregate_constraints':'7 gaps sum to 19/27, so min gap <=19/189; equal gaps attain this',
        'nonrecursive_beats_cantor':ablations[1]['CRE_abs']>ablations[2]['CRE_abs'],
        'ratio_nonrecursive_to_cantor':ablations[1]['CRE_abs']/ablations[2]['CRE_abs'],
        'no_guard_zero_infimum':'distinct uniform bins touch; arbitrarily small cross-bin perturbations exist',
        'claim_scope':'Cantor unique optimum only within fixed-depth same-rho symmetric recursive family'})
    print('primary audit complete')

if __name__=='__main__': main()
