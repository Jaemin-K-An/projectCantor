"""Small, explicitly exploratory boundary-generation diagnostic.

Each perturbed residual is generated with routed and locked-original policy.
Their difference isolates the policy action from the shared residual shift.
Boundary projection is counterfactual and never presented as a natural prompt.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from _common import ROOT,RESULTS,W_R,RHOS,assets,freeze_check,read,write,save_table,sha
from cantor_guard.models import load_model,decoder_layers,chat_prompt
from cantor_guard_v335a.p0_residual import last_valid_index
from cantor_guard_v32.metrics32 import coherence32
from cantor_guard_v360.routing import Router,project_to_x
from scripts.v3_5_1._stats import shared_index,max_t_intervals


def edit_distance(a,b):
    previous=list(range(len(b)+1))
    for i,av in enumerate(a):
        row=[i+1]
        for j,bv in enumerate(b): row.append(min(row[-1]+1,previous[j+1]+1,previous[j]+(av!=bv)))
        previous=row
    return previous[-1]/max(len(a),len(b),1)


def kl(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    from scipy.special import logsumexp
    loga=a-logsumexp(a,axis=-1,keepdims=True)
    logb=b-logsumexp(b,axis=-1,keepdims=True)
    return np.maximum(0.,np.sum(np.exp(loga)*(loga-logb),axis=-1))


def generate_cell(bundle,prompts,router,sensor,v,eta,norm,locked,settings):
    texts=[];tokens=[];logits=[];metadata=[]
    c=router.leaves[settings['source_leaf_index']]
    base_x=c.hi*W_R-.01*router.certificate
    original_action=(c.index+1)/8
    for start in range(0,len(prompts),settings['batch_size']):
        chunk=prompts[start:start+settings['batch_size']]
        enc=bundle.tokenizer([chat_prompt(bundle,x) for x in chunk],return_tensors='pt',padding=True).to(bundle.device)
        last=last_valid_index(enc['attention_mask']); trace={}; first=[True]
        def hook(_module,_inputs,output):
            if not first[0]: return output
            first[0]=False
            h=output[0] if isinstance(output,tuple) else output
            row=h[torch.arange(len(chunk),device=h.device),last].float().detach().cpu().numpy().astype(float)
            base=project_to_x(row,sensor,base_x)
            attacked=base-norm*sensor.w_hat
            # Route the actual float32 state delivered to the live model.
            attacked=attacked.astype(np.float32).astype(float)
            actual_d=sensor.distance(attacked)
            kind,leaf,action,_=router.classify_x(np.maximum(0,-actual_d))
            if locked: action=np.full(len(row),original_action)
            q=np.minimum(eta*action,settings['q_cap'])
            corrected=attacked+(q*np.linalg.norm(attacked,axis=1))[:,None]*v
            new=h.clone();new[torch.arange(len(chunk),device=h.device),last]=torch.as_tensor(corrected,dtype=h.dtype,device=h.device)
            trace.update(kind=kind,leaf=leaf,action=action,q=q,actual_d=actual_d,
                projection_norm=np.linalg.norm(base-row,axis=1),
                realized_perturbation_norm=np.linalg.norm(attacked-base,axis=1),
                coordinate_roundoff=np.abs(-actual_d-(base_x+norm)))
            return (new,)+output[1:] if isinstance(output,tuple) else new
        handle=decoder_layers(bundle)[14].register_forward_hook(hook)
        try:
            with torch.no_grad():
                out=bundle.model.generate(**enc,max_new_tokens=settings['max_new_tokens'],do_sample=False,
                    pad_token_id=bundle.tokenizer.pad_token_id,return_dict_in_generate=True,output_scores=True)
        finally: handle.remove()
        ids=out.sequences[:,enc['input_ids'].shape[1]:].cpu().tolist()
        for ids_row in ids:
            if bundle.tokenizer.eos_token_id in ids_row: ids_row=ids_row[:ids_row.index(bundle.tokenizer.eos_token_id)]
            tokens.append(ids_row)
        texts.extend(bundle.tokenizer.batch_decode(ids,skip_special_tokens=True))
        logits.append(out.scores[0].float().cpu().numpy())
        for i in range(len(chunk)):
            metadata.append({k: (value[i].item() if isinstance(value[i],np.generic) else value[i]) for k,value in trace.items()})
        del out
    return texts,tokens,np.concatenate(logits),metadata


def score_pairs(rows,settings):
    cache_path=RESULTS/'private/refusal_cache.json'
    cache=read(cache_path) if cache_path.exists() else {}
    def key(p,c):return hashlib.sha256((p+'\0'+c).encode()).hexdigest()
    old=ROOT/'results/v3_5_1/private/refusal_score_cache.csv'
    if old.exists():
        df=pd.read_csv(old)
        cache.update(dict(zip(df.pair_sha256,df.action_class.astype(int))))
    pending={key(r['prompt'],r['completion']):(r['prompt'],r['completion']) for r in rows if key(r['prompt'],r['completion']) not in cache}
    if pending:
        tok=AutoTokenizer.from_pretrained(settings['evaluator'],revision=settings['evaluator_revision'],local_files_only=True)
        model=AutoModelForSequenceClassification.from_pretrained(settings['evaluator'],revision=settings['evaluator_revision'],local_files_only=True).eval()
        items=list(pending.items())
        for start in range(0,len(items),16):
            chunk=items[start:start+16]
            enc=tok([v[0] for _,v in chunk],[v[1] for _,v in chunk],return_tensors='pt',padding=True,truncation=True,max_length=512)
            with torch.no_grad(): pred=model(**enc).logits.argmax(-1).tolist()
            for (k,_),value in zip(chunk,pred):cache[k]=value
            if start%128==0 or start+16>=len(items):
                write(cache_path,cache);print(f'external refusal scoring {min(start+16,len(items))}/{len(items)}',flush=True)
        del model
    for row in rows:
        row['action_class']=cache[key(row['prompt'],row['completion'])]
        row['refusal']=int(row['action_class'] in settings['refusal_classes'])
    return len(pending)


def main():
    p=freeze_check();s=p['secondary']
    dest=RESULTS/'tables/behavioral_secondary.json'
    if dest.exists():raise SystemExit('secondary completed; do not rerun')
    private=RESULTS/'private';private.mkdir(parents=True,exist_ok=True)
    sensor,v=assets()
    prompts_df=pd.read_csv(ROOT/'results/v3_5_1/cache/D_risk_cal_v351.csv').iloc[:s['n_prompts']]
    prompts=prompts_df.prompt.astype(str).tolist();pids=prompts_df.pid.astype(str).tolist()
    if not torch.backends.mps.is_available():raise SystemExit('frozen secondary backend requires MPS; rerun with device access')
    bundle=load_model(p['model'],device='mps')
    inherited=read(ROOT/'configs/v3_5_1/controller.json')['eta_per_arm']
    cells={};all_rows=[]
    for ri,rho in enumerate(RHOS):
        router=Router(rho,W_R=W_R)
        name='1/3' if rho==1/3 else f'{rho:.2f}'
        specs=[('baseline',0.,False)]
        for factor in s['regimes']:
            specs.extend([(f'own_{factor:g}',factor*router.certificate,False),(f'own_{factor:g}',factor*router.certificate,True)])
        specs.extend([('absolute_1.1',1.1*W_R/27,False),('absolute_1.1',1.1*W_R/27,True)])
        for regime,norm,locked in specs:
            mode='locked' if locked else 'routed'
            tag=f'rho{ri}_{regime}_{mode}'
            file=private/f'{tag}.json';lp=private/f'{tag}_logits.npz'
            if file.exists() and lp.exists():
                got=read(file);lg=np.load(lp)['logits']
            else:
                texts,tokens,lg,meta=generate_cell(bundle,prompts,router,sensor,v,inherited['CANTOR_'+name],norm,locked,s)
                got=[dict(pid=pid,prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),completion=text,
                    tokens=toks,rho=rho,regime=regime,mode=mode,norm=norm,certificate=router.certificate,
                    projected_source_leaf=s['source_leaf_index'],**m) for pid,prompt,text,toks,m in zip(pids,prompts,texts,tokens,meta)]
                np.savez_compressed(lp,logits=lg);write(file,got)
            cells[(ri,regime,mode)]=(got,lg);all_rows.extend(got)
            print(f'secondary {tag}: {len(got)} rows complete',flush=True)
    provenance=bundle.provenance();del bundle;torch.mps.empty_cache()
    new_scores=score_pairs(all_rows,s)
    # rows in cells and all_rows reference the same dictionaries.
    metrics=[]
    for (ri,regime,mode),(rows,lg) in cells.items():
        if regime=='baseline':continue
        base,blg=cells[(ri,'baseline','routed')]
        locked,llg=cells[(ri,regime,'locked')]
        baseline_kl=kl(blg,lg);isolated_kl=kl(llg,lg)
        for i,row in enumerate(rows):
            coh=coherence32(row['completion'])
            metrics.append(dict(pid=row['pid'],prompt_sha256=row['prompt_sha256'],rho=row['rho'],regime=regime,mode=mode,
                norm=row['norm'],certificate=row['certificate'],kind=row['kind'],leaf=row['leaf'],
                direct_terminal_switch=row['kind']=='leaf' and row['leaf']!=s['source_leaf_index'],
                action=row['action'],q=row['q'],token1_KL_vs_baseline=float(baseline_kl[i]),
                token1_top1_change=int(np.argmax(blg[i])!=np.argmax(lg[i])),
                refusal=row['refusal'],refusal_label_change=int(row['refusal']!=base[i]['refusal']),
                normalized_token_edit_distance=edit_distance(base[i]['tokens'],row['tokens']),
                coherence=coh,degeneration=int(coh<.5),
                policy_isolated_KL=float(isolated_kl[i]),
                policy_isolated_top1_change=int(np.argmax(llg[i])!=np.argmax(lg[i])),
                policy_isolated_edit_distance=edit_distance(locked[i]['tokens'],row['tokens']),
                projection_norm=row['projection_norm'],realized_perturbation_norm=row['realized_perturbation_norm'],
                coordinate_roundoff=row['coordinate_roundoff']))
    df=pd.DataFrame(metrics);save_table('behavioral_secondary.csv',df)
    write(private/'all_scored_completions.json',all_rows)
    routed=df[df['mode']=='routed']
    mean_metrics=['direct_terminal_switch','token1_KL_vs_baseline','token1_top1_change','refusal_label_change',
        'normalized_token_edit_distance','coherence','degeneration','policy_isolated_KL','policy_isolated_top1_change','policy_isolated_edit_distance']
    means=routed.groupby(['rho','regime'])[mean_metrics].mean().reset_index()
    absolute=routed[routed.regime=='absolute_1.1'].pivot(index='pid',columns='rho',values='policy_isolated_KL').reindex(index=pids,columns=RHOS)
    radii=np.array([Router(r,W_R=W_R).certificate for r in RHOS]);xc=radii-radii.mean()
    slopes=absolute.to_numpy()@xc/np.sum(xc**2)
    idx=shared_index(len(pids),n_boot=s['bootstrap_replicates'],seed=s['bootstrap_seed'])
    cache=RESULTS/'cache';cache.mkdir(exist_ok=True)
    np.save(cache/'secondary_prompt_bootstrap.npy',idx)
    boot=slopes[idx].mean(axis=1);ci=np.quantile(boot,[.025,.975])
    per={str(r):absolute[r].to_numpy() for r in RHOS}
    contrasts=[(str(1/3),str(r)) for r in RHOS if r!=1/3]
    simultaneous=max_t_intervals(per,contrasts,idx)
    write(dest,{'scope':s['scope'],'model':provenance,'n_prompts':len(pids),'total_generated_rows':len(all_rows),
        'prompt_order':pids,'source_selection':s['prompt_selection'],'source_projection':s['source_offset_below_right_edge'],
        'new_unique_refusal_pairs':new_scores,'evaluator':s['evaluator'],'evaluator_revision':s['evaluator_revision'],
        'means':means.to_dict('records'),
        'alignment':{'endpoint':'policy-isolated routed-vs-locked token1 KL at fixed absolute 1.1*epsilon_C',
            'mean_slope':float(slopes.mean()),'prompt_bootstrap_95_CI':ci.tolist(),'negative_alignment_supported':bool(ci[1]<0),
            'interpretation':'association across partitions on counterfactual boundary probes, not natural behavioral superiority'},
        'max_T':simultaneous,'max_q':float(df.q.max()),'max_float32_coordinate_roundoff':float(df.coordinate_roundoff.max()),
        'completions_local_only':True,'raw_metrics_sha256':sha(RESULTS/'raw/behavioral_secondary.csv'),
        'primary_result_not_changed':True})
    print('behavioral secondary complete; alignment CI',ci)

if __name__=='__main__':main()
