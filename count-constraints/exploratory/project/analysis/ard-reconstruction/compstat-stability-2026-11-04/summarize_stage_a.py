"""Read-only scientific re-scoring and paired summaries of the fixed Stage A outputs."""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
INPUT=ROOT/'analysis/ard-reconstruction/jspi-rerun-2026-10-14/fresh-results'
BENCH=ROOT/'analysis/ard-reconstruction/compstat-conditional-2026-11-04/results'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def auc(a,s):
    pos,neg=s[a==1],s[a==0]
    if not len(pos) or not len(neg):return None
    diff=pos[:,None]-neg[None,:]
    return float((np.sum(diff>0)+.5*np.sum(diff==0))/diff.size)


def metrics(p,arrays,benchmark=None):
    y,b,a,truth=(arrays[x] for x in ('Y','B','hidden_A','true_P'))
    i,j=np.triu_indices(40,1);labels=a[i,j].astype(int);groups=b.argmax(axis=1)
    u=(y[i,groups[j]]*y[j,groups[i]])>0;s=p[i,j]
    out={'auc_raw':auc(labels,s),'auc_U':auc(labels[u],s[u]),'auc_zero_adjusted':auc(labels,s*u),
         'probability_rmse':float(np.sqrt(np.mean((s-truth[i,j])**2))),
         'ard_rmse':float(np.sqrt(np.mean((p@b-y)**2))),'brier_A':float(np.mean((s-labels)**2))}
    if benchmark is not None:
        v=benchmark['unresolved'][i,j];q=benchmark['q_oracle'][i,j]
        out['auc_V']=auc(labels[v],s[v])
        out['brier_A_V']=float(np.mean((s[v]-labels[v])**2))
        out['conditional_excess_brier_V']=float(np.mean((s[v]-q[v])**2))
    return out


def stats(values):
    vals=np.array([v for v in values if v is not None],dtype=float)
    if not len(vals):return {'n':0,'mean':None,'mcse':None}
    return {'n':len(vals),'mean':float(vals.mean()),'sd':float(vals.std(ddof=1)),
            'mcse':float(vals.std(ddof=1)/math.sqrt(len(vals))),
            'min':float(vals.min()),'max':float(vals.max()),
            'positive':int(np.sum(vals>0)),'negative':int(np.sum(vals<0)),'zero':int(np.sum(vals==0))}


def mdiff(a,b):
    i,j=np.triu_indices(len(a),1);d=(a-b)[i,j]
    return {'rms':float(np.sqrt(np.mean(d*d))),'max_abs':float(np.max(np.abs(d)))}


def main():
    contract=json.loads((HERE/'RUN_CONTRACT.json').read_text())
    circle=json.loads((HERE/'circular/results.json').read_text())
    nuclear=json.loads((HERE/'nuclear/SUMMARY.json').read_text())
    assert len(circle['runs'])==80 and nuclear['saved_fits']==240 and not nuclear['missing'] and not nuclear['failed']
    per_cell=[];provenance={}
    for k in contract['k']:
        for seed in contract['seeds']:
            fn=f'run_k{k}_seed{seed}.npz';ip=INPUT/fn;assert sha(ip)==contract['input_sha256'][fn]
            with np.load(ip) as z:arrays={x:z[x] for x in z.files}
            benchmark=None
            if k==8:
                bp=BENCH/fn
                with np.load(bp) as z:benchmark={x:z[x] for x in z.files}
                provenance[str(bp.relative_to(ROOT))]=sha(bp)
            cdir=HERE/'circular'/fn.removesuffix('.npz')
            with np.load(cdir/'summary.npz') as z:cm={x:z[x] for x in z.files}
            complete=json.loads((cdir/'complete.json').read_text())
            estimates={'Density':arrays['density_P'],'Circular_old':arrays['bayes_P'],'Nuclear_old':arrays['nuclear_P'],
                       'Circular_long':cm['pooled_P'],'Circular_first_half':cm['first_half_P'],
                       'Circular_last_half':cm['last_half_P'],'Circular_group12':cm['group12_P'],'Circular_group34':cm['group34_P']}
            for c in range(4):estimates[f'Circular_chain{c+1}']=cm['chain_mean_P'][c]
            nu={}
            for penalty in contract['nuclear']['lambda']:
                name=f'run_k{k}_seed{seed}_lambda{penalty:g}'
                # The nuclear implementation owns its exact filename convention.
                matches=list((HERE/'nuclear/fits').glob(f'*k{k}_seed{seed}_*'))
                found=[]
                for p in matches:
                    if p.suffix=='.json':
                        meta=json.loads(p.read_text())
                        if meta['penalty']==penalty:found.append((p,meta))
                assert len(found)==1,(k,seed,penalty)
                mp,nmeta=found[0];npz=mp.with_suffix('.npz');assert sha(npz)==nmeta['npz_sha256']
                with np.load(npz) as z:
                    estimates[f'Nuclear_lambda{penalty:g}']=z['final_P']
                    estimates[f'Nuclear_early_lambda{penalty:g}']=z['early_P']
                nu[str(penalty)]={'status':nmeta['status'],'early':nmeta['early'],'final':nmeta['final'],
                                  'matrix_change':nmeta['matrix_change']}
            if benchmark is not None:
                estimates['Q_uniform']=benchmark['q_uniform'];estimates['Q_oracle']=benchmark['q_oracle']
            scored={name:metrics(p,arrays,benchmark) for name,p in estimates.items()}
            # Conditional graph marginals target A, so do not report their error against generating P*.
            for name in ('Q_uniform','Q_oracle'):
                if name in scored:scored[name].pop('probability_rmse')
            i,j=np.triu_indices(40,1);group=arrays['B'].argmax(axis=1)
            product=arrays['Y'][i,group[j]]*arrays['Y'][j,group[i]];a=arrays['hidden_A'][i,j].astype(int);u=product>0
            scored['Product']={'auc_raw':auc(a,product),'auc_U':auc(a[u],product[u]),'auc_zero_adjusted':auc(a,product)}
            if benchmark is not None:
                v=benchmark['unresolved'][i,j];scored['Product']['auc_V']=auc(a[v],product[v])
            distances={'old_to_long':mdiff(arrays['bayes_P'],cm['pooled_P']),
                       'first_to_last_half':mdiff(cm['first_half_P'],cm['last_half_P']),
                       'group12_to_group34':mdiff(cm['group12_P'],cm['group34_P']),
                       'chain_pairs':[{'left':a+1,'right':b+1,**mdiff(cm['chain_mean_P'][a],cm['chain_mean_P'][b])}
                                      for a in range(4) for b in range(a+1,4)]}
            per_cell.append({'k':k,'seed':seed,'metrics':scored,'matrix_distances':distances,
                             'matrix_diagnostics':complete['matrix_diagnostics'],
                             'scalar_diagnostics':complete['scalar_diagnostics'],'nuclear':nu})
    summaries=[];contrasts=[];alerts=[]
    for k in contract['k']:
        group=[x for x in per_cell if x['k']==k];models=list(group[0]['metrics'])
        for name in models:
            for metric in group[0]['metrics'][name]:
                summaries.append({'k':k,'model':name,'metric':metric,**stats([x['metrics'][name][metric] for x in group])})
        pairs=[('Circular_long','Density'),('Circular_old','Density'),('Circular_long','Circular_old'),
               ('Circular_last_half','Circular_first_half'),('Circular_group34','Circular_group12')]
        for lam in (.25,1,4):
            model=f'Nuclear_lambda{lam:g}';pairs.extend([(model,'Density'),(model,'Product'),(model,f'Nuclear_early_lambda{lam:g}')])
            if k==8:pairs.append((model,'Q_uniform'))
        if k==8:pairs.extend([('Circular_long','Q_uniform'),('Q_oracle','Q_uniform')])
        for left,right in pairs:
            common=set(group[0]['metrics'][left])&set(group[0]['metrics'][right])
            for metric in sorted(common):
                vals=[x['metrics'][left][metric]-x['metrics'][right][metric] for x in group]
                item={'k':k,'left':left,'right':right,'metric':metric,**stats(vals)};contrasts.append(item)
        # Predeclared review flag uses graph-paired contrasts, not unpaired marginal MCSEs.
        for left,right in [('Circular_long','Circular_old'),('Circular_last_half','Circular_first_half'),
                           ('Circular_group34','Circular_group12')]+[(f'Nuclear_lambda{x:g}',f'Nuclear_early_lambda{x:g}') for x in (.25,1,4)]:
            for metric in group[0]['metrics'][left]:
                reference=('Q_uniform' if metric in ('auc_V','brier_A_V','conditional_excess_brier_V') else
                           'Product' if metric.startswith('auc') else 'Density')
                before=stats([x['metrics'][right][metric]-x['metrics'][reference][metric] for x in group])
                after=stats([x['metrics'][left][metric]-x['metrics'][reference][metric] for x in group])
                delta=after['mean']-before['mean'];threshold=.5*max(before['mcse'],after['mcse'])
                sign_reversal=abs(before['mean'])>2*before['mcse'] and before['mean']*after['mean']<0
                if (abs(delta)>threshold and abs(delta)>1e-14) or sign_reversal:
                    alerts.append({'k':k,'comparison':left+' minus '+right,'reference':reference,'metric':metric,
                        'mean_change':delta,'half_max_graph_mcse':threshold,'before_contrast':before,'after_contrast':after,
                        'previous_resolved_direction_reversed':sign_reversal})
    report={'status':'complete','contract_sha256':sha(HERE/'RUN_CONTRACT.json'),'seeds':contract['seeds'],
            'runs':per_cell,'summary':summaries,'paired_contrasts':contrasts,'computation_review_flags':alerts,
            'matrix_screen_failed_cells':[{'k':x['k'],'seed':x['seed'],**x['matrix_diagnostics']} for x in per_cell if not x['matrix_diagnostics']['screen_pass']],
            'scalar_screen_failed_cells':[{'k':x['k'],'seed':x['seed'],**x['scalar_diagnostics']} for x in per_cell if not x['scalar_diagnostics']['screen_pass']],
            'nuclear_final_caps':nuclear['final_caps'],'conditional_benchmark_input_hashes':provenance,
            'predeclared_threshold_gate_cleared':not nuclear['final_caps'] and all(x['matrix_diagnostics']['screen_pass'] and x['scalar_diagnostics']['screen_pass'] for x in per_cell),
            'review_flag_interpretation':'Flags require judgment; old-to-long changes do not alone imply the new long run is unstable. New half/group contrasts and diagnostic warnings are separately reported.',
            'limitations':['Original40pairedgraphs, no new confirmation sample','Diagnostic threshold success is not posterior calibration proof',
                           'All predefined lambda settings reported; hidden truth never selects lambda/chains']}
    with (HERE/'SUMMARY.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    for filename,rows in [('summary.csv',summaries),('paired-contrasts.csv',contrasts)]:
        with (HERE/filename).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'cells':len(per_cell),'matrix_failures':len(report['matrix_screen_failed_cells']),
        'scalar_failures':len(report['scalar_screen_failed_cells']),'nuclear_caps':len(nuclear['final_caps']),
        'review_flags':len(alerts),'predeclared_threshold_gate_cleared':report['predeclared_threshold_gate_cleared']}))


if __name__=='__main__':main()
