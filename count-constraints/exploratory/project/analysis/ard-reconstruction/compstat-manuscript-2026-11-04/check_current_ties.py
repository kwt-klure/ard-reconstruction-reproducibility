"""Fixed numerical tie sensitivity for the updated K8 manuscript comparison; no fits."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[3]


def auc(labels,score,tol):
    d=score[labels==1,None]-score[None,labels==0]
    return float((np.sum(d>tol)+.5*np.sum(np.abs(d)<=tol))/d.size)


def compute():
    lane=ROOT/'analysis/ard-reconstruction'
    contract=json.loads((lane/'compstat-stability-2026-11-04/RUN_CONTRACT.json').read_text())
    seeds=contract['seeds'];rows=[];source_hashes={};i,j=np.triu_indices(40,1)
    for seed in seeds:
        paths={
          'input':lane/f'jspi-rerun-2026-10-14/fresh-results/run_k8_seed{seed}.npz',
          'circular':lane/f'compstat-stability-2026-11-04/circular/run_k8_seed{seed}/summary.npz',
          'nuclear':lane/f'compstat-stability-2026-11-04/nuclear/fits/run_k8_seed{seed}_lambda1.npz',
          'conditional':lane/f'compstat-conditional-2026-11-04/results/run_k8_seed{seed}.npz'}
        arrays={}
        for name,p in paths.items():
            source_hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
            with np.load(p) as z:arrays[name]={k:z[k] for k in z.files}
        original=arrays['input'];g=original['B'].argmax(axis=1);a=original['hidden_A'][i,j].astype(int)
        product=original['Y'][i,g[j]]*original['Y'][j,g[i]];u=product>0;v=arrays['conditional']['unresolved'][i,j]
        scores={'Product':product,'TrueP':original['true_P'][i,j],
                'Circular_long':arrays['circular']['pooled_P'][i,j],
                'Nuclear_final_lambda1':arrays['nuclear']['final_P'][i,j],
                'Q_uniform':arrays['conditional']['q_uniform'][i,j],'Q_oracle':arrays['conditional']['q_oracle'][i,j]}
        domains={'raw':np.ones(len(i),bool),'U':u,'V':v}
        for name,s in scores.items():
            for domain,mask in domains.items():
                exact=auc(a[mask],s[mask],0)
                for eps in (1e-12,1e-10):rows.append({'seed':seed,'score':name,'domain':domain,'epsilon':eps,'exact':exact,'tolerant':auc(a[mask],s[mask],eps)})
            exact=auc(a,s*u,0)
            for eps in (1e-12,1e-10):rows.append({'seed':seed,'score':name,'domain':'zero_adjusted','epsilon':eps,'exact':exact,'tolerant':auc(a,s*u,eps)})
    means=[]
    for name in scores:
        for domain in ('raw','U','V','zero_adjusted'):
            for eps in (1e-12,1e-10):
                group=[r for r in rows if (r['score'],r['domain'],r['epsilon'])==(name,domain,eps)]
                delta=[r['tolerant']-r['exact'] for r in group]
                means.append({'score':name,'domain':domain,'epsilon':eps,'n':len(group),'mean_shift':float(np.mean(delta)),'max_absolute_graph_shift':float(np.max(np.abs(delta)))})
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in source_hashes.items())
    return {'scope':'Updated manuscript K8 scores: longer circular, final lambda1 nuclear, unchanged product/Pstar/conditional Q. Not all penalty-grid scores.','seeds':seeds,'source_sha256':source_hashes,'records':rows,'summary':means,'max_absolute_mean_shift':max(abs(r['mean_shift']) for r in means),'conditional_Q_V_unchanged':all(r['mean_shift']==0 and r['max_absolute_graph_shift']==0 for r in means if r['score'] in ('Q_uniform','Q_oracle') and r['domain']=='V'),'no_fitting_or_generation':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    assert not args.output.exists(),'Refusing existing output'
    report=compute()
    with args.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k in ('scope','max_absolute_mean_shift','conditional_Q_V_unchanged','no_fitting_or_generation')}))
