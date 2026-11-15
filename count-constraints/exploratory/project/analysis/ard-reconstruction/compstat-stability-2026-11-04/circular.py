"""Four-chain, direct-probability diagnostics of the unchanged circular working posterior."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SOURCE=ROOT/'analysis/ard-reconstruction/social-networks-fixed40/run_experiment.py'
spec=importlib.util.spec_from_file_location('original_ard_runner',SOURCE)
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok,message):
    if not ok:raise ValueError(message)


def initial_state(seed,n,kind):
    rng=np.random.default_rng(seed)
    if kind=='iid':
        angles=rng.uniform(-np.pi,np.pi,n)
        alpha=float(rng.normal(-2.,1.))
    elif kind=='equally_spaced':
        angles=rng.permutation(np.linspace(-np.pi,np.pi,n,endpoint=False))
        alpha=-4.
    elif kind=='two_clusters':
        centers=rng.permutation(np.repeat([0.,np.pi],n//2))
        angles=(centers+rng.normal(0.,.1,n)+np.pi)%(2*np.pi)-np.pi
        alpha=0.
    else:raise ValueError(kind)
    return rng,alpha,angles


def sample_chain(y,b,seed,warmup,draws,kind='iid',p_draws=None):
    """Original proposal/adaptation order, augmented only by storage and chosen starts."""
    n=len(b);rng,alpha,angles=initial_state(seed,n,kind)
    initial={'alpha':alpha,'angles':angles.tolist(),'seed':seed,'kind':kind}
    p=original.probabilities(alpha,angles,1.5)
    ll=original.poisson_loglik(y,p@b)
    prior=lambda x:-.5*((x+2.)/2.)**2
    steps=np.array([.2,.7]);accepts=np.zeros(2);window_accepts=np.zeros(2)
    saved=np.zeros((draws,3));mean_p=np.zeros_like(p)
    first_sum=np.zeros_like(p);last_sum=np.zeros_like(p);upper=np.triu_indices(n,1)
    for iteration in range(warmup+draws):
        proposal_alpha=alpha+rng.normal(0,steps[0])
        proposal_p=original.probabilities(proposal_alpha,angles,1.5)
        proposal_ll=original.poisson_loglik(y,proposal_p@b)
        if np.log(rng.random())<proposal_ll-ll+prior(proposal_alpha)-prior(alpha):
            alpha,p,ll=proposal_alpha,proposal_p,proposal_ll
            window_accepts[0]+=1
            if iteration>=warmup:accepts[0]+=1
        for node in rng.permutation(n):
            proposal_angles=angles.copy()
            proposal_angles[node]=(angles[node]+rng.normal(0,steps[1])+np.pi)%(2*np.pi)-np.pi
            proposal_p=original.probabilities(alpha,proposal_angles,1.5)
            proposal_ll=original.poisson_loglik(y,proposal_p@b)
            if np.log(rng.random())<proposal_ll-ll:
                angles,p,ll=proposal_angles,proposal_p,proposal_ll
                window_accepts[1]+=1
                if iteration>=warmup:accepts[1]+=1
        if iteration<warmup and (iteration+1)%50==0:
            rates=window_accepts/np.array([50,50*n])
            steps*=np.exp(np.clip(rates-np.array([.35,.35]),-.3,.3))
            steps=np.clip(steps,.02,3.)
            window_accepts[:]=0
        if iteration>=warmup:
            index=iteration-warmup
            mean_p+=p/draws
            saved[index]=alpha,ll,np.mean((p@b-y)**2)
            if p_draws is not None:p_draws[index]=p[upper]
            if index<draws//2:first_sum+=p/(draws//2)
            else:last_sum+=p/(draws//2)
    return {'mean_P':mean_p,'first_half_P':first_sum,'last_half_P':last_sum,'scalars':saved,
            'initial':initial,'final':{'alpha':alpha,'angles':angles.tolist(),'rng_state':rng.bit_generator.state},
            'acceptance':(accepts/np.array([draws,draws*n])).tolist(),'steps':steps.tolist()}


def json_safe(x):
    if isinstance(x,dict):return {str(k):json_safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [json_safe(v) for v in x]
    if isinstance(x,np.ndarray):return json_safe(x.tolist())
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.bool_,)):return bool(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x


def diagnostic_summary(diag):
    rhat=np.asarray(diag['rhat']);ess=np.asarray(diag['bulk_ess']);mcse=np.asarray(diag['mean_mcse'])
    bad=(~np.isfinite(rhat))|(~np.isfinite(ess))|(rhat>=1.01)|(ess<400)
    return json_safe({'variables':len(rhat),'rhat_max':np.max(rhat),'rhat_q95':np.quantile(rhat,.95),
        'bulk_ess_min':np.min(ess),'bulk_ess_median':np.median(ess),'rhat_failures':int(np.sum((rhat>=1.01)|~np.isfinite(rhat))),
        'ess_failures':int(np.sum((ess<400)|~np.isfinite(ess))),'flagged_variables':int(bad.sum()),
        'mean_mcse_rms':np.sqrt(np.mean(mcse**2)),'mean_mcse_max':np.max(mcse),'screen_pass':bool(not bad.any())})


def run_cell(task):
    k,seed,contract_file,output_root=task
    started=time.perf_counter();contract=json.loads(Path(contract_file).read_text())
    name=f'run_k{k}_seed{seed}';inp=ROOT/contract['inputs_relative']/(name+'.npz')
    require(sha(inp)==contract['input_sha256'][name+'.npz'],'Input drift')
    with np.load(inp,allow_pickle=False) as z:y=z['Y'].copy();b=z['B'].copy()
    original.validate_inputs(y,b)
    destination=Path(output_root)/name;destination.mkdir(exist_ok=False)
    warmup,draws=contract['circular']['warmup'],contract['circular']['draws']
    pdraw=np.lib.format.open_memmap(destination/'P_draws.npy',mode='w+',dtype=np.float64,shape=(4,draws,780))
    scalars=np.lib.format.open_memmap(destination/'scalar_draws.npy',mode='w+',dtype=np.float64,shape=(4,draws,3))
    chain_results=[]
    for c,(kind,offset) in enumerate(zip(contract['circular']['initializations'],contract['circular']['seed_offsets'])):
        result=sample_chain(y,b,seed+offset,warmup,draws,kind,pdraw[c])
        scalars[c]=result['scalars'];pdraw.flush();scalars.flush()
        chain_results.append(result)
        with (destination/f'chain{c+1}-state.json').open('x') as f:
            json.dump(json_safe({key:result[key] for key in ('initial','final','acceptance','steps')}),f,indent=2);f.write('\n')
    # Store means before diagnosis, so warnings do not affect retention of chains/cells.
    means=np.stack([x['mean_P'] for x in chain_results])
    np.savez_compressed(destination/'summary.npz',pooled_P=means.mean(axis=0),chain_mean_P=means,
        first_half_P=np.mean([x['first_half_P'] for x in chain_results],axis=0),
        last_half_P=np.mean([x['last_half_P'] for x in chain_results],axis=0),
        group12_P=means[:2].mean(axis=0),group34_P=means[2:].mean(axis=0))
    from diagnostics import diagnose
    matrix_diag=diagnose(pdraw);scalar_diag=diagnose(scalars)
    np.savez_compressed(destination/'diagnostics.npz',**{f'P_{key}':value for key,value in matrix_diag.items()},
        **{f'scalar_{key}':value for key,value in scalar_diag.items()})
    require(sha(inp)==contract['input_sha256'][name+'.npz'],'Input changed during cell')
    record={'k':k,'seed':seed,'input_sha256':sha(inp),'warmup':warmup,'draws':draws,'chains':4,
        'matrix_diagnostics':diagnostic_summary(matrix_diag),'scalar_diagnostics':diagnostic_summary(scalar_diag),
        'elapsed_seconds':time.perf_counter()-started,
        'output_sha256':{p.name:sha(p) for p in sorted(destination.iterdir()) if p.is_file()}}
    with (destination/'complete.json').open('x') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--contract',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();contract=json.loads(args.contract.read_text())
    require(not args.output.exists(),'Refuse existing output');args.output.mkdir()
    require(sha(SOURCE)==contract['original_runner_sha256'],'Original runner drift')
    require(sha(__file__)==contract['code_sha256']['circular.py'],'Circular implementation drift')
    require(sha(HERE/'diagnostics.py')==contract['code_sha256']['diagnostics.py'],'Diagnostics drift')
    jobs=[(k,seed,str(args.contract.resolve()),str(args.output.resolve())) for k in contract['k'] for seed in contract['seeds']]
    started=time.perf_counter();records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending=[pool.submit(run_cell,t) for t in jobs]
        for future in as_completed(pending):
            record=future.result();records.append(record)
            print(json.dumps({'completed':len(records),'total':len(jobs),'k':record['k'],'seed':record['seed'],
                'matrix_flags':record['matrix_diagnostics']['flagged_variables'],'seconds':record['elapsed_seconds']}),flush=True)
    order={(k,seed):i for i,(k,seed,*_) in enumerate(jobs)}
    records.sort(key=lambda r:order[(r['k'],r['seed'])])
    report={'status':'complete','scope':'All fixed80 cells; diagnostics warnings do not trigger extra chains or exclusion',
            'contract_sha256':sha(args.contract),'code_sha256':sha(__file__),'workers':args.workers,
            'total_seconds':time.perf_counter()-started,'runs':records}
    with (args.output/'results.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'complete':len(records),'seconds':report['total_seconds']}))


if __name__=='__main__':main()
