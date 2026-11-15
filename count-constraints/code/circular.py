import numpy as np
import original


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
