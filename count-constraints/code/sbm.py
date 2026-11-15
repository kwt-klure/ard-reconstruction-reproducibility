import math
import numpy as np

NS = 2026110501


def parameters(grid=4096):
    alpha, zeta, rho, weight = -2.103309259461831, 1.5, .15, 180/780
    theta = 2 * np.pi * np.arange(grid) / grid
    probability = 1 / (1 + np.exp(-(alpha + zeta * np.cos(theta))))
    variance = float(np.mean((probability - rho)**2))
    delta = math.sqrt(variance / (weight * (1 - weight)))
    pin, pout = rho + (1-weight)*delta, rho - weight*delta
    assert 0 < pout < pin < 1
    return {'community_count':4, 'community_size':10, 'rho':rho,
            'p_within':pin, 'p_between':pout, 'within_dyad_fraction':weight,
            'target_probability_variance':variance,
            'matched_average_bernoulli_noise':rho*(1-rho)-variance,
            'circular_reference_alpha':alpha, 'circular_reference_zeta':zeta,
            'quadrature_grid':grid, 'circular_mean_quadrature':float(probability.mean())}


def make_block(block_id, spec):
    labels_rng = np.random.default_rng(np.random.SeedSequence([NS, 1, block_id]))
    edges_rng = np.random.default_rng(np.random.SeedSequence([NS, 3, block_id]))
    communities = labels_rng.permutation(np.repeat(np.arange(4, dtype=np.int64), 10))
    uniforms = edges_rng.random(780)
    p = np.where(communities[:,None] == communities[None,:], spec['p_within'], spec['p_between'])
    np.fill_diagonal(p, 0)
    i,j = np.triu_indices(40, 1)
    a = np.zeros((40,40), dtype=np.uint8)
    a[i,j] = uniforms < p[i,j]
    a += a.T
    b = np.eye(8, dtype=np.uint8)[np.arange(40) % 8]
    y = a @ b
    cross = np.zeros((4,8), dtype=np.int64)
    np.add.at(cross, (communities, np.arange(40)%8), 1)
    return {'Y':y,'B':b,'hidden_A':a,'true_P':p}, {
        'communities':communities,'upper_uniforms':uniforms,'community_by_observation':cross}
