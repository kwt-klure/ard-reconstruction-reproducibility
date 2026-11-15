import numpy as np

ZETA = 1.5


def probabilities(alpha, angles, zeta=ZETA):
    logits = alpha + zeta * np.cos(angles[:, None] - angles[None, :])
    p = 1.0 / (1.0 + np.exp(-logits))
    np.fill_diagonal(p, 0.0)
    return p


def poisson_loglik(y, rates):
    # Omit log(y!), constant in all parameter updates; no Bernoulli-edge inputs.
    return float(np.sum(y * np.log(np.maximum(rates, 1e-300)) - rates))
