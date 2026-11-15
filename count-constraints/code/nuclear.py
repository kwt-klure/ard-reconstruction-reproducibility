import numpy as np


def validate_inputs(y, b):
    n, k = b.shape
    if y.shape != (n, k) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(b)):
        raise ValueError("Y/B shape or finiteness")
    if not (np.all((b == 0) | (b == 1)) and np.all(b.sum(axis=1) == 1)):
        raise ValueError("B is not a one-hot partition")
    if not (np.all(y >= 0) and np.all(y == np.floor(y))):
        raise ValueError("Y is not nonnegative integer count data")
    if not np.all(y <= b.sum(axis=0)[None, :] - b):
        raise ValueError("Y exceeds simple graph capacity")
    if not np.allclose(b.T @ y, (b.T @ y).T):
        raise ValueError("Y/B fail reciprocal block totals")


def project_constraints(x):
    result = np.clip((x + x.T) / 2, 0, 1)
    np.fill_diagonal(result, 0)
    return result


def objective(p, y, b, penalty):
    return float(0.5 * np.sum((p @ b - y) ** 2) + penalty * np.linalg.svd(p, compute_uv=False).sum())


def assert_estimate(p):
    if not (np.all(np.isfinite(p)) and np.array_equal(p, p.T)
            and np.all(np.diag(p) == 0) and np.all((p >= 0) & (p <= 1))):
        raise ValueError("Estimate is not finite, symmetric, zero-diagonal and in [0,1]")
