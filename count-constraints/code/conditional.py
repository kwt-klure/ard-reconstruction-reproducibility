from functools import lru_cache
import itertools
import math
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


@lru_cache(maxsize=4096)
def enumerate_bipartite(rows, cols):
    """List every binary table once, by row combinations with column pruning."""
    m, n = len(rows), len(cols)
    require(m <= 5 and n <= 5, 'This enumerator is bounded to blocks at most 5x5')
    require(all(isinstance(x, int) and 0 <= x <= n for x in rows), 'Invalid row margins')
    require(all(isinstance(x, int) and 0 <= x <= m for x in cols), 'Invalid column margins')
    require(sum(rows) == sum(cols), 'Inconsistent margin totals')
    current = np.zeros((m, n), dtype=np.uint8)
    answers = []

    def visit(i, remaining):
        if i == m:
            if not any(remaining):
                answers.append(current.copy())
            return
        for chosen in itertools.combinations(range(n), rows[i]):
            chosen_set = set(chosen)
            rest = tuple(remaining[j] - (j in chosen_set) for j in range(n))
            if min(rest, default=0) < 0 or max(rest, default=0) > m - i - 1:
                continue
            current[i] = 0
            current[i, list(chosen)] = 1
            visit(i + 1, rest)

    visit(0, cols)
    require(len(answers) > 0, 'Empty fixed-margin fibre')
    answer = np.stack(answers)
    answer.flags.writeable = False
    return answer


@lru_cache(maxsize=128)
def all_simple_graphs(m):
    require(1 <= m <= 5, 'Within group enumeration is bounded to five nodes')
    u, v = np.triu_indices(m, 1)
    masks = np.arange(1 << len(u), dtype=np.uint32)
    bits = ((masks[:, None] >> np.arange(len(u), dtype=np.uint32)) & 1).astype(np.uint8)
    answer = np.zeros((len(masks), m, m), dtype=np.uint8)
    answer[:, u, v] = bits
    answer[:, v, u] = bits
    answer.flags.writeable = False
    return answer


@lru_cache(maxsize=2048)
def enumerate_within(degrees):
    m = len(degrees)
    require(all(isinstance(x, int) and 0 <= x < m for x in degrees), 'Invalid degrees')
    candidates = all_simple_graphs(m)
    answer = candidates[np.all(candidates.sum(axis=2) == np.array(degrees)[None, :], axis=1)]
    require(len(answer) > 0, 'Empty degree-sequence fibre')
    answer.flags.writeable = False
    return answer


def block_marginals(configurations, probabilities, within=False):
    """Exact support counts; log-shifted exhaustive weighted sums using np.longdouble."""
    count = len(configurations)
    require(count > 0, 'Empty fibre')
    probabilities = np.asarray(probabilities, dtype=np.longdouble)
    if within:
        u, v = np.triu_indices(len(probabilities), 1)
        p = probabilities[u, v]
        binary = configurations[:, u, v]
    else:
        p = probabilities.ravel()
        binary = configurations.reshape(count, -1)
    require(np.all((p > 0) & (p < 1)), 'Strictly interior P-star required')
    log_odds = np.log(p) - np.log1p(-p)
    log_weights = np.sum(binary.astype(np.longdouble) * log_odds[None, :], axis=1)
    shift = log_weights.max()
    weights = np.exp(log_weights - shift)
    weighted_mean = np.sum(configurations.astype(np.longdouble) * weights[:, None, None], axis=0) / weights.sum()
    ones = configurations.sum(axis=0, dtype=np.int64)
    qu = ones.astype(float) / count
    qo = np.asarray(weighted_mean, dtype=float)
    qo[ones == 0] = 0.0
    qo[ones == count] = 1.0
    return {'support_count': count, 'ones': ones, 'q_uniform': qu, 'q_oracle': qo,
            'log_weight_normalizer': float(shift + np.log(weights.sum()))}


def reconstruct(counts, roster, truth):
    n, k = roster.shape
    require(counts.shape == (n, k) and truth.shape == (n, n), 'Input dimensions')
    require(np.isin(roster, (0, 1)).all() and np.all(roster.sum(axis=1) == 1), 'Partition required')
    require(np.all(counts == np.floor(counts)) and np.all(counts >= 0), 'Nonnegative integer counts')
    groups = [np.flatnonzero(roster[:, g]) for g in range(k)]
    qu, qo = np.zeros((n, n)), np.zeros((n, n))
    numerator, denominator = np.zeros((n, n), dtype=np.int64), np.zeros((n, n), dtype=np.int64)
    blocks = []
    for g in range(k):
        for h in range(g, k):
            left, right = groups[g], groups[h]
            ix = np.ix_(left, right)
            rows = tuple(int(x) for x in counts[left, h])
            cols = tuple(int(x) for x in counts[right, g])
            if g == h:
                configs = enumerate_within(rows)
            else:
                configs = enumerate_bipartite(rows, cols)
            result = block_marginals(configs, truth[ix], within=(g == h))
            for array, values in ((qu, result['q_uniform']), (qo, result['q_oracle']),
                                  (numerator, result['ones']), (denominator, result['support_count'])):
                array[ix] = values
                if g != h:
                    array[np.ix_(right, left)] = np.asarray(values).T
            blocks.append({'g': g, 'h': h, 'left_nodes': left.tolist(), 'right_nodes': right.tolist(),
                           'rows': rows, 'cols': cols, 'support_count': result['support_count'],
                           'log_weight_normalizer': result['log_weight_normalizer']})
    np.fill_diagonal(denominator, 0)
    offdiag = ~np.eye(n, dtype=bool)
    forced_zero = (numerator == 0) & offdiag
    forced_one = (numerator == denominator) & offdiag
    unresolved = offdiag & ~forced_zero & ~forced_one
    for q in (qu, qo):
        require(np.array_equal(q, q.T) and np.all(np.diag(q) == 0), 'Q symmetry/diagonal')
        require(np.all((q >= 0) & (q <= 1)), 'Q outside [0,1]')
        require(np.max(np.abs(q @ roster - counts)) < 2e-10, 'Q B != Y')
        require(np.all((q[unresolved] > 0) & (q[unresolved] < 1)), 'Weighted support lost to rounding')
    return {'q_uniform': qu, 'q_oracle': qo, 'forced_zero': forced_zero,
            'forced_one': forced_one, 'unresolved': unresolved,
            'support_numerator': numerator, 'support_denominator': denominator}, blocks
