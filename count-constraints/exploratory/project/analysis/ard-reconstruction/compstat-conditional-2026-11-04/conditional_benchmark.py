"""Exact small-block ARD fibre enumeration. No graph generation or estimator fitting."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def auc(labels, scores, tolerance=0.0):
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return None
    diff = pos[:, None] - neg[None, :]
    return float((np.count_nonzero(diff > tolerance) + .5 * np.count_nonzero(np.abs(diff) <= tolerance)) / diff.size)


def forced_rank_score(score, f0, f1, free):
    """Rank-only layers F0 < V < F1; no probability interpretation."""
    answer = np.zeros_like(score, dtype=float)
    answer[f1] = 1.0
    if np.any(free):
        values = np.unique(score[free])
        answer[free] = (np.searchsorted(values, score[free]) + 1) / (len(values) + 1)
    return answer


def evaluate(original, solved):
    a, p, b, y = (original[x] for x in ('hidden_A', 'true_P', 'B', 'Y'))
    n = len(a)
    i, j = np.triu_indices(n, 1)
    labels = a[i, j].astype(int)
    group = b.argmax(axis=1)
    product = y[i, group[j]] * y[j, group[i]]
    u = product > 0
    f0, f1, v = (solved[x][i, j] for x in ('forced_zero', 'forced_one', 'unresolved'))
    require(np.all(labels[f0] == 0) and np.all(labels[f1] == 1), 'Forced set contradicts realized graph')
    require(np.all(f0[~u]), 'Zero-count exclusion must be forced zero')
    scores = {'Q_uniform': solved['q_uniform'][i, j], 'Q_oracle': solved['q_oracle'][i, j],
              'TrueP': p[i, j], 'Circular_saved': original['bayes_P'][i, j],
              'Nuclear_saved': original['nuclear_P'][i, j], 'Density_saved': original['density_P'][i, j],
              'Product': product, 'ZeroOnly': u.astype(float)}
    domains = {'all_pairs': np.ones_like(u), 'zero_nonexcluded_U': u, 'full_fibre_unresolved_V': v}
    m, nn = labels.sum(), (labels == 0).sum()
    mv, nv = labels[v].sum(), (labels[v] == 0).sum()
    multiplier = float(mv * nv / (m * nn))
    oracle = scores['Q_oracle']
    metrics = {}
    for name, score in scores.items():
        row = {'auc': {}, 'auc_tie_sensitivity': {}, 'brier': {}, 'conditional_expected_brier': {},
               'conditional_excess_brier': {}}
        for domain, mask in domains.items():
            row['auc'][domain] = auc(labels[mask], score[mask])
            row['auc_tie_sensitivity'][domain] = {
                str(eps): auc(labels[mask], score[mask], eps) for eps in (1e-12, 1e-10)}
            if name not in ('Product', 'ZeroOnly'):
                row['brier'][domain] = float(np.mean((score[mask] - labels[mask]) ** 2)) if mask.any() else None
                excess = float(np.mean((score[mask] - oracle[mask]) ** 2)) if mask.any() else None
                irreducible = float(np.mean(oracle[mask] * (1 - oracle[mask]))) if mask.any() else None
                row['conditional_excess_brier'][domain] = excess
                row['conditional_expected_brier'][domain] = excess + irreducible if mask.any() else None
        row['zero_adjusted_full_auc'] = auc(labels, score * u)
        fullrank = forced_rank_score(score, f0, f1, v)
        row['full_forced_layer_auc'] = auc(labels, fullrank)
        identity = 1 - multiplier * (1 - row['auc']['full_fibre_unresolved_V']) if v.any() else 1.0
        require(abs(row['full_forced_layer_auc'] - identity) < 2e-12, 'Full-forced layer identity failed')
        if name in ('Q_uniform', 'Q_oracle'):
            require(abs(row['auc']['all_pairs'] - identity) < 2e-12, 'Conditional score identity failed')
        metrics[name] = row
    counts = {'pairs': len(i), 'edges': int(m), 'nonedges': int(nn),
              'zero_excluded': int((~u).sum()), 'zero_excluded_nonedge_fraction': float((~u).sum() / nn),
              'forced_zero': int(f0.sum()), 'forced_one': int(f1.sum()),
              'extra_forced_zero_inside_U': int((f0 & u).sum()), 'unresolved_V': int(v.sum()),
              'edges_V': int(mv), 'nonedges_V': int(nv),
              'forced_zero_nonedge_fraction': float(f0.sum() / nn),
              'forced_one_edge_fraction': float(f1.sum() / m), 'full_auc_unresolved_weight': multiplier}
    return metrics, counts


def summary(values):
    available = np.array([x for x in values if x is not None], dtype=float)
    if not len(available):
        return {'n': 0, 'mean': None, 'sd': None, 'mcse': None}
    return {'n': len(available), 'mean': float(available.mean()),
            'sd': float(available.std(ddof=1)) if len(available) > 1 else None,
            'mcse': float(available.std(ddof=1) / np.sqrt(len(available))) if len(available) > 1 else None,
            'min': float(available.min()), 'max': float(available.max())}


def run(contract_path, output, pilot=False):
    contract = json.loads(contract_path.read_text())
    root = Path(__file__).resolve().parents[3]
    inputs = root / contract['inputs_relative']
    seeds = [contract['pilot_seed']] if pilot else contract['seeds']
    require(not output.exists(), 'Refusing existing output directory')
    require(digest(inputs / 'results.json') == contract['metadata_sha256'], 'Source metadata drift')
    for name, h in contract['input_sha256'].items():
        require(digest(inputs / name) == h, 'Input hash drift: ' + name)
    output.mkdir(parents=True)
    started = time.perf_counter()
    rows = []
    for seed in seeds:
        name = f'run_k8_seed{seed}.npz'
        with np.load(inputs / name, allow_pickle=False) as z:
            original = {key: z[key] for key in z.files}
        require(np.array_equal(original['hidden_A'] @ original['B'], original['Y']), 'Original Y != AB')
        require(original['B'].shape == (40, 8) and np.all(original['B'].sum(axis=0) == 5), 'Wrong design')
        tick = time.perf_counter()
        solved, blocks = reconstruct(original['Y'], original['B'], original['true_P'])
        metrics, counts = evaluate(original, solved)
        np.savez_compressed(output / name, **solved)
        fibre_count = math.prod(block['support_count'] for block in blocks)
        rows.append({'seed': seed, 'k': 8, 'blocks': blocks, 'counts': counts, 'metrics': metrics,
                     'fibre_count_exact': str(fibre_count), 'log10_fibre_count': math.log10(fibre_count),
                     'seconds': time.perf_counter() - tick})
        print(json.dumps({'seed': seed, 'forced_zero': counts['forced_zero'], 'forced_one': counts['forced_one'],
                          'V': counts['unresolved_V'], 'seconds': rows[-1]['seconds']}), flush=True)
    models = list(rows[0]['metrics'])
    aggregated = {}
    for model in models:
        aggregated[model] = {metric: {domain: summary([r['metrics'][model][metric][domain] for r in rows])
                                    for domain in rows[0]['metrics'][model][metric]}
                             for metric in ('auc', 'brier', 'conditional_expected_brier', 'conditional_excess_brier')}
        for metric in ('zero_adjusted_full_auc', 'full_forced_layer_auc'):
            aggregated[model][metric] = summary([r['metrics'][model][metric] for r in rows])
    contrasts = {}
    for left, right in [('Nuclear_saved', 'Product'), ('Q_uniform', 'Product'),
                        ('Nuclear_saved', 'Q_uniform'), ('Q_oracle', 'Q_uniform'),
                        ('Q_oracle', 'Nuclear_saved'), ('Q_uniform', 'Circular_saved')]:
        for metric in ('auc', 'brier', 'conditional_excess_brier'):
            if not rows[0]['metrics'][right][metric]:
                continue
            for domain in rows[0]['metrics'][left][metric]:
                vals = []
                for row in rows:
                    a, b = row['metrics'][left][metric][domain], row['metrics'][right][metric][domain]
                    vals.append(a - b if a is not None and b is not None else None)
                contrasts[f'{left}_minus_{right}/{metric}/{domain}'] = summary(vals)
    for name, h in contract['input_sha256'].items():
        require(digest(inputs / name) == h, 'Input changed during run')
    report = {'schema_version': 1, 'analysis': 'Exact finite-fibre conditional benchmarks on saved K8 graphs',
              'contract_sha256': digest(contract_path), 'script_sha256': digest(__file__),
              'pilot': pilot, 'n': 40, 'k': 8, 'seeds': seeds, 'input_sha256': contract['input_sha256'],
              'runs': rows, 'summary': aggregated, 'paired_contrasts': contrasts,
              'count_summary': {key: summary([r['counts'][key] for r in rows]) for key in rows[0]['counts']},
              'elapsed_seconds': time.perf_counter() - started,
              'limitations': ['Same saved40 graphs, not independent confirmation', 'K8 only',
                              'These scores use the original, shorter estimator fits',
                              'Full-forced-layer scores are ranking-only diagnostics; uniform law is selected reference']}
    with (output / 'results.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False); f.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    report = run(args.contract, args.output, args.pilot)
    print(json.dumps({'complete': len(report['runs']), 'seconds': report['elapsed_seconds']}))
