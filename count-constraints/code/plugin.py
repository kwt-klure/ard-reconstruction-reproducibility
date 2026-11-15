import hashlib
import json
import math
from pathlib import Path
import numpy as np
import conditional as enumerator
import evaluate as evaluator
from evaluate import require


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def read_npz(path, keys):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in keys}


def validate_matrix(p):
    p = np.asarray(p)
    require(p.ndim == 2 and p.shape[0] == p.shape[1], 'Matrix must be square')
    require(np.isfinite(p).all() and np.array_equal(p, p.T), 'Nonfinite/asymmetric matrix')
    require(np.all(np.diag(p) == 0) and np.all((p >= 0) & (p <= 1)), 'Invalid probabilities')


def clip_matrix(p, epsilon):
    validate_matrix(p)
    require(0 < epsilon < .5, 'Invalid fixed clipping epsilon')
    out = np.clip(p, epsilon, 1 - epsilon)
    np.fill_diagonal(out, 0.)
    return out


def matrix_bounds(p, v=None, epsilon=None):
    i, j = np.triu_indices(len(p), 1)
    x = p[i, j]
    answer = dict(min=float(x.min()), max=float(x.max()), zeros=int((x == 0).sum()),
                  ones=int((x == 1).sum()), pairs=len(x))
    if epsilon is not None:
        changed = (x < epsilon) | (x > 1 - epsilon)
        answer.update(epsilon=epsilon, clipped=int(changed.sum()), clipped_fraction=float(changed.mean()),
                      clipped_V=int(changed[v[i, j]].sum()),
                      clipped_fraction_V=float(changed[v[i, j]].mean()) if v.any() else None)
    return answer


def condition_matrix(y, b, p):
    """Observable-only interface. Exact feasible support is independent of weights.

    Unlike the old oracle wrapper this permits float64 marginals rounding to 0/1
    on exact unresolved pairs; such rounding is diagnosed, never used to redefine V.
    """
    validate_matrix(p)
    n, k = b.shape
    require(y.shape == b.shape and p.shape == (n, n), 'Input dimensions')
    require(np.isin(b, (0, 1)).all() and np.all(b.sum(axis=1) == 1), 'Partition required')
    require(np.all(y >= 0) and np.all(y == np.floor(y)), 'Integer counts required')
    off = ~np.eye(n, dtype=bool)
    require(np.all((p[off] > 0) & (p[off] < 1)), 'Conditioning requires interior weights')
    groups = [np.flatnonzero(b[:, g]) for g in range(k)]
    q = np.zeros((n, n))
    logit_score = np.zeros((n, n))
    support_num = np.zeros((n, n), dtype=np.int64)
    support_den = np.zeros((n, n), dtype=np.int64)
    blocks = []
    for g in range(k):
        for h in range(g, k):
            left, right = groups[g], groups[h]
            ix = np.ix_(left, right)
            rows = tuple(int(x) for x in y[left, h])
            cols = tuple(int(x) for x in y[right, g])
            configs = (enumerator.enumerate_within(rows) if g == h
                       else enumerator.enumerate_bipartite(rows, cols))
            probs = np.asarray(p[ix], dtype=np.longdouble)
            if g == h:
                ii, jj = np.triu_indices(len(left), 1)
                edge_p, bits = probs[ii, jj], configs[:, ii, jj]
            else:
                edge_p, bits = probs.ravel(), configs.reshape(len(configs), -1)
            log_odds = np.log(edge_p) - np.log1p(-edge_p)
            logw = np.sum(bits.astype(np.longdouble) * log_odds[None, :], axis=1)
            weights = np.exp(logw - logw.max())
            present = np.sum(configs.astype(np.longdouble) * weights[:, None, None], axis=0)
            absent = np.sum((1 - configs).astype(np.longdouble) * weights[:, None, None], axis=0)
            value = np.asarray(present / (present + absent), dtype=np.float64)
            ones = configs.sum(axis=0, dtype=np.int64)
            value[ones == 0] = 0.
            value[ones == len(configs)] = 1.
            # Forced edges receive placeholders; exact support masks supply their
            # rank layer. Only unresolved finite logits are ever used as scores.
            free = (ones > 0) & (ones < len(configs))
            require(np.all(present[free] > 0) and np.all(absent[free] > 0), 'Marginal mass underflow')
            logits = np.zeros_like(value)
            logits[free] = np.asarray(np.log(present[free]) - np.log(absent[free]), dtype=float)
            for arr, val in ((q, value), (support_num, ones), (support_den, len(configs))):
                arr[ix] = val
                if g != h:
                    arr[np.ix_(right, left)] = np.asarray(val).T
            logit_score[ix] = logits
            if g != h:
                logit_score[np.ix_(right, left)] = logits.T
            blocks.append(dict(g=g, h=h, count=len(configs),
                               logweight_span=float(logw.max() - logw.min()),
                               weight_underflows=int((weights == 0).sum())))
    np.fill_diagonal(support_den, 0)
    unresolved = off & (support_num > 0) & (support_num < support_den)
    validate_matrix(q)
    residual = float(np.max(np.abs(q @ b - y)))
    require(residual < 2e-10, 'Conditional QB != Y')
    return q, logit_score, dict(max_constraint_error=residual,
                   rounded_boundary_V=int(np.count_nonzero(np.triu(unresolved & ((q == 0) | (q == 1)), 1))),
                   weight_underflows=sum(x['weight_underflows'] for x in blocks),
                   max_logweight_span=max(x['logweight_span'] for x in blocks),
                   fibre_count_exact=str(math.prod(x['count'] for x in blocks)), blocks=blocks), support_num, support_den


def score_matrix(p, arrays, reference, logit_score=None):
    validate_matrix(p)
    i, j = np.triu_indices(len(p), 1)
    labels, scores = arrays['hidden_A'][i, j].astype(int), p[i, j]
    v, oracle = reference['unresolved'][i, j], reference['q_oracle'][i, j]
    primary_scores = scores if logit_score is None else logit_score[i, j]
    full_scores = scores if logit_score is None else evaluator.forced_rank_score(
        primary_scores, reference['forced_zero'][i, j], reference['forced_one'][i, j], v)
    metrics = dict(auc_full=evaluator.auc(labels, full_scores),
                   auc_V=evaluator.auc(labels[v], primary_scores[v]),
                   auc_full_probability=evaluator.auc(labels, scores),
                   auc_V_probability=evaluator.auc(labels[v], scores[v]),
                   brier_A_V=float(np.mean((scores[v] - labels[v]) ** 2)) if v.any() else None,
                   conditional_excess_brier_V=float(np.mean((scores[v] - oracle[v]) ** 2)) if v.any() else None,
                   max_constraint_error=float(np.max(np.abs(p @ arrays['B'] - arrays['Y']))))
    for eps in (1e-12, 1e-10):
        metrics[f'auc_V_tie_{eps:g}'] = evaluator.auc(labels[v], scores[v], eps)
        metrics[f'auc_full_tie_{eps:g}'] = evaluator.auc(labels, scores, eps)
    metrics['auc_V_logit_minus_probability'] = (
        metrics['auc_V'] - metrics['auc_V_probability'] if metrics['auc_V'] is not None else None)
    metrics['auc_full_logit_minus_probability'] = (
        metrics['auc_full'] - metrics['auc_full_probability'] if metrics['auc_full'] is not None else None)
    return metrics


def eps_tag(eps):
    return f'{eps:g}'


def evaluate_case(case, contract, destination):
    arrays = read_npz(case['input'], ('Y', 'B', 'hidden_A', 'true_P'))
    require(arrays['B'].shape == (40, 8) and np.all(arrays['B'].sum(axis=0) == 5), 'Wrong fixed design')
    require(np.array_equal(arrays['hidden_A'] @ arrays['B'], arrays['Y']), 'Saved Y != AB')
    refkeys = ('q_uniform', 'q_oracle', 'forced_zero', 'forced_one', 'unresolved',
               'support_numerator', 'support_denominator')
    reference = read_npz(case['reference'], refkeys)
    evaluator.context(arrays, reference)
    reproduced, _ = enumerator.reconstruct(arrays['Y'], arrays['B'], arrays['true_P'])
    reproduce_errors = {}
    for key in refkeys:
        if key in ('q_uniform', 'q_oracle'):
            reproduce_errors[key] = float(np.max(np.abs(reproduced[key] - reference[key])))
            require(reproduce_errors[key] < 2e-13, 'Saved reference mismatch: ' + key)
        else:
            require(np.array_equal(reproduced[key], reference[key]), 'Saved support mismatch: ' + key)
    fits = {'Circular': read_npz(case['circular'], ('pooled_P',))['pooled_P']}
    cm = json.loads(case['circular_meta'].read_text())
    fit_metadata = {'Circular': {key: cm[key] for key in ('matrix_diagnostics', 'scalar_diagnostics')}}
    require(cm['input_sha256'] == sha(case['input']), 'Circular input hash mismatch')
    for penalty, path in case['nuclear'].items():
        meta = json.loads(path.with_suffix('.json').read_text())
        require(meta['npz_sha256'] == sha(path), 'Nuclear matrix hash mismatch')
        fits['Nuclear_lambda' + penalty] = read_npz(path, ('final_P',))['final_P']
        fit_metadata['Nuclear_lambda' + penalty] = {key: meta[key] for key in ('status', 'early', 'final')}
    matrices = {'Q_uniform': reference['q_uniform'], 'Q_oracle': reference['q_oracle'], **fits}
    logit_scores = {}
    diagnostics, bounds = {}, {}
    epsilons = [contract['construction']['epsilon_primary'], *contract['construction']['epsilon_sensitivity']]
    v = reference['unresolved']
    for name, p in fits.items():
        validate_matrix(p)
        bounds[name] = matrix_bounds(p)
        variants = [(name, p)] if name == 'Circular' else [
            (name + '_eps' + eps_tag(eps), clip_matrix(p, eps)) for eps in epsilons]
        for variant, weight_p in variants:
            if name != 'Circular':
                eps = float(variant.split('_eps')[-1])
                bounds[variant] = matrix_bounds(p, v, eps)
                matrices['Clipped_' + variant] = weight_p
            q, logits, diag, num, den = condition_matrix(arrays['Y'], arrays['B'], weight_p)
            require(np.array_equal(num, reference['support_numerator']) and
                    np.array_equal(den, reference['support_denominator']), 'Plug-in support drift')
            qname = 'Q_plugin_' + variant
            matrices[qname], diagnostics[qname] = q, diag
            logit_scores[qname] = logits
    changes = []
    primary = eps_tag(epsilons[0])
    for name in fits:
        if name == 'Circular':
            continue
        base = matrices[f'Q_plugin_{name}_eps{primary}']
        for eps in epsilons[1:]:
            score = matrices[f'Q_plugin_{name}_eps{eps_tag(eps)}']
            diff = score - base
            changes.append(dict(method=name, epsilon=eps, reference_epsilon=epsilons[0],
                                max_abs=float(np.max(np.abs(diff))),
                                rms_V=float(np.sqrt(np.mean(diff[v] ** 2))) if v.any() else None,
                                max_abs_V=float(np.max(np.abs(diff[v]))) if v.any() else None))
    i, j = np.triu_indices(40, 1)
    labels = arrays['hidden_A'][i, j]
    row = dict(cohort=case['cohort'], seed=case['seed'],
               metrics={name: score_matrix(p, arrays, reference, logit_scores.get(name)) for name, p in matrices.items()},
               counts=dict(V=int(v[i, j].sum()), edges_V=int(labels[v[i, j]].sum()),
                           nonedges_V=int(np.sum(labels[v[i, j]] == 0))),
               reference_reproduction=reproduce_errors, fit_metadata=fit_metadata,
               bounds=bounds, conditional_diagnostics=diagnostics, epsilon_score_changes=changes)
    outdir = destination / case['cohort']
    outdir.mkdir(exist_ok=True)
    np.savez_compressed(outdir / f"run_k8_seed{case['seed']}.npz", **matrices,
                        **{name + '_logit_score': score for name, score in logit_scores.items()},
                        unresolved=v, forced_zero=reference['forced_zero'], forced_one=reference['forced_one'],
                        support_numerator=reference['support_numerator'], support_denominator=reference['support_denominator'])
    write_json(outdir / f"run_k8_seed{case['seed']}.json", row)
    return row


def contrast_specs(models, contract):
    pairs = []
    for plugin in [m for m in models if m.startswith('Q_plugin_')]:
        variant = plugin.removeprefix('Q_plugin_')
        raw = variant.split('_eps')[0]
        pairs += [(plugin, raw), (plugin, 'Q_uniform'), ('Q_oracle', plugin)]
        if '_eps' in variant:
            pairs.append((plugin, 'Clipped_' + variant))
            pairs.append(('Clipped_' + variant, raw))
    primary = eps_tag(contract['construction']['epsilon_primary'])
    for lam in (.25, 1, 4):
        prefix = f'Q_plugin_Nuclear_lambda{lam:g}_eps'
        for epsilon in contract['construction']['epsilon_sensitivity']:
            pairs.append((prefix + eps_tag(epsilon), prefix + primary))
    return pairs


def summarize(rows, contract):
    summary, contrasts = [], []
    metrics = list(rows[0]['metrics']['Q_uniform'])
    models = list(rows[0]['metrics'])
    for cohort in contract['scope']['cohorts']:
        group = [r for r in rows if r['cohort'] == cohort]
        require(len(group) == 40, 'Incomplete cohort')
        seeds = [r['seed'] for r in group]
        for model in models:
            for metric in metrics:
                values = [r['metrics'][model][metric] for r in group]
                reasons = [None if x is not None else 'undefined:domain_empty_or_single_class' for x in values]
                summary.append(dict(cohort=cohort, model=model, metric=metric,
                                    **evaluator.block_stats(values, reasons, seeds)))
        for left, right in contrast_specs(models, contract):
            for metric in metrics:
                values = [(r['metrics'][left][metric] - r['metrics'][right][metric])
                          if r['metrics'][left][metric] is not None and r['metrics'][right][metric] is not None
                          else None for r in group]
                reasons = [None if x is not None else 'undefined:domain_empty_or_single_class' for x in values]
                contrasts.append(dict(cohort=cohort, left=left, right=right, metric=metric,
                                      **evaluator.block_stats(values, reasons, seeds)))
    return summary, contrasts
