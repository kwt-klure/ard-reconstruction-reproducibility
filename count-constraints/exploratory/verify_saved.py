#!/usr/bin/env python3
"""Verify the exploratory graph inputs, saved fits, conditional results and scientific summaries."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE / 'project'
BASE = PROJECT / 'analysis/ard-reconstruction'
COND = BASE / 'compstat-conditional-2026-11-04'
STAGE = BASE / 'compstat-stability-2026-11-04'
INPUT = BASE / 'jspi-rerun-2026-10-14/fresh-results'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read(path):
    return json.loads(path.read_text())


def compare(a, b, label):
    if isinstance(b, dict):
        require(isinstance(a, dict) and set(a) == set(b), label + ': keys')
        for k in b:
            compare(a[k], b[k], label + '/' + str(k))
    elif isinstance(b, list):
        require(isinstance(a, list) and len(a) == len(b), label + ': list')
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, label + '/' + str(i))
    elif b is None or isinstance(b, (str, bool)):
        require(a == b, label + ': value')
    else:
        require(np.isfinite(a) and np.isfinite(b) and np.isclose(a, b, rtol=2e-12, atol=2e-12), label + ': numeric')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, help='optional new report outside S2')
    args = p.parse_args()
    if args.report:
        requested = args.report.expanduser().absolute()
        report_path = requested.resolve(strict=False)
        require(not requested.is_symlink() and not report_path.exists(), 'refusing existing report')
        require(HERE not in report_path.parents, 'report must be outside frozen package')
    original = read(HERE / 'ORIGINAL_RUN_HASHES.json')
    files = read(HERE / 'MANIFEST.json')['files']
    actual = {str(f.relative_to(HERE)) for f in HERE.rglob('*') if f.is_file()}
    require(actual == set(files) | {'MANIFEST.json'}, 'package membership differs')
    for rel, record in files.items():
        path = HERE / rel
        require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], 'package hash: ' + rel)
    omissions = read(HERE / 'OMITTED_TRACES.json')
    require(len(omissions['files']) == 80 and not list(HERE.rglob('P_draws.npy')), 'trace omission scope')
    require(sum(x['bytes'] for x in omissions['files']) == omissions['total_file_bytes'], 'trace byte count')
    omitted = {r['package_path']: r for r in omissions['files']}
    contract = read(STAGE / 'RUN_CONTRACT.json')
    for rel, expected in contract['input_sha256'].items():
        require(sha(INPUT / rel) == expected, 'input contract hash: ' + rel)
        with np.load(INPUT / rel, allow_pickle=False) as z:
            require(np.array_equal(z['hidden_A'] @ z['B'], z['Y']), 'Y != AB: ' + rel)
    require(sha(INPUT / 'results.json') == contract['metadata_sha256'], 'input metadata hash')
    for rel, expected in contract['code_sha256'].items():
        require(sha(STAGE / rel) == expected, 'code contract hash: ' + rel)
    require(sha(BASE / 'social-networks-fixed40/run_experiment.py') == contract['original_runner_sha256'], 'original runner hash')
    v = load(STAGE / 'verify_stability.py', 'saved_stage_verifier')
    cv = load(COND / 'verify_independent.py', 'conditional_independent')
    primary = load(COND / 'conditional_benchmark.executed.py', 'conditional_primary_executed')
    cond = read(COND / 'results/results.json')
    require(cond['script_sha256'] == original['conditional_code'], 'executed conditional source')
    require(cond['contract_sha256'] == original['conditional_specification'], 'conditional contract hash')
    require(cond['seeds'] == contract['seeds'] and len(cond['runs']) == 40, 'conditional case membership')
    conditional_errors = []
    for row in cond['runs']:
        name = f"run_k8_seed{row['seed']}.npz"
        checked = cv.verify_graph(INPUT / name, COND / 'results' / name)
        conditional_errors.append(checked['errors']['max_abs_q_oracle_difference'])
        expected_blocks = [(b['groups'][0], b['groups'][1], int(b['support_count'])) for b in checked['blocks']]
        require([(b['g'], b['h'], b['support_count']) for b in row['blocks']] == expected_blocks, 'conditional support metadata')
        with np.load(INPUT / name, allow_pickle=False) as z, np.load(COND / 'results' / name, allow_pickle=False) as q:
            arrays = {k: z[k] for k in z.files}
            solved = {k: q[k] for k in q.files}
            metrics, counts = primary.evaluate(arrays, solved)
            compare(row['metrics'], metrics, 'conditional metrics ' + name)
            compare(row['counts'], counts, 'conditional counts ' + name)
    for model, groups in cond['summary'].items():
        for metric, group in groups.items():
            if metric in ('zero_adjusted_full_auc', 'full_forced_layer_auc'):
                compare(group, primary.summary([r['metrics'][model][metric] for r in cond['runs']]), 'conditional aggregate')
            else:
                for domain, value in group.items():
                    compare(value, primary.summary([r['metrics'][model][metric][domain] for r in cond['runs']]), 'conditional aggregate')
    for key, value in cond['count_summary'].items():
        compare(value, primary.summary([r['counts'][key] for r in cond['runs']]), 'conditional count summary')
    for key, value in cond['paired_contrasts'].items():
        names, metric, domain = key.split('/')
        left, right = names.split('_minus_')
        differences = []
        for row in cond['runs']:
            a, b = row['metrics'][left][metric][domain], row['metrics'][right][metric][domain]
            differences.append(a - b if a is not None and b is not None else None)
        compare(value, primary.summary(differences), 'conditional paired contrast')
    tie_lane = BASE / 'compstat-manuscript-2026-11-04'
    tie_module = load(tie_lane / 'check_current_ties.py', 'current_K8_tie_audit')
    tie_report = tie_module.compute()
    compare(read(tie_lane / 'current-tie-sensitivity.json'), tie_report, 'updated K8 tie audit')
    ncontract = read(STAGE / 'nuclear/RUN_CONTRACT.json')
    require(sha(STAGE / 'nuclear/RUN_CONTRACT.json') == contract['nuclear']['own_contract_sha256'], 'nuclear contract hash')
    # Immutable saved NPZ metadata records the original solver and specification hashes.
    archived_contract = dict(contract)
    archived_contract['nuclear'] = dict(contract['nuclear'], own_contract_sha256=original['nuclear_specification'])
    archived_contract['code_sha256'] = dict(contract['code_sha256'], **{'nuclear.py':original['nuclear_code']})
    fits = []
    for k in contract['k']:
        for seed in contract['seeds']:
            for penalty in contract['nuclear']['lambda']:
                fits.append(v.verify_nuclear_fit(STAGE, archived_contract, ncontract, INPUT, k, seed, penalty))
    v.verify_nuclear_manifest(STAGE)
    circular = read(STAGE / 'circular/results.json')
    require(len(circular['runs']) == 80, 'circular membership')
    require([(r['k'], r['seed']) for r in circular['runs']] == [(k, s) for k in contract['k'] for s in contract['seeds']], 'circular order')
    scalar_checks = 0
    for record in circular['runs']:
        name = f"run_k{record['k']}_seed{record['seed']}"
        directory = STAGE / 'circular' / name
        complete = read(directory / 'complete.json')
        require(complete == record, 'circular index versus metadata')
        require(complete['input_sha256'] == contract['input_sha256'][name + '.npz'], 'circular input hash')
        for filename, expected in complete['output_sha256'].items():
            path = directory / filename
            if filename == 'P_draws.npy':
                require(omitted[str(path.relative_to(HERE))]['sha256_recorded'] == expected, 'omission recorded hash')
            else:
                require(sha(path) == expected, 'circular included hash')
        with np.load(directory / 'summary.npz', allow_pickle=False) as means:
            for key in ('pooled_P', 'first_half_P', 'last_half_P', 'group12_P', 'group34_P'):
                v.feasible(means[key], name + '/' + key)
            require(means['chain_mean_P'].shape == (4, 40, 40), 'chain means shape')
            for matrix in means['chain_mean_P']:
                v.feasible(matrix, 'chain mean')
            v.close(means['pooled_P'], means['chain_mean_P'].mean(axis=0), 'pooled chain identity')
            v.close(means['group12_P'], means['chain_mean_P'][:2].mean(axis=0), 'group12 identity')
            v.close(means['group34_P'], means['chain_mean_P'][2:].mean(axis=0), 'group34 identity')
            v.close(means['pooled_P'], (means['first_half_P'] + means['last_half_P']) / 2, 'half identity')
        scalars = np.load(directory / 'scalar_draws.npy', allow_pickle=False)
        require(scalars.shape == (4, 3200, 3) and scalars.dtype == np.float64 and np.isfinite(scalars).all(), 'scalar traces')
        with np.load(directory / 'diagnostics.npz', allow_pickle=False) as archive:
            diag = {key.removeprefix('P_'): archive[key] for key in archive.files if key.startswith('P_')}
            require(all(array.shape == (780,) for array in diag.values()), 'P diagnostic dimensions')
            for key in ('rhat', 'bulk_ess', 'mean_ess', 'mean_mcse'):
                require(np.isfinite(diag[key]).all() and (diag[key] > 0).all(), 'saved P diagnostic values')
            compare(complete['matrix_diagnostics'], v.diagnostic_summary(diag, 1.01, 400), 'P diagnostic summary')
            v.verify_diagnostics(scalars, archive, 'scalar', complete['scalar_diagnostics'], contract['circular'], range(3))
            scalar_checks += 3
        with np.load(INPUT / (name + '.npz'), allow_pickle=False) as z:
            y, b = z['Y'], z['B']
        for index, (kind, offset) in enumerate(zip(contract['circular']['initializations'], contract['circular']['seed_offsets'])):
            state = read(directory / f'chain{index + 1}-state.json')
            compare(state['initial'], v.expected_initial(record['seed'] + offset, kind), 'initial state')
            alpha, angles = state['final']['alpha'], np.asarray(state['final']['angles'])
            v.close(alpha, scalars[index, -1, 0], 'last alpha')
            last_p = 1 / (1 + np.exp(-(alpha + 1.5 * np.cos(angles[:, None] - angles[None, :]))))
            np.fill_diagonal(last_p, 0)
            rates = last_p @ b
            v.close(scalars[index, -1, 1:], [np.sum(y * np.log(np.maximum(rates, 1e-300)) - rates), np.mean((rates - y) ** 2)], 'last scalar from final state')
            generator = np.random.PCG64()
            generator.state = state['final']['rng_state']
    # Re-score the included mean matrices; never run a fitting routine.
    summary = load(STAGE / 'summarize_stage_a.py', 'saved_stage_summary')
    with tempfile.TemporaryDirectory(prefix='S2 saved summary ') as temp:
        output = Path(temp)
        shutil.copyfile(STAGE / 'RUN_CONTRACT.json', output / 'RUN_CONTRACT.json')
        shutil.copytree(STAGE / 'nuclear', output / 'nuclear')
        shutil.copytree(STAGE / 'circular', output / 'circular')
        summary.HERE = output
        summary.main()
        saved, rebuilt = read(STAGE / 'SUMMARY.json'), read(output / 'SUMMARY.json')
        require(saved.pop('contract_sha256') == original['stability_specification'], 'original summary specification')
        require(rebuilt.pop('contract_sha256') == sha(STAGE / 'RUN_CONTRACT.json'), 'portable summary specification')
        compare(saved, rebuilt, 'stage aggregate')
        for name in ('summary.csv', 'paired-contrasts.csv'):
            require((STAGE / name).read_bytes() == (output / name).read_bytes(), 'stage CSV reproduction: ' + name)
    report = {'status': 'passed', 'included_files_hashed': len(files), 'original_input_archives': 80,
              'conditional_graphs_independently_checked': 40,
              'max_conditional_oracle_difference': max(conditional_errors),
              'nuclear_fits_checked': len(fits), 'nuclear_final_caps_retained': sum(r['final_capped'] for r in fits),
              'circular_saved_summary_cells_checked': 80, 'scalar_variables_checked': scalar_checks,
              'stage_summary_recomputed_from_saved_estimates': True,
              'current_K8_tie_sensitivity_recomputed': True,
              'current_K8_max_absolute_mean_AUC_shift': tie_report['max_absolute_mean_shift'],
              'omitted_P_trace_files': 80, 'omitted_P_trace_bytes': omissions['total_file_bytes'],
              'raw_P_trace_hashes_verified': False, 'raw_P_mean_or_diagnostics_recomputed': False,
              'independent_ESS_recomputed': False, 'ADMM_trajectory_replayed': False,
              'scope': 'Included data only. Full P-trace checks require regeneration; original verification reports are historical evidence.'}
    if args.report:
        with report_path.open('x') as f:
            json.dump(report, f, indent=2)
            f.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
