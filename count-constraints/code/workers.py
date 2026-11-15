import hashlib
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
import numpy as np
import circular
import conditional
import diagnostics
import nuclear
import adaptive_nuclear

def read_contract():
    return json.loads((HERE / "RUN_CONTRACT.json").read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(circular.json_safe(value), handle, indent=2, allow_nan=False)
        handle.write('\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def case_name(k, seed):
    return f'run_k{k}_seed{seed}'


def population_density(alpha, grid=4096):
    # Periodic trapezoidal quadrature on the uniform angle difference.
    angles = 2 * np.pi * np.arange(grid) / grid
    return float(np.mean(1 / (1 + np.exp(-(alpha + 1.5 * np.cos(angles))))))


def calibrate(target):
    lo, hi = -20., 20.
    for _ in range(80):
        mid = (lo + hi) / 2
        if population_density(mid) < target:
            lo = mid
        else:
            hi = mid
    alpha = (lo + hi) / 2
    checks = {str(grid): population_density(alpha, grid) for grid in (2048, 4096, 8192)}
    if max(abs(value - target) for value in checks.values()) > 2e-13:
        raise ValueError('Deterministic calibration failed')
    return {'alpha': alpha, 'population_mean_target': target, 'quadrature_checks': checks}


def chain_seed(block, density_index, k, chain):
    seq = np.random.SeedSequence([NS, 2, block, density_index, k, chain])
    return int(seq.generate_state(1, dtype=np.uint64)[0])


def generate_block(block, alphas, n=40):
    rng = np.random.default_rng(np.random.SeedSequence([NS, 1, block]))
    angles = rng.uniform(-np.pi, np.pi, n)
    upper = np.triu_indices(n, 1)
    uniforms = rng.random(len(upper[0]))
    graphs = []
    for alpha in alphas:
        p = circular.original.probabilities(alpha, angles, 1.5)
        a = np.zeros((n, n), dtype=np.uint8)
        a[upper] = uniforms < p[upper]
        a += a.T
        graphs.append((a, p))
    return angles, uniforms, graphs


def read_input(density, k, seed):
    name = density + '/' + case_name(k, seed) + '.npz'
    path = HERE / 'inputs' / name
    manifest = json.loads((HERE / 'inputs/MANIFEST.json').read_text())
    if sha(path) != manifest[name]:
        raise ValueError('Input drift: ' + name)
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}, manifest[name]


def run_circular(task):
    density, k, seed = task
    tick = time.monotonic()
    contract = read_contract()
    data, input_hash = read_input(density, k, seed)
    y, b = data['Y'], data['B']
    dest = HERE / 'circular' / density / case_name(k, seed)
    dest.mkdir(parents=True, exist_ok=False)
    warmup, draws = contract['circular']['warmup'], contract['circular']['draws']
    pdraw = np.lib.format.open_memmap(dest / 'P_draws.npy', mode='w+', dtype=np.float64, shape=(4, draws, 780))
    scalars = np.lib.format.open_memmap(dest / 'scalar_draws.npy', mode='w+', dtype=np.float64, shape=(4, draws, 3))
    chains = []
    for c, kind in enumerate(contract['circular']['initializations']):
        result = circular.sample_chain(y, b, chain_seed(seed, DENSITIES.index(density), k, c), warmup, draws, kind, pdraw[c])
        scalars[c] = result['scalars']
        pdraw.flush(); scalars.flush()
        write_json(dest / f'chain{c+1}-state.json', {key: result[key] for key in ('initial', 'final', 'acceptance', 'steps')})
        chains.append(result)
    means = np.stack([r['mean_P'] for r in chains])
    np.savez_compressed(dest / 'summary.npz', pooled_P=means.mean(axis=0), chain_mean_P=means,
                        first_half_P=np.mean([r['first_half_P'] for r in chains], axis=0),
                        last_half_P=np.mean([r['last_half_P'] for r in chains], axis=0),
                        group12_P=means[:2].mean(axis=0), group34_P=means[2:].mean(axis=0))
    md, sd = diagnostics.diagnose(pdraw), diagnostics.diagnose(scalars)
    np.savez_compressed(dest / 'diagnostics.npz', **{f'P_{key}': value for key, value in md.items()},
                        **{f'scalar_{key}': value for key, value in sd.items()})
    record = {'density': density, 'k': k, 'seed': seed, 'input_sha256': input_hash,
              'chains': 4, 'warmup': warmup, 'draws': draws,
              'matrix_diagnostics': circular.diagnostic_summary(md), 'scalar_diagnostics': circular.diagnostic_summary(sd),
              'elapsed_seconds': time.monotonic() - tick,
              'output_sha256': {p.name: sha(p) for p in sorted(dest.iterdir()) if p.is_file()}}
    write_json(dest / 'complete.json', record)
    return record


def run_nuclear(task):
    density, k, seed, penalty = task
    tick = time.monotonic()
    contract = read_contract(); settings = contract['nuclear']
    data, input_hash = read_input(density, k, seed)
    dest = HERE / 'nuclear' / density
    dest.mkdir(parents=True, exist_ok=True)
    stem = case_name(k, seed) + f'_lambda{penalty:g}'
    if (dest / (stem + '.npz')).exists() or (dest / (stem + '.json')).exists():
        raise ValueError('Refuse existing nuclear output')
    adaptive = adaptive_nuclear
    early, final, metadata = adaptive.fit_nuclear(data['Y'], data['B'], penalty=penalty, rho=settings['rho'],
        max_iterations=settings['iteration_cap'], tolerance=settings['tolerance'],
        early_tolerance=settings['early_tolerance'], early_cap=settings['early_cap'])
    nuclear.assert_estimate(early); nuclear.assert_estimate(final)
    np.savez_compressed(dest / (stem + '.npz'), early_P=early, final_P=final)
    record = {'density': density, 'k': k, 'seed': seed, 'penalty': penalty,
              'input_sha256': input_hash, 'status': 'converged' if metadata['final']['converged'] else 'iteration_cap',
              **metadata, 'npz_sha256': sha(dest / (stem + '.npz')), 'seconds': time.monotonic() - tick}
    write_json(dest / (stem + '.json'), record)
    return record


def run_conditional(task):
    density, k, seed = task
    tick = time.monotonic(); read_contract()
    data, input_hash = read_input(density, k, seed)
    solved, blocks = conditional.reconstruct(data['Y'], data['B'], data['true_P'])
    dest = HERE / 'conditional' / density
    dest.mkdir(parents=True, exist_ok=True)
    stem = case_name(k, seed)
    if (dest / (stem + '.npz')).exists():
        raise ValueError('Refuse existing conditional output')
    np.savez_compressed(dest / (stem + '.npz'), **solved)
    record = {'density': density, 'k': k, 'seed': seed, 'blocks': blocks,
              'input_sha256': input_hash, 'npz_sha256': sha(dest / (stem + '.npz')),
              'full_fibre_support': str(math.prod(b['support_count'] for b in blocks)),
              'seconds': time.monotonic() - tick}
    write_json(dest / (stem + '.json'), record)
    return record
