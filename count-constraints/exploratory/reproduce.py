#!/usr/bin/env python3
"""Reproduce the fixed-input analyses in a new, never-existing workspace."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
BASE = Path('analysis/ard-reconstruction')
COND = BASE / 'compstat-conditional-2026-11-04'
STAGE = BASE / 'compstat-stability-2026-11-04'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--lane', choices=('conditional', 'nuclear', 'circular', 'all'), default='all')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    requested = args.workspace.expanduser().absolute()
    destination = requested.resolve(strict=False)
    if requested.is_symlink() or destination.exists():
        p.error('workspace must be a new, non-existing path')
    if HERE == destination or HERE in destination.parents:
        p.error('workspace must be outside the frozen S2 package')
    manifest = json.loads((HERE / 'MANIFEST.json').read_text())['files']
    selected = json.loads((HERE / 'REPRODUCTION_FILES.json').read_text())['files']
    # Validate every selected input before creating the workspace.
    for rel in selected:
        if sha(HERE / rel) != manifest[rel]['sha256']:
            raise ValueError('source hash mismatch: ' + rel)
    destination.mkdir(parents=True, exist_ok=False)
    copied = {}
    for rel in selected:
        target = destination / Path(rel).relative_to('project')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / rel, target)
        if sha(target) != manifest[rel]['sha256']:
            raise ValueError('copy hash mismatch: ' + rel)
        copied[str(target.relative_to(destination))] = manifest[rel]['sha256']
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    state = {'started_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.executable,
             'lane': args.lane, 'prepare_only': args.prepare_only, 'copied_sha256': copied,
             'source_contracts': 'Portable specifications with the original scientific settings. Generated outputs live in this new workspace.',
             'commands': [], 'status': 'prepared'}
    state_path = destination / 'REPRODUCTION.json'

    def persist():
        state_path.write_text(json.dumps(state, indent=2) + '\n')

    def fail(reason):
        state['status'] = 'failed_outputs_retained'
        state['failure'] = reason
        persist()
        raise SystemExit(reason)

    def run(arguments, cap_seconds=None):
        command = [sys.executable, '-B', '-u', *map(str, arguments)]
        state['commands'].append(command)
        state['status'] = 'running'
        persist()
        log_path = destination / f'command-{len(state["commands"]):02d}.log'
        print(json.dumps({'running': command, 'log': str(log_path), 'cap_seconds': cap_seconds}), flush=True)
        with log_path.open('x') as log:
            process = subprocess.Popen(command, cwd=destination, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=(os.name == 'posix'))
            try:
                returncode = process.wait(timeout=cap_seconds)
            except subprocess.TimeoutExpired:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name == 'posix':
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
                fail('wall-clock cap reached; all partial outputs retained')
        if returncode:
            state['status'] = 'failed_outputs_retained'
            state['returncode'] = returncode
            persist()
            raise SystemExit(returncode)

    persist()
    if args.prepare_only:
        print(json.dumps({'status': 'prepared_no_computation', 'workspace': str(destination), 'files': len(copied)}))
        return
    c, s = destination / COND, destination / STAGE
    if args.lane in ('conditional', 'all'):
        run([c / 'conditional_benchmark.executed.py', '--contract', c / 'RUN_CONTRACT.json', '--output', c / 'results'])
        run([c / 'verify_independent.py', '--inputs', destination / BASE / 'jspi-rerun-2026-10-14/fresh-results',
             '--outputs', c / 'results', '--report', c / 'independent-verification-new.json'])
    if args.lane in ('nuclear', 'all'):
        run([s / 'nuclear.py', 'run', '--global-contract', s / 'RUN_CONTRACT.json'])
        finished = json.loads((s / 'nuclear/EXECUTION_FINISHED.json').read_text())
        summary = json.loads((s / 'nuclear/SUMMARY.json').read_text())
        if (finished['completed'] != 240 or finished['infrastructure_failures']
                or summary['saved_fits'] != 240 or summary['missing']):
            fail('nuclear run incomplete; inspect retained completion and summary records')
        state['nuclear_failed_fits'] = len(summary['failed'])
        state['nuclear_final_caps'] = len(summary['final_caps'])
        if summary['failed']:
            state['status'] = 'completed_with_fit_failures_outputs_retained'
            persist()
            raise SystemExit('all nuclear case records retained, but failed fits prevent a successful reproduction')
    if args.lane in ('circular', 'all'):
        spec = json.loads((s / 'RUN_CONTRACT.json').read_text())
        run([s / 'circular.py', '--contract', s / 'RUN_CONTRACT.json', '--output', s / 'circular',
             '--workers', spec['circular']['workers']], cap_seconds=spec['resources']['wall_clock_cap_seconds'])
    if args.lane == 'all':
        run([s / 'summarize_stage_a.py'])
        run([s / 'verify_stability.py', '--root', s, '--report', s / 'verification-new.json'])
    for rel, expected in copied.items():
        if sha(destination / rel) != expected:
            raise ValueError('copied source/input/contract changed: ' + rel)
    state['status'] = 'complete_outputs_retained'
    state['completed_utc'] = datetime.now(timezone.utc).isoformat()
    persist()
    print(json.dumps({'status': state['status'], 'workspace': str(destination)}))


if __name__ == '__main__':
    main()
