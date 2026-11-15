"""Recreate the synthetic graphs and optional expensive fits in a fresh study directory."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
import os
from pathlib import Path
import shutil

for variable in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import sbm
import workers

BASE = Path(__file__).resolve().parents[1]

def configure(directory):
    workers.HERE = Path(directory)
    contract = workers.read_contract()
    workers.NS = contract["rng"]["namespace"]
    workers.DENSITIES = [den["id"] for den in contract["densities"]]
    return contract

def generate(study, directory, compare):
    contract = configure(directory)
    inputs = directory / "inputs"
    inputs.mkdir()
    manifest = {}
    def save(relative, **arrays):
        path = inputs / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if compare:
            with np.load(BASE / "studies" / study / "inputs" / relative, allow_pickle=False) as prior:
                if set(prior.files) != set(arrays):
                    raise ValueError(f"Schema differs for {relative}")
                for key, value in arrays.items():
                    if value.dtype.kind in "fc":
                        np.testing.assert_allclose(value, prior[key], rtol=0, atol=2e-14)
                    else:
                        np.testing.assert_array_equal(value, prior[key])
        np.savez_compressed(path, **arrays)
        manifest[str(relative)] = workers.sha(path)
    for seed in contract["seeds"]:
        if study == "sbm":
            spec = sbm.parameters()
            for key in spec:
                np.testing.assert_allclose(spec[key],contract["sbm"][key],rtol=0,atol=2e-14)
            arrays, latent = sbm.make_block(seed, spec)
            save(Path("blocks") / f"block_seed{seed}.npz", **latent)
            save(Path("sbm") / f"run_k8_seed{seed}.npz", **arrays)
            continue
        alphas = [-2.8, workers.calibrate(.15)["alpha"], workers.calibrate(.25)["alpha"]]
        np.testing.assert_allclose(alphas,[d["alpha"] for d in contract["densities"]],rtol=0,atol=2e-14)
        angles, uniforms, graphs = workers.generate_block(seed, alphas)
        save(Path("blocks") / f"block_seed{seed}.npz", angles=angles, upper_uniforms=uniforms)
        for density, (a,p) in zip(workers.DENSITIES, graphs):
            for k in contract["k"]:
                b = np.eye(k, dtype=np.uint8)[np.arange(40) % k]
                save(Path(density) / f"run_k{k}_seed{seed}.npz", Y=a@b, B=b, hidden_A=a, true_P=p)
    workers.write_json(inputs / "MANIFEST.json",manifest)
    print(f"{study}: generated {len(manifest)} synthetic archives")

def fit_one(job):
    phase, task = job
    return getattr(workers, "run_" + phase)(task)["seed"]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=["fresh","sbm"])
    parser.add_argument("phase", choices=["generate","conditional","nuclear","circular"])
    parser.add_argument("directory", type=Path, help="New directory on generate; same directory for subsequent phases.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--compare-inputs", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.phase == "generate":
        directory.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(BASE/"studies"/args.study/"RUN_CONTRACT.json",directory/"RUN_CONTRACT.json")
        generate(args.study, directory, args.compare_inputs)
        return
    contract = configure(directory)
    if contract["rng"]["namespace"] != json.loads((BASE/"studies"/args.study/"RUN_CONTRACT.json").read_text())["rng"]["namespace"]:
        raise ValueError("Study design does not match")
    (directory / args.phase).mkdir()
    ks = [8] if args.phase == "conditional" else contract["k"]
    jobs = list(product(workers.DENSITIES, ks, contract["seeds"]))
    if args.phase == "nuclear":
        jobs = [(*job, lam) for job in jobs for lam in contract["nuclear"]["penalties"]]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=configure, initargs=(directory,)) as pool:
        for i,_ in enumerate(pool.map(fit_one, [(args.phase,job) for job in jobs]),1):
            if i % 20 == 0:
                print(f"{args.study} {args.phase}: {i}/{len(jobs)}", flush=True)

if __name__ == "__main__":
    main()
