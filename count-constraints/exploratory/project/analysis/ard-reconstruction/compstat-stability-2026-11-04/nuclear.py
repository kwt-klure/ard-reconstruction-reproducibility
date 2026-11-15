"""Stage A1/A2: fixed-input, fixed-grid nuclear ADMM stability evidence.

The estimator accepts only Y/B and prespecified solver settings. Evaluation
uses hidden arrays only after fitting. Original operation order is retained.
No data generation, tuning, retry, or convergence theorem is supplied here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
INPUT = HERE.parent / "jspi-rerun-2026-10-14/fresh-results"
ORIGINAL = HERE.parent / "social-networks-fixed40/run_experiment.py"
OUTPUT = HERE / "nuclear"
SEEDS = (101, 202, 303, 404, *range(1001, 1037))
KS = (4, 8)
PENALTIES = (0.25, 1.0, 4.0)
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json_new(path, data):
    """Never silently replace evidence from a prior execution."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


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


def fit_nuclear(y, b, penalty=1.0, rho=2.0, max_iterations=50000,
                tolerance=1e-7, early_tolerance=1e-6, early_cap=10000):
    """Original consensus ADMM with an additional same-trajectory checkpoint.

    Return feasible constrained matrices at the first original stopping point
    and at the stricter fixed stopping point. No hidden truth enters this API.
    """
    validate_inputs(y, b)
    if not (0 < tolerance <= early_tolerance and 0 < early_cap <= max_iterations):
        raise ValueError("Invalid checkpoint/stopping settings")
    if penalty <= 0 or rho <= 0:
        raise ValueError("penalty and rho must be positive")
    n = len(b)
    inverse = np.linalg.inv(b @ b.T + 2 * rho * np.eye(n))
    data_term = y @ b.T
    p = np.zeros((n, n))
    lowrank, constrained, u, v = [np.zeros_like(p) for _ in range(4)]
    early_p, early_metadata = None, None
    converged = False
    for iteration in range(1, max_iterations + 1):
        p = (data_term + rho * (lowrank - u + constrained - v)) @ inverse
        old_lowrank, old_constrained = lowrank, constrained
        left, singular, right = np.linalg.svd(p + u, full_matrices=False)
        lowrank = (left * np.maximum(singular - penalty / rho, 0)) @ right
        constrained = project_constraints(p + v)
        u += p - lowrank
        v += p - constrained
        primal = float(np.sqrt(np.sum((p - lowrank) ** 2) + np.sum((p - constrained) ** 2)))
        dual = float(rho * np.linalg.norm((lowrank - old_lowrank) + (constrained - old_constrained)))
        if not np.isfinite(primal + dual):
            raise FloatingPointError("Nonfinite ADMM residual")
        if early_p is None and ((primal < early_tolerance and dual < early_tolerance) or iteration == early_cap):
            early_p = constrained.copy()
            early_metadata = {"iterations": iteration,
                              "converged": bool(primal < early_tolerance and dual < early_tolerance),
                              "primal_residual": primal, "dual_residual": dual,
                              "objective": objective(early_p, y, b, penalty)}
        if primal < tolerance and dual < tolerance:
            converged = True
            break
    final_metadata = {"iterations": iteration, "converged": converged,
                      "primal_residual": primal, "dual_residual": dual,
                      "objective": objective(constrained, y, b, penalty)}
    assert early_p is not None
    return early_p, constrained, {"early": early_metadata, "final": final_metadata}


def assert_estimate(p):
    if not (np.all(np.isfinite(p)) and np.array_equal(p, p.T)
            and np.all(np.diag(p) == 0) and np.all((p >= 0) & (p <= 1))):
        raise ValueError("Estimate is not finite, symmetric, zero-diagonal and in [0,1]")


def auc(labels, scores):
    positive, negative = scores[labels == 1], scores[labels == 0]
    if not len(positive) or not len(negative):
        return None
    difference = positive[:, None] - negative[None, :]
    return float(np.mean((difference > 0) + 0.5 * (difference == 0)))


def metrics(p, y, b, a, truth):
    upper = np.triu_indices(len(b), 1)
    directed = y[:, np.argmax(b, axis=1)]
    mask = ((directed > 0) & (directed.T > 0))[upper]
    labels, scores = a[upper], p[upper]
    return {"auc": auc(labels, scores),
            "probability_rmse": float(np.sqrt(np.mean((scores - truth[upper]) ** 2))),
            "ard_rmse": float(np.sqrt(np.mean((p @ b - y) ** 2))),
            "within_u_auc": auc(labels[mask], scores[mask]),
            "zero_adjusted_auc": auc(labels, scores * mask),
            "u_edges": int(np.count_nonzero(labels[mask] == 1)),
            "u_nonedges": int(np.count_nonzero(labels[mask] == 0))}


def difference_metrics(before, after):
    upper = np.triu_indices(len(before), 1)
    delta = (after - before)[upper]
    return {"rms": float(np.sqrt(np.mean(delta ** 2))),
            "max_abs": float(np.max(np.abs(delta)))}


def fit_filename(k, seed, penalty):
    return f"run_k{k}_seed{seed}_lambda{penalty:g}.npz"


def prepare(workers=2):
    for name in THREAD_VARS:
        if os.environ.get(name) != "1":
            raise RuntimeError(f"Set {name}=1 before importing NumPy")
    files = {f"run_k{k}_seed{seed}.npz": sha256(INPUT / f"run_k{k}_seed{seed}.npz")
             for k in KS for seed in SEEDS}
    contract = {"schema_version": 1, "created_utc": utc_now(), "stage": "A1/A2 only",
                "input_directory": str(INPUT), "input_sha256": files,
                "solver_source": str(Path(__file__).resolve()), "solver_sha256": sha256(__file__),
                "original_method_source": str(ORIGINAL), "original_method_sha256": sha256(ORIGINAL),
                "n": 40, "ks": KS, "seeds": SEEDS, "penalties": PENALTIES, "fits": 240,
                "initialization": "all P, lowrank, constrained, u, v zero",
                "rho": 2.0, "max_iterations": 50000, "tolerance": 1e-7,
                "early_tolerance": 1e-6, "early_cap": 10000,
                "early_rule": "first both residuals < 1e-6 or iteration 10000, whichever first",
                "workers": workers, "wall_clock_cap_seconds": 7200,
                "smoke": {"k": 4, "seed": 101, "penalty": 1.0},
                "selection": "all 80 cells at all 3 penalties, including capped/failed fits; no winner selection",
                "fit_input_boundary": "Y and B only; A/P* used after fitting for evaluation",
                "runtime": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                            "thread_environment": {name: os.environ.get(name) for name in THREAD_VARS},
                            "numpy_configuration": np.__config__.CONFIG}}
    write_json_new(OUTPUT / "RUN_CONTRACT.json", contract)
    print(json.dumps({"contract": str(OUTPUT / "RUN_CONTRACT.json"), "sha256": sha256(OUTPUT / "RUN_CONTRACT.json")}))


def read_contract():
    contract = json.loads((OUTPUT / "RUN_CONTRACT.json").read_text())
    if sha256(__file__) != contract["solver_sha256"]:
        raise RuntimeError("Frozen solver source hash changed")
    if sha256(ORIGINAL) != contract["original_method_sha256"]:
        raise RuntimeError("Original method source hash changed")
    for name in THREAD_VARS:
        if os.environ.get(name) != "1":
            raise RuntimeError(f"Thread environment changed: {name}")
    return contract


def execute_one(job):
    k, seed, penalty, destination = job
    contract = read_contract()
    name = f"run_k{k}_seed{seed}.npz"
    source_hash = sha256(INPUT / name)
    if source_hash != contract["input_sha256"][name]:
        raise RuntimeError(f"Input hash changed: {name}")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    filename = fit_filename(k, seed, penalty)
    if (destination / filename).exists() or (destination / filename.replace(".npz", ".json")).exists():
        raise FileExistsError(f"Refusing overwrite: {filename}")
    metadata = {"k": k, "seed": seed, "penalty": penalty, "filename": filename,
                "source_input_sha256": source_hash,
                "source_immutable_method_sha256": contract["original_method_sha256"],
                "solver_sha256": contract["solver_sha256"],
                "contract_sha256": sha256(OUTPUT / "RUN_CONTRACT.json"), "started_utc": utc_now()}
    started = time.perf_counter()
    with np.load(INPUT / name, allow_pickle=False) as source:
        # Only observed arrays are passed to the fitting function.
        y, b = source["Y"], source["B"]
        try:
            early_p, final_p, diagnostic = fit_nuclear(y, b, penalty=penalty)
            assert_estimate(early_p)
            assert_estimate(final_p)
            metadata.update(diagnostic)
            metadata["status"] = "converged" if diagnostic["final"]["converged"] else "iteration_cap"
            metadata["matrix_change"] = difference_metrics(early_p, final_p)
            # Hidden truth is loaded only after both fits/checkpoints exist.
            a, truth = source["hidden_A"], source["true_P"]
            metadata["early_metrics"] = metrics(early_p, y, b, a, truth)
            metadata["final_metrics"] = metrics(final_p, y, b, a, truth)
            if penalty == 1.0:
                old = source["nuclear_P"]
                metadata["original_early_comparison"] = {
                    "array_equal": bool(np.array_equal(old, early_p)),
                    **difference_metrics(old, early_p)}
        except Exception as error:
            metadata["status"] = "failed"
            metadata["error"] = f"{type(error).__name__}: {error}"
            metadata["traceback"] = traceback.format_exc()
            early_p = final_p = np.full((len(b), len(b)), np.nan)
    metadata["seconds"] = time.perf_counter() - started
    metadata["finished_utc"] = utc_now()
    arrays = {"early_P": early_p, "final_P": final_p, "metadata_json": np.array(json.dumps(metadata, allow_nan=False))}
    for checkpoint in ("early", "final"):
        if checkpoint in metadata:
            arrays.update({f"{checkpoint}_{key}": np.array(value) for key, value in metadata[checkpoint].items()})
    with (destination / filename).open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    metadata["npz_sha256"] = sha256(destination / filename)
    write_json_new(destination / filename.replace(".npz", ".json"), metadata)
    return metadata


def run(global_contract):
    contract = read_contract()
    global_contract = Path(global_contract).resolve()
    # Existence alone is insufficient: parse a concrete immutable JSON contract.
    global_data = json.loads(global_contract.read_text())
    if not isinstance(global_data, dict) or not global_data:
        raise RuntimeError("Global execution contract is empty")
    files = [INPUT / name for name in contract["input_sha256"]]
    if any(sha256(path) != contract["input_sha256"][path.name] for path in files):
        raise RuntimeError("A source file changed before full execution")
    execution = {"started_utc": utc_now(), "global_contract": str(global_contract),
                 "global_contract_sha256": sha256(global_contract),
                 "nuclear_contract_sha256": sha256(OUTPUT / "RUN_CONTRACT.json"),
                 "command": sys.argv, "workers": contract["workers"]}
    write_json_new(OUTPUT / "EXECUTION_STARTED.json", execution)
    started = time.monotonic()
    jobs = [(k, seed, penalty, str(OUTPUT / "fits")) for k in KS for seed in SEEDS for penalty in PENALTIES]
    completed, failures = 0, []
    # Submit at most one fit per worker. At the wall-clock guard, stop scheduling;
    # preserve the at most two already-running bounded fits before shutting down.
    remaining = iter(jobs)
    deadline_reached = False
    with ProcessPoolExecutor(max_workers=contract["workers"]) as pool:
        futures = {pool.submit(execute_one, job): job for job in [next(remaining) for _ in range(contract["workers"])]}
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                try:
                    result = future.result()
                    completed += 1
                    print(json.dumps({"completed": completed, "k": result["k"], "seed": result["seed"],
                                      "penalty": result["penalty"], "status": result["status"],
                                      "iterations": result.get("final", {}).get("iterations"),
                                      "seconds": result["seconds"]}), flush=True)
                except Exception as error:
                    failures.append({"job": job[:3], "error": repr(error)})
                if time.monotonic() - started > contract["wall_clock_cap_seconds"]:
                    if not deadline_reached:
                        failures.append({"error": "wall-clock cap reached; no new fits scheduled; active fits preserved"})
                    deadline_reached = True
                if not deadline_reached:
                    job = next(remaining, None)
                    if job is not None:
                        futures[pool.submit(execute_one, job)] = job
    write_json_new(OUTPUT / "EXECUTION_FINISHED.json", {"finished_utc": utc_now(),
                   "wall_seconds": time.monotonic() - started, "completed": completed,
                   "infrastructure_failures": failures})
    summarize()


def summarize():
    read_contract()
    rows, manifest, missing = [], [], []
    for k in KS:
        for seed in SEEDS:
            for penalty in PENALTIES:
                filename = fit_filename(k, seed, penalty)
                path = OUTPUT / "fits" / filename.replace(".npz", ".json")
                if not path.exists():
                    missing.append({"k": k, "seed": seed, "penalty": penalty})
                    continue
                item = json.loads(path.read_text())
                if sha256(OUTPUT / "fits" / filename) != item["npz_sha256"]:
                    raise RuntimeError(f"Output changed: {filename}")
                manifest.append({"k": k, "seed": seed, "penalty": penalty,
                                 "filename": f"fits/{filename}", "sha256": item["npz_sha256"],
                                 "metadata_filename": f"fits/{path.name}", "metadata_sha256": sha256(path),
                                 "status": item["status"]})
                row = {key: item[key] for key in ("k", "seed", "penalty", "status", "seconds", "filename")}
                for group in ("early", "final", "early_metrics", "final_metrics", "matrix_change", "original_early_comparison"):
                    row.update({f"{group}_{key}": value for key, value in item.get(group, {}).items()})
                rows.append(row)
    summary = []
    for k in KS:
        for penalty in PENALTIES:
            group = [row for row in rows if row["k"] == k and row["penalty"] == penalty]
            valid = [row for row in group if row["status"] != "failed"]
            entry = {"k": k, "penalty": penalty, "expected": 40, "saved": len(group), "valid": len(valid),
                     "failed": sum(row["status"] == "failed" for row in group),
                     "early_capped": sum(not row.get("early_converged", False) for row in valid),
                     "final_capped": sum(row["status"] == "iteration_cap" for row in valid)}
            if len(valid) == 40:
                for key in ("early_iterations", "final_iterations", "matrix_change_rms", "matrix_change_max_abs"):
                    values = np.array([row[key] for row in valid])
                    entry[key] = {"min": float(values.min()), "median": float(np.median(values)),
                                  "mean": float(values.mean()), "max": float(values.max())}
                for metric in ("auc", "probability_rmse", "ard_rmse", "within_u_auc", "zero_adjusted_auc"):
                    for checkpoint in ("early", "final", "change"):
                        before = [row[f"early_metrics_{metric}"] for row in valid]
                        after = [row[f"final_metrics_{metric}"] for row in valid]
                        if any(value is None for value in before + after):
                            entry[f"{checkpoint}_{metric}"] = {"mean": None, "mcse": None, "reason": "undefined cells retained"}
                            continue
                        values = np.asarray(after) - before if checkpoint == "change" else np.asarray(before if checkpoint == "early" else after)
                        entry[f"{checkpoint}_{metric}"] = {"mean": float(values.mean()),
                                                            "mcse": float(values.std(ddof=1) / np.sqrt(40))}
            summary.append(entry)
    comparisons = [row for row in rows if row["penalty"] == 1.0 and row["status"] != "failed"]
    readout = {"generated_utc": utc_now(), "expected_fits": 240, "saved_fits": len(rows), "missing": missing,
               "failed": [row for row in rows if row["status"] == "failed"],
               "final_caps": [{key: row[key] for key in ("k", "seed", "penalty", "final_iterations", "final_primal_residual", "final_dual_residual")} for row in rows if row["status"] == "iteration_cap"],
               "original_lambda1_comparison": {"expected": 80, "compared": len(comparisons),
                   "exact_array_equal": sum(row["original_early_comparison_array_equal"] for row in comparisons),
                   "max_upper_difference": max((row["original_early_comparison_max_abs"] for row in comparisons), default=None)},
               "summary": summary,
               "interpretation": "Fixed-design numerical and lambda sensitivity evidence only; no tuning selection or general convergence guarantee."}
    write_json_new(OUTPUT / "FILES_MANIFEST.json", manifest)
    write_json_new(OUTPUT / "SUMMARY.json", readout)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (OUTPUT / "fits.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"saved": len(rows), "missing": len(missing), "caps": len(readout["final_caps"]),
                      "failed": len(readout["failed"]), "original_lambda1_comparison": readout["original_lambda1_comparison"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "smoke", "run", "summarize"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--global-contract", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.workers)
    elif args.phase == "smoke":
        print(json.dumps(execute_one((4, 101, 1.0, str(OUTPUT / "smoke"))), allow_nan=False), flush=True)
    elif args.phase == "run":
        if args.global_contract is None:
            parser.error("run requires --global-contract")
        run(args.global_contract)
    else:
        summarize()


if __name__ == "__main__":
    main()
