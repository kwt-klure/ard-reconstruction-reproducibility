"""Read-only independent verification of persisted Stage A outputs.

Usage: python verify_stability.py --root STAGE_DIRECTORY --report NEW_JSON

Only the explicitly requested new report is written. No fitting, imports from
the experiment implementations, repairs, exclusions or retries occur. Exit 0
means the saved evidence is internally consistent, not that scientific screens
passed. Scientific flags, explicitly recorded failed fits and iteration caps
are findings. Missing files, changed hashes, inconsistent metadata and numerical
recomputation disagreements are verification errors (exit 1).

Full raw-array means and MCSE identities are checked. Independent literal rank
and folded split R-hat checks cover fixed P indices 0, 1, 37, 389, 778, 779 and
all three scalars in every cell. ESS is NOT independently recomputed here; its
saved range, masks and relationship to raw-scale mean MCSE are checked. ADMM
residuals and first-hit stopping times cannot be reconstructed from matrices
alone; their recorded logical consistency is checked without rerunning fits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np


SEEDS = (101, 202, 303, 404, *range(1001, 1037))
KS = (4, 8)
PENALTIES = (0.25, 1.0, 4.0)
TOL = 2e-12
RHAT_INDICES = (0, 1, 37, 389, 778, 779)
N = 40
UPPER = np.triu_indices(N, 1)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    # JSON NaN/Infinity are invalid evidence, even though Python accepts them.
    def reject(value):
        raise ValueError(f"nonstandard JSON numeric constant: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def close(actual, expected, message, *, atol=TOL, rtol=0.0):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape, f"{message}: shape {a.shape} != {b.shape}")
    require(np.all(np.isfinite(a)) and np.all(np.isfinite(b)),
            f"{message}: nonfinite comparison")
    if a.dtype == bool or b.dtype == bool:
        require(np.array_equal(a, b), f"{message}: boolean mismatch")
        return 0.0
    delta = float(np.max(np.abs(a - b))) if a.size else 0.0
    require(np.allclose(a, b, atol=atol, rtol=rtol), f"{message}: max difference {delta}")
    return delta


def compare_json(actual, expected, message):
    require(set(actual) == set(expected), f"{message}: keys disagree")
    for key, value in expected.items():
        label = f"{message}.{key}"
        if value is None or isinstance(value, (bool, str)):
            require(actual[key] == value, f"{label}: {actual[key]!r} != {value!r}")
        elif isinstance(value, dict):
            compare_json(actual[key], value, label)
        elif isinstance(value, (int, float, np.number)):
            close(actual[key], value, label)
        else:
            require(actual[key] == value, f"{label}: mismatch")


def feasible(matrix, label):
    require(matrix.shape == (N, N), f"{label}: shape")
    require(matrix.dtype == np.float64, f"{label}: not float64")
    require(np.isfinite(matrix).all(), f"{label}: nonfinite")
    require(np.array_equal(matrix, matrix.T), f"{label}: asymmetric")
    require(np.all(np.diag(matrix) == 0), f"{label}: nonzero diagonal")
    require(np.all((matrix >= 0) & (matrix <= 1)), f"{label}: outside [0,1]")


def from_upper(values):
    result = np.zeros((*values.shape[:-1], N, N), dtype=np.float64)
    result[..., UPPER[0], UPPER[1]] = values
    result[..., UPPER[1], UPPER[0]] = values
    return result


def literal_normal_scores(values):
    """Average ranks via literal tie groups, independent of vectorized code."""
    flat = np.asarray(values, dtype=float).reshape(-1)
    order = sorted(range(len(flat)), key=flat.__getitem__)
    scores = np.empty(len(flat))
    normal = NormalDist()
    start = 0
    while start < len(flat):
        end = start + 1
        while end < len(flat) and flat[order[end]] == flat[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2
        score = normal.inv_cdf((average_rank - 0.375) / (len(flat) + 0.25))
        scores[order[start:end]] = score
        start = end
    return scores.reshape(values.shape)


def literal_rhat(draws):
    half = draws.shape[1] // 2
    split = np.array([part for row in draws for part in (row[:half], row[-half:])])
    if not np.isfinite(split).all() or any(np.max(row) == np.min(row) for row in split):
        return math.inf
    folded = np.abs(split - np.median(split))
    if folded.max() == folded.min():
        return math.inf
    estimates = []
    for values in (split, folded):
        z = literal_normal_scores(values)
        means = np.array([np.mean(row) for row in z])
        within = sum(float(np.var(row, ddof=1)) for row in z) / len(z)
        between = half * float(np.var(means, ddof=1))
        estimates.append(math.sqrt(((half - 1) / half * within + between / half) / within))
    return max(estimates)


def safe_float(value):
    return float(value) if np.isfinite(value) else None


def diagnostic_summary(diag, rhat_limit, ess_limit):
    r, e, m = (diag[key] for key in ("rhat", "bulk_ess", "mean_mcse"))
    bad_r = ~np.isfinite(r) | (r >= rhat_limit)
    bad_e = ~np.isfinite(e) | (e < ess_limit)
    return {"variables": len(r), "rhat_max": safe_float(np.max(r)),
            "rhat_q95": safe_float(np.quantile(r, .95)),
            "bulk_ess_min": safe_float(np.min(e)), "bulk_ess_median": safe_float(np.median(e)),
            "rhat_failures": int(bad_r.sum()), "ess_failures": int(bad_e.sum()),
            "flagged_variables": int((bad_r | bad_e).sum()),
            "mean_mcse_rms": safe_float(np.sqrt(np.mean(m ** 2))),
            "mean_mcse_max": safe_float(np.max(m)), "screen_pass": bool(not (bad_r | bad_e).any())}


def verify_diagnostics(draws, archive, prefix, persisted, contract, indices):
    names = ("rhat", "bulk_ess", "mean_ess", "mean_mcse", "nonfinite",
             "constant", "folded_constant", "invalid")
    diag = {name: archive[f"{prefix}_{name}"] for name in names}
    variables = draws.shape[-1]
    for name, values in diag.items():
        require(values.shape == (variables,), f"{prefix}_{name}: shape")
        require(values.dtype == (bool if name in names[4:] else np.float64),
                f"{prefix}_{name}: dtype")
    require(not diag["nonfinite"].any(), f"{prefix}: raw finite but nonfinite mask set")
    half = draws.shape[1] // 2
    constant = ((draws[:, :half].max(axis=1) == draws[:, :half].min(axis=1)) |
                (draws[:, -half:].max(axis=1) == draws[:, -half:].min(axis=1))).any(axis=0)
    require(np.array_equal(diag["constant"], constant), f"{prefix}: constant mask")
    # Folded checks in bounded chunks avoid copying the complete draw array.
    folded_constant = np.zeros(variables, dtype=bool)
    for start in range(0, variables, 32):
        stop = min(start + 32, variables)
        block = np.asarray(draws[:, :, start:stop])
        folded = np.abs(block - np.median(block, axis=(0, 1)))
        folded_constant[start:stop] = ((folded.max(axis=(0, 1)) == folded.min(axis=(0, 1)))
                                       & ~constant[start:stop])
    require(np.array_equal(diag["folded_constant"], folded_constant), f"{prefix}: folded mask")
    invalid = diag["invalid"]
    require(np.all(invalid | ~(constant | folded_constant)), f"{prefix}: unmarked invalid variable")
    for name in names[:4]:
        values = diag[name]
        require(not np.isnan(values).any(), f"{prefix}_{name}: NaN diagnostic")
        if name in ("rhat", "mean_mcse"):
            require(np.isposinf(values[invalid]).all(), f"{prefix}_{name}: invalid sentinel")
        else:
            require(np.all(values[invalid] == 0), f"{prefix}_{name}: invalid sentinel")
        require(np.isfinite(values[~invalid]).all() and np.all(values[~invalid] > 0),
                f"{prefix}_{name}: invalid value passed")
    for name in ("bulk_ess", "mean_ess"):
        require(np.all(diag[name] <= draws.shape[0] * draws.shape[1] *
                       math.log10(draws.shape[0] * draws.shape[1]) + 1e-8),
                f"{prefix}_{name}: beyond fixed ESS cap")
    sd = draws.std(axis=(0, 1), ddof=1)
    mcse_delta = close(diag["mean_mcse"][~invalid],
                       sd[~invalid] / np.sqrt(diag["mean_ess"][~invalid]), f"{prefix}: raw MCSE identity")
    max_rhat_delta = 0.0
    for index in indices:
        reference = literal_rhat(draws[:, :, index])
        saved = diag["rhat"][index]
        if np.isfinite(reference):
            max_rhat_delta = max(max_rhat_delta, close(saved, reference, f"{prefix}: literal R-hat[{index}]"))
        else:
            require(np.isposinf(saved) and invalid[index], f"{prefix}: degenerate R-hat passed")
    with np.errstate(invalid="ignore"):
        expected = diagnostic_summary(diag, contract["diagnostic_rhat_flag"], contract["bulk_ess_flag"])
    compare_json(persisted, expected, f"{prefix}: saved diagnostic summary")
    require(not (expected["screen_pass"] and invalid.any()), f"{prefix}: invalid diagnostic screen passed")
    return {"flagged_variables": expected["flagged_variables"], "invalid_variables": int(invalid.sum()),
            "mcse_identity_max_difference": mcse_delta, "literal_rhat_max_difference": max_rhat_delta}


def expected_initial(seed, kind):
    rng = np.random.default_rng(seed)
    if kind == "iid":
        angles, alpha = rng.uniform(-np.pi, np.pi, N), float(rng.normal(-2., 1.))
    elif kind == "equally_spaced":
        angles, alpha = rng.permutation(np.linspace(-np.pi, np.pi, N, endpoint=False)), -4.
    else:
        centers = rng.permutation(np.repeat([0., np.pi], N // 2))
        angles, alpha = (centers + rng.normal(0., .1, N) + np.pi) % (2 * np.pi) - np.pi, 0.
    return {"seed": seed, "kind": kind, "alpha": alpha, "angles": angles.tolist()}


def verify_circular_cell(stage, contract, inputs, k, seed):
    name = f"run_k{k}_seed{seed}"
    path = stage / "circular" / name
    complete = read_json(path / "complete.json")
    spec = contract["circular"]
    for key, value in {"k": k, "seed": seed, "warmup": 1600, "draws": 3200, "chains": 4,
                       "input_sha256": contract["input_sha256"][name + ".npz"]}.items():
        require(complete[key] == value, f"{name}: complete.{key}")
    expected_files = {"P_draws.npy", "scalar_draws.npy", "summary.npz", "diagnostics.npz",
                      *(f"chain{c}-state.json" for c in range(1, 5))}
    require(set(complete["output_sha256"]) == expected_files, f"{name}: output hash coverage")
    require({p.name for p in path.iterdir()} == expected_files | {"complete.json"}, f"{name}: unexpected/missing file")
    for filename, expected_hash in complete["output_sha256"].items():
        require(sha256(path / filename) == expected_hash, f"{name}/{filename}: hash changed")
    p = np.load(path / "P_draws.npy", mmap_mode="r", allow_pickle=False)
    scalar = np.load(path / "scalar_draws.npy", mmap_mode="r", allow_pickle=False)
    for array, shape, label in ((p, (4, 3200, 780), "P"), (scalar, (4, 3200, 3), "scalar")):
        require(array.shape == shape and array.dtype == np.float64, f"{name}: {label} shape/dtype")
        require(np.isfinite(array).all(), f"{name}: {label} nonfinite raw draws")
    require(np.all((p >= 0) & (p <= 1)), f"{name}: P draws outside [0,1]")
    chain_means = from_upper(p.mean(axis=1))
    calculated = {"chain_mean_P": chain_means, "pooled_P": chain_means.mean(axis=0),
                  "first_half_P": from_upper(p[:, :1600].mean(axis=(0, 1))),
                  "last_half_P": from_upper(p[:, 1600:].mean(axis=(0, 1))),
                  "group12_P": chain_means[:2].mean(axis=0), "group34_P": chain_means[2:].mean(axis=0)}
    max_mean_delta = 0.0
    with np.load(path / "summary.npz", allow_pickle=False) as summary:
        require(set(summary.files) == set(calculated), f"{name}: summary keys")
        for key, expected in calculated.items():
            max_mean_delta = max(max_mean_delta, close(summary[key], expected, f"{name}: {key}"))
    with np.load(inputs / (name + ".npz"), allow_pickle=False) as source:
        y, b = source["Y"], source["B"]
    for chain, (kind, offset) in enumerate(zip(spec["initializations"], spec["seed_offsets"])):
        state = read_json(path / f"chain{chain + 1}-state.json")
        compare_json(state["initial"], expected_initial(seed + offset, kind), f"{name}: chain{chain+1} initialization")
        final = state["final"]
        angles = np.asarray(final["angles"])
        require(angles.shape == (N,) and np.isfinite(angles).all(), f"{name}: final angles")
        close(final["alpha"], scalar[chain, -1, 0], f"{name}: final alpha")
        last_p = 1 / (1 + np.exp(-(float(final["alpha"]) + 1.5 * np.cos(angles[:, None] - angles[None, :]))))
        np.fill_diagonal(last_p, 0)
        close(last_p[UPPER], p[chain, -1], f"{name}: final state probability draw")
        for index in (0, 1599, 1600, 3199):
            rates = from_upper(p[chain, index]) @ b
            ll = np.sum(y * np.log(np.maximum(rates, 1e-300)) - rates)
            close(scalar[chain, index, 1:], [ll, np.mean((rates - y) ** 2)], f"{name}: scalar trace check")
        acceptance, steps = np.asarray(state["acceptance"]), np.asarray(state["steps"])
        require(acceptance.shape == (2,) and np.isfinite(acceptance).all() and
                np.all((acceptance >= 0) & (acceptance <= 1)), f"{name}: acceptance rates")
        require(steps.shape == (2,) and np.isfinite(steps).all() and np.all((steps >= .02) & (steps <= 3)),
                f"{name}: proposal steps")
        rng = np.random.PCG64()
        rng.state = final["rng_state"]  # Validate the persisted state without drawing.
    with np.load(path / "diagnostics.npz", allow_pickle=False) as archive:
        p_diag = verify_diagnostics(p, archive, "P", complete["matrix_diagnostics"], spec, RHAT_INDICES)
        s_diag = verify_diagnostics(scalar, archive, "scalar", complete["scalar_diagnostics"], spec, range(3))
    return {"cell": name, "chains_verified": 4, "mean_max_difference": max_mean_delta,
            "P": p_diag, "scalar": s_diag, "complete_sha256": sha256(path / "complete.json")}


def rank_auc(labels, scores):
    """Mann-Whitney average-rank AUC, independent of pairwise comparison code."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    positives, negatives = int(np.sum(labels == 1)), int(np.sum(labels == 0))
    if not positives or not negatives:
        return None
    require(positives + negatives == len(labels), "AUC labels not binary")
    order = np.argsort(scores, kind="stable")
    rank_sum = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        rank_sum += ((start + 1 + end) / 2) * np.sum(labels[order[start:end]] == 1)
        start = end
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def evaluation_metrics(p, y, b, adjacency, truth):
    groups = np.argmax(b, axis=1)
    possible = np.array([y[i, groups[j]] > 0 and y[j, groups[i]] > 0 for i, j in zip(*UPPER)])
    labels, scores = adjacency[UPPER], p[UPPER]
    return {"auc": rank_auc(labels, scores),
            "probability_rmse": float(np.sqrt(np.mean(np.square(scores - truth[UPPER])))),
            "ard_rmse": float(np.sqrt(np.mean(np.square(p @ b - y)))),
            "within_u_auc": rank_auc(labels[possible], scores[possible]),
            "zero_adjusted_auc": rank_auc(labels, scores * possible),
            "u_edges": int(np.sum(labels[possible] == 1)), "u_nonedges": int(np.sum(labels[possible] == 0))}


def matrix_difference(before, after):
    delta = (after - before)[UPPER]
    return {"rms": float(np.sqrt(np.mean(delta ** 2))), "max_abs": float(np.max(np.abs(delta)))}


def verify_nuclear_fit(stage, contract, nuclear_contract, inputs, k, seed, penalty):
    name = f"run_k{k}_seed{seed}_lambda{penalty:g}"
    path = stage / "nuclear" / "fits" / name
    meta = read_json(Path(str(path) + ".json"))
    # Path.with_suffix would interpret the decimal in lambda0.25 as a suffix.
    npz_path = Path(str(path) + ".npz")
    require(sha256(npz_path) == meta["npz_sha256"], f"{name}: NPZ hash")
    expectations = {"k": k, "seed": seed, "penalty": penalty, "filename": name + ".npz",
                    "source_input_sha256": contract["input_sha256"][f"run_k{k}_seed{seed}.npz"],
                    "source_immutable_method_sha256": contract["original_runner_sha256"],
                    "solver_sha256": contract["code_sha256"]["nuclear.py"],
                    "contract_sha256": contract["nuclear"]["own_contract_sha256"]}
    for key, value in expectations.items():
        require(meta[key] == value, f"{name}: metadata {key}")
    require(meta["status"] in ("converged", "iteration_cap", "failed"), f"{name}: unknown fit status")
    with np.load(npz_path, allow_pickle=False) as arrays:
        embedded = json.loads(str(arrays["metadata_json"].item()))
        require(embedded == {key: value for key, value in meta.items() if key != "npz_sha256"}, f"{name}: embedded metadata")
        early, final = arrays["early_P"], arrays["final_P"]
        if meta["status"] == "failed":
            require(bool(meta.get("error")), f"{name}: failed without error")
            require(early.shape == final.shape == (N, N) and np.isnan(early).all() and np.isnan(final).all(),
                    f"{name}: failed-fit sentinel")
            return {"fit": name, "status": "failed", "error": meta["error"]}
        feasible(early, name + ":early")
        feasible(final, name + ":final")
        with np.load(inputs / f"run_k{k}_seed{seed}.npz", allow_pickle=False) as source:
            y, b, adjacency, truth = source["Y"], source["B"], source["hidden_A"], source["true_P"]
            for label, matrix, tol, cap in (("early", early, nuclear_contract["early_tolerance"], nuclear_contract["early_cap"]),
                                            ("final", final, nuclear_contract["tolerance"], nuclear_contract["max_iterations"])):
                checkpoint = meta[label]
                iterations = checkpoint["iterations"]
                require(type(iterations) is int and 1 <= iterations <= cap, f"{name}: {label} iterations")
                primal, dual = checkpoint["primal_residual"], checkpoint["dual_residual"]
                require(np.isfinite([primal, dual]).all() and min(primal, dual) >= 0, f"{name}: {label} residuals")
                converged = bool(primal < tol and dual < tol)
                require(checkpoint["converged"] is converged, f"{name}: {label} residual flag")
                require(converged or iterations == cap, f"{name}: {label} premature stopping")
                for key, value in checkpoint.items():
                    require(arrays[f"{label}_{key}"].shape == (), f"{name}: checkpoint scalar shape")
                    close(arrays[f"{label}_{key}"], value, f"{name}: duplicated {label}_{key}")
                # For a real symmetric matrix, nuclear norm equals sum |eigenvalue|.
                objective = .5 * np.sum((matrix @ b - y) ** 2) + penalty * np.abs(np.linalg.eigvalsh(matrix)).sum()
                close(checkpoint["objective"], objective, f"{name}: {label} objective", atol=2e-10, rtol=2e-12)
                compare_json(meta[f"{label}_metrics"], evaluation_metrics(matrix, y, b, adjacency, truth), f"{name}: {label} metrics")
            require(meta["early"]["iterations"] <= meta["final"]["iterations"], f"{name}: checkpoint order")
            require(meta["status"] == ("converged" if meta["final"]["converged"] else "iteration_cap"), f"{name}: status flag")
            compare_json(meta["matrix_change"], matrix_difference(early, final), f"{name}: matrix change")
            if penalty == 1:
                old = source["nuclear_P"]
                comparison = {"array_equal": bool(np.array_equal(old, early)), **matrix_difference(old, early)}
                compare_json(meta["original_early_comparison"], comparison, f"{name}: baseline comparison")
                require(comparison["array_equal"], f"{name}: lambda1 early matrix differs from original baseline")
    return {"fit": name, "status": meta["status"], "early_capped": not meta["early"]["converged"],
            "final_capped": not meta["final"]["converged"], "npz_sha256": meta["npz_sha256"]}


def verify(stage):
    stage = Path(stage).resolve()
    report = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
              "root": str(stage), "verifier_sha256": sha256(__file__), "errors": [], "findings": [],
              "inputs_verified": 0, "circular": [], "nuclear": [],
              "scope": {"expected_inputs": 80, "expected_circular_cells": 80, "expected_chains": 320,
                        "expected_nuclear_fits": 240, "raw_mean_absolute_tolerance": TOL,
                        "literal_rhat_P_indices": list(RHAT_INDICES), "literal_rhat_scalar_indices": [0, 1, 2],
                        "independent_ess_recalculation": False, "ADMM_trajectory_replay": False}}

    def attempt(label, function):
        try:
            return function()
        except (OSError, ValueError, KeyError, TypeError, IndexError, AssertionError) as error:
            report["errors"].append({"scope": label, "error": f"{type(error).__name__}: {error}"})
            return None

    contract = attempt("global contract", lambda: read_json(stage / "RUN_CONTRACT.json"))
    if contract is None:
        return finish_report(report)
    report["contract_sha256"] = sha256(stage / "RUN_CONTRACT.json")
    # Resolve input paths against the actual repository, independent of cwd.
    repo = Path(__file__).resolve().parents[3]
    inputs = repo / contract["inputs_relative"]

    def check_contract():
        require(contract["n"] == N and contract["k"] == list(KS) and contract["seeds"] == list(SEEDS), "fixed80 design")
        require(set(contract["input_sha256"]) == {f"run_k{k}_seed{s}.npz" for k in KS for s in SEEDS}, "input hash membership")
        required = {"chains": 4, "warmup": 1600, "draws": 3200, "zeta": 1.5,
                    "initializations": ["iid", "iid", "equally_spaced", "two_clusters"],
                    "seed_offsets": [10000, 20000, 30000, 40000],
                    "diagnostic_rhat_flag": 1.01, "bulk_ess_flag": 400}
        for key, value in required.items():
            require(contract["circular"][key] == value, f"contract circular.{key}")
        require(contract["nuclear"]["lambda"] == list(PENALTIES) and contract["nuclear"]["rho"] == 2 and
                contract["nuclear"]["tolerance"] == 1e-7 and contract["nuclear"]["iteration_cap"] == 50000,
                "contract nuclear settings")
        for filename, expected in contract["code_sha256"].items():
            require(sha256(stage / filename) == expected, f"frozen code hash: {filename}")
        require(sha256(stage.parent / "social-networks-fixed40/run_experiment.py") == contract["original_runner_sha256"], "original runner hash")

    attempt("contract and code", check_contract)
    for filename, expected in contract["input_sha256"].items():
        result = attempt("input " + filename, lambda f=filename, h=expected: require(sha256(inputs / f) == h, "input hash mismatch") or True)
        report["inputs_verified"] += int(result is True)
    for k in KS:
        for seed in SEEDS:
            result = attempt(f"circular k{k} seed{seed}", lambda k=k, s=seed: verify_circular_cell(stage, contract, inputs, k, s))
            if result is not None:
                report["circular"].append(result)
                if result["P"]["flagged_variables"] or result["scalar"]["flagged_variables"]:
                    report["findings"].append({"scope": result["cell"], "kind": "MCMC_screen_flags",
                                               "P": result["P"]["flagged_variables"], "scalar": result["scalar"]["flagged_variables"]})

    def check_circular_index():
        require({p.name for p in (stage / "circular").iterdir() if p.is_dir()} ==
                {f"run_k{k}_seed{s}" for k in KS for s in SEEDS}, "circular directory membership")
        index = read_json(stage / "circular/results.json")
        require(index["status"] == "complete" and index["contract_sha256"] == report["contract_sha256"], "circular results status/contract")
        require(index["code_sha256"] == contract["code_sha256"]["circular.py"], "circular results code hash")
        runs = index["runs"]
        require([(run["k"], run["seed"]) for run in runs] == [(k, s) for k in KS for s in SEEDS], "circular results membership/order")
        for run in runs:
            require(run == read_json(stage / f"circular/run_k{run['k']}_seed{run['seed']}/complete.json"), "circular results record changed")
    attempt("circular results index", check_circular_index)
    nuclear_contract = attempt("nuclear contract", lambda: read_json(stage / "nuclear/RUN_CONTRACT.json"))
    if nuclear_contract is not None:
        def check_nuclear_contract():
            require(sha256(stage / "nuclear/RUN_CONTRACT.json") == contract["nuclear"]["own_contract_sha256"], "nuclear contract hash")
            require(nuclear_contract["input_sha256"] == contract["input_sha256"], "nuclear input hashes")
            for key, value in {"n": N, "ks": list(KS), "seeds": list(SEEDS), "penalties": list(PENALTIES),
                               "fits": 240, "rho": 2.0, "max_iterations": 50000, "tolerance": 1e-7,
                               "early_tolerance": 1e-6, "early_cap": 10000}.items():
                require(nuclear_contract[key] == value, f"nuclear contract {key}")
        attempt("nuclear contract settings", check_nuclear_contract)
        for k in KS:
            for seed in SEEDS:
                for penalty in PENALTIES:
                    result = attempt(f"nuclear k{k} seed{seed} lambda{penalty:g}",
                                     lambda k=k, s=seed, p=penalty: verify_nuclear_fit(stage, contract, nuclear_contract, inputs, k, s, p))
                    if result is not None:
                        report["nuclear"].append(result)
                        if result["status"] != "converged" or result.get("early_capped"):
                            report["findings"].append({"kind": "ADMM_numerical_warning", **result})
        attempt("nuclear manifest", lambda: verify_nuclear_manifest(stage))
    return finish_report(report)


def verify_nuclear_manifest(stage):
    manifest = read_json(stage / "nuclear/FILES_MANIFEST.json")
    expected = {(k, s, p) for k in KS for s in SEEDS for p in PENALTIES}
    require(len(manifest) == 240 and {(r["k"], r["seed"], r["penalty"]) for r in manifest} == expected, "nuclear manifest membership")
    require({p.name for p in (stage / "nuclear/fits").iterdir()} ==
            {f"run_k{k}_seed{s}_lambda{p:g}.{extension}" for k, s, p in expected for extension in ("json", "npz")},
            "nuclear fit-directory membership")
    for item in manifest:
        filename = f"fits/run_k{item['k']}_seed{item['seed']}_lambda{item['penalty']:g}"
        require(item["filename"] == filename + ".npz" and item["metadata_filename"] == filename + ".json", "nuclear manifest path")
        for name, hash_name in (("filename", "sha256"), ("metadata_filename", "metadata_sha256")):
            require(sha256(stage / "nuclear" / item[name]) == item[hash_name], f"nuclear manifest hash: {item[name]}")
        require(read_json(stage / "nuclear" / item["metadata_filename"])["status"] == item["status"], "manifest status")
    finished = read_json(stage / "nuclear/EXECUTION_FINISHED.json")
    require(finished["completed"] == 240 and not finished["infrastructure_failures"], "nuclear execution incomplete")


def finish_report(report):
    report["circular_cells_verified"] = len(report["circular"])
    report["chains_verified"] = sum(row["chains_verified"] for row in report["circular"])
    report["nuclear_fits_verified"] = len(report["nuclear"])
    report["verification_pass"] = not report["errors"] and (report["inputs_verified"], report["circular_cells_verified"], report["chains_verified"], report["nuclear_fits_verified"]) == (80, 80, 320, 240)
    report["scientific_warnings_present"] = bool(report["findings"])
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["interpretation"] = "Verification pass means consistent saved evidence only; scientific screen warnings are retained and do not imply verifier failure."
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists() or args.report.is_symlink():
        parser.error("Refusing to overwrite an existing report")
    if not args.report.parent.is_dir():
        parser.error("Report parent directory must already exist")
    report = verify(args.root)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"report": str(args.report.resolve()), "verification_pass": report["verification_pass"],
                      "inputs": report["inputs_verified"], "circular_cells": report["circular_cells_verified"],
                      "chains": report["chains_verified"], "nuclear_fits": report["nuclear_fits_verified"],
                      "errors": len(report["errors"]), "findings": len(report["findings"])}))
    return 0 if report["verification_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
