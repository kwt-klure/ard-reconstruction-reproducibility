"""Matched-input ARD reconstruction experiment; NumPy-only, no hidden-edge fitting."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

N = 40
KS = (4, 8)
SEEDS = (101, 202, 303, 404, *range(1001, 1037))
ZETA = 1.5
ALPHA = -2.8
PENALTY = 1.0


def probabilities(alpha, angles, zeta=ZETA):
    logits = alpha + zeta * np.cos(angles[:, None] - angles[None, :])
    p = 1.0 / (1.0 + np.exp(-logits))
    np.fill_diagonal(p, 0.0)
    return p


def generate(seed, k, n=N):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-np.pi, np.pi, n)
    p = probabilities(ALPHA, angles)
    a = np.triu((rng.random((n, n)) < p).astype(float), 1)
    a += a.T
    # Deterministic balanced groups independent of latent positions and edges.
    b = np.eye(k)[np.arange(n) % k]
    y = a @ b
    return y, b, a, p


def validate_inputs(y, b):
    n, k = b.shape
    assert y.shape == (n, k)
    assert np.all((b == 0) | (b == 1)) and np.all(b.sum(axis=1) == 1)
    assert np.all(y >= 0) and np.all(y == np.floor(y))
    assert np.all(y <= b.sum(axis=0)[None, :] - b)
    assert np.allclose(b.T @ y, (b.T @ y).T)


def poisson_loglik(y, rates):
    # Omit log(y!), constant in all parameter updates; no Bernoulli-edge inputs.
    return float(np.sum(y * np.log(np.maximum(rates, 1e-300)) - rates))


def sample_chain(y, b, seed, warmup, draws, zeta=ZETA):
    rng = np.random.default_rng(seed)
    n = len(b)
    angles = rng.uniform(-np.pi, np.pi, n)
    alpha = float(rng.normal(-2.0, 1.0))
    p = probabilities(alpha, angles, zeta)
    ll = poisson_loglik(y, p @ b)
    # alpha ~ Normal(-2, 2^2); independent uniform circular position priors.
    prior = lambda x: -0.5 * ((x + 2.0) / 2.0) ** 2
    steps = np.array([0.2, 0.7])
    accepts = np.zeros(2)
    window_accepts = np.zeros(2)
    saved = np.zeros((draws, 3))
    mean_p = np.zeros_like(p)
    for iteration in range(warmup + draws):
        proposal_alpha = alpha + rng.normal(0, steps[0])
        proposal_p = probabilities(proposal_alpha, angles, zeta)
        proposal_ll = poisson_loglik(y, proposal_p @ b)
        if np.log(rng.random()) < proposal_ll - ll + prior(proposal_alpha) - prior(alpha):
            alpha, p, ll = proposal_alpha, proposal_p, proposal_ll
            window_accepts[0] += 1
            if iteration >= warmup:
                accepts[0] += 1
        for node in rng.permutation(n):
            proposal_angles = angles.copy()
            proposal_angles[node] = (angles[node] + rng.normal(0, steps[1]) + np.pi) % (2 * np.pi) - np.pi
            proposal_p = probabilities(alpha, proposal_angles, zeta)
            proposal_ll = poisson_loglik(y, proposal_p @ b)
            if np.log(rng.random()) < proposal_ll - ll:
                angles, p, ll = proposal_angles, proposal_p, proposal_ll
                window_accepts[1] += 1
                if iteration >= warmup:
                    accepts[1] += 1
        if iteration < warmup and (iteration + 1) % 50 == 0:
            rates = window_accepts / np.array([50, 50 * n])
            steps *= np.exp(np.clip(rates - np.array([0.35, 0.35]), -0.3, 0.3))
            steps = np.clip(steps, 0.02, 3.0)
            window_accepts[:] = 0
        if iteration >= warmup:
            index = iteration - warmup
            mean_p += p / draws
            saved[index] = alpha, ll, np.mean((p @ b - y) ** 2)
    return mean_p, saved, accepts / np.array([draws, draws * n]), steps


def diagnostics(chains):
    # Classic split R-hat, and initial-positive-sequence ESS; not rank-normalized.
    m, draws = chains.shape
    half = draws // 2
    split = np.concatenate([chains[:, :half], chains[:, -half:]], axis=0)
    within = np.mean(np.var(split, axis=1, ddof=1))
    between = half * np.var(split.mean(axis=1), ddof=1)
    variance = (half - 1) / half * within + between / half
    rhat = float(np.sqrt(variance / within)) if within > 0 else float("inf")
    centered = split - split.mean(axis=1, keepdims=True)
    acov = np.array([np.mean(np.sum(centered[:, :half - lag] * centered[:, lag:], axis=1) / half) for lag in range(half)])
    rho = 1.0 - (within - acov) / max(variance, 1e-300)
    rho[0] = 1.0
    pairs = []
    for lag in range(0, half - 1, 2):
        pair = rho[lag] + rho[lag + 1]
        if pair < 0:
            break
        pairs.append(min(pair, pairs[-1]) if pairs else pair)
    tau = max(-1.0 + 2.0 * sum(pairs), 1.0)
    ess = float(min(m * 2 * half / tau, m * 2 * half))
    return {"split_rhat": rhat, "ess_initial_positive": ess}


def fit_bayes(y, b, seed, warmup=400, draws=800, zeta=ZETA):
    validate_inputs(y, b)
    fits = [sample_chain(y, b, seed + offset, warmup, draws, zeta) for offset in (10000, 20000)]
    traces = np.stack([fit[1] for fit in fits])
    diag = {name: diagnostics(traces[:, :, j]) for j, name in enumerate(("alpha", "poisson_loglik", "ard_mse"))}
    diag["screen_pass"] = bool(all(v["split_rhat"] < 1.05 and v["ess_initial_positive"] >= 100 for v in diag.values()))
    diag["acceptance_alpha_angle"] = [fit[2].tolist() for fit in fits]
    diag["final_proposal_scales"] = [fit[3].tolist() for fit in fits]
    return np.mean([fit[0] for fit in fits], axis=0), diag, traces


def project_constraints(x):
    result = np.clip((x + x.T) / 2, 0, 1)
    np.fill_diagonal(result, 0)
    return result


def fit_nuclear(y, b, penalty=PENALTY, rho=2.0, max_iterations=10000, tolerance=1e-6):
    """Consensus ADMM: min .5 ||PB-Y||_F^2 + penalty ||P||_*, P in C."""
    validate_inputs(y, b)
    n = len(b)
    inverse = np.linalg.inv(b @ b.T + 2 * rho * np.eye(n))
    data_term = y @ b.T
    p = np.zeros((n, n))
    lowrank, constrained, u, v = [np.zeros_like(p) for _ in range(4)]
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
        if primal < tolerance and dual < tolerance:
            converged = True
            break
    objective = float(0.5 * np.sum((constrained @ b - y) ** 2) + penalty * np.linalg.svd(constrained, compute_uv=False).sum())
    return constrained, {"iterations": iteration, "converged": converged, "primal_residual": primal, "dual_residual": dual, "objective": objective}


def fit_density(y, b):
    validate_inputs(y, b)
    n = len(b)
    p = np.full((n, n), y.sum() / (n * (n - 1)))
    np.fill_diagonal(p, 0)
    return p


def auc(labels, scores):
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    differences = positive[:, None] - negative[None, :]
    return float(np.mean((differences > 0) + 0.5 * (differences == 0)))


def metrics(estimate, y, b, a, truth):
    upper = np.triu_indices(len(b), 1)
    return {"auc": auc(a[upper], estimate[upper]), "probability_rmse": float(np.sqrt(np.mean((estimate[upper] - truth[upper]) ** 2))), "ard_rmse": float(np.sqrt(np.mean((estimate @ b - y) ** 2)))}


def assert_estimate(p):
    assert np.allclose(p, p.T, atol=1e-10)
    assert np.all(np.diag(p) == 0)
    assert np.all((p >= 0) & (p <= 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--draws", type=int, default=800)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records, diagnostic_records = [], []
    started = time.perf_counter()
    for k in KS:
        for seed in SEEDS:
            y, b, a, truth = generate(seed, k)
            validate_inputs(y, b)
            assert np.array_equal(y.sum(axis=1), a.sum(axis=1))
            assert y.sum() == 2 * np.triu(a, 1).sum()
            # Only these two observed arrays cross the estimator input boundary.
            start = time.perf_counter()
            bayes, bayes_diag, traces = fit_bayes(y, b, seed, args.warmup, args.draws)
            bayes_time = time.perf_counter() - start
            start = time.perf_counter()
            nuclear, nuclear_diag = fit_nuclear(y, b)
            nuclear_time = time.perf_counter() - start
            start = time.perf_counter()
            density = fit_density(y, b)
            density_time = time.perf_counter() - start
            for model, estimate, seconds in (("Bayes_circle", bayes, bayes_time), ("Nuclear_probability", nuclear, nuclear_time), ("Density", density, density_time)):
                assert_estimate(estimate)
                row = {"n": N, "k": k, "seed": seed, "model": model, **metrics(estimate, y, b, a, truth), "seconds": seconds}
                records.append(row)
                print(json.dumps(row), flush=True)
            diagnostic_records.append({"k": k, "seed": seed, "bayes": bayes_diag, "nuclear": nuclear_diag})
            np.savez_compressed(args.output / f"run_k{k}_seed{seed}.npz", Y=y, B=b, hidden_A=a, true_P=truth, bayes_P=bayes, nuclear_P=nuclear, density_P=density, bayes_traces=traces)
    with (args.output / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = []
    for k in KS:
        for model in ("Bayes_circle", "Nuclear_probability", "Density"):
            group = [r for r in records if r["k"] == k and r["model"] == model]
            row = {"k": k, "model": model, "replicates": len(group)}
            for metric in ("auc", "probability_rmse", "ard_rmse", "seconds"):
                values = [r[metric] for r in group]
                row[metric + "_mean"] = float(np.mean(values))
                row[metric + "_sd"] = float(np.std(values, ddof=1))
            summary.append(row)
    with (args.output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    metadata = {"n": N, "k": KS, "seeds": SEEDS, "dgp_alpha": ALPHA, "fixed_zeta": ZETA, "nuclear_penalty": PENALTY, "chains": 2, "warmup_per_chain": args.warmup, "kept_per_chain": args.draws, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}, "total_seconds": time.perf_counter() - started}
    (args.output / "results.json").write_text(json.dumps({"metadata": metadata, "runs": records, "summary": summary, "diagnostics": diagnostic_records}, indent=2) + "\n")
    labels = {"Bayes_circle": "Circular Bayes", "Nuclear_probability": "Nuclear-norm comparator", "Density": "Constant density"}
    tex = [r"\begin{tabular}{rlrrr}", r"\hline", r"$K$ & Method & AUC & Probability RMSE & ARD RMSE \\", r"\hline"]
    for row in summary:
        cells = [f"{row[metric + '_mean']:.3f} ({row[metric + '_sd']:.3f})" for metric in ("auc", "probability_rmse", "ard_rmse")]
        tex.append(f"{row['k']} & {labels[row['model']]} & " + " & ".join(cells) + r" \\")
    tex += [r"\hline", r"\end{tabular}", "% Entries are mean (sample SD), 40 graph seeds; not standard errors."]
    (args.output / "results_table.tex").write_text("\n".join(tex) + "\n")


if __name__ == "__main__":
    main()
