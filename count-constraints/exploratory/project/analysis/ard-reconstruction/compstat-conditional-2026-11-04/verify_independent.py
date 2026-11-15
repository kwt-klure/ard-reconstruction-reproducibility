#!/usr/bin/env python3
"""Independent, bounded verification of the saved K=8 conditional benchmarks.

Cross-group blocks use memoized remaining-column-margin dynamic programming.
For each edge a separate constrained partition calculation supplies an exact
Python-integer count and a floating log partition. No matrices from the primary
enumerator, nor any primary-solver functions, are read. Within-group blocks are
exhausted as labelled undirected edge bitmasks using Bernoulli probabilities.

The only persisted output of this script is an exclusive-create JSON report.
It never generates data, fits an estimator, or changes existing result files.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import unittest

import numpy as np


MARGINAL_TOL = 2e-11
CONSTRAINT_TOL = 2e-10


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logadd(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    high, low = max(left, right), min(left, right)
    return high + math.log1p(math.exp(low - high))


def integer_vector(values, name: str) -> tuple[int, ...]:
    values = np.asarray(values)
    require(values.ndim == 1 and np.all(np.isfinite(values)),
            f"{name} must be a finite vector")
    require(np.array_equal(values, np.rint(values)), f"{name} must be integral")
    return tuple(int(value) for value in values)


def cross_partition(rows, columns, log_odds, fixed=None):
    """Return (exact count, log weighted partition, cached-state count).

    A state is (next row, remaining column degrees). The row's bitmask is
    selected by population count; this does not enumerate complete matrices.
    ``fixed`` is optionally (row, column, bit), and is applied while traversing
    that row. Zero-count branches have log partition minus infinity.
    """
    rows = integer_vector(rows, "row margins")
    columns = integer_vector(columns, "column margins")
    m, n = len(rows), len(columns)
    logs = np.asarray(log_odds, dtype=float)
    require(logs.shape == (m, n) and np.all(np.isfinite(logs)),
            "finite log odds must match block shape")
    require(all(0 <= r <= n for r in rows), "invalid row margins")
    require(all(0 <= c <= m for c in columns), "invalid column margins")
    require(sum(rows) == sum(columns), "unequal margin totals")
    if fixed is not None:
        fi, fj, value = fixed
        require(0 <= fi < m and 0 <= fj < n and value in (0, 1),
                "invalid edge constraint")

    patterns = []
    for i, margin in enumerate(rows):
        options = []
        for mask in range(1 << n):
            if mask.bit_count() != margin:
                continue
            if fixed is not None and i == fixed[0]:
                if ((mask >> fixed[1]) & 1) != fixed[2]:
                    continue
            bits = tuple((mask >> j) & 1 for j in range(n))
            log_weight = math.fsum(logs[i, j] for j in range(n) if bits[j])
            options.append((bits, log_weight))
        patterns.append(options)

    tails = tuple(sum(rows[i:]) for i in range(m + 1))

    @lru_cache(maxsize=None)
    def visit(i, remaining):
        if sum(remaining) != tails[i] or any(c < 0 or c > m - i for c in remaining):
            return 0, -math.inf
        if i == m:
            return 1, 0.0
        count, log_partition = 0, -math.inf
        for bits, log_weight in patterns[i]:
            next_remaining = tuple(c - bit for c, bit in zip(remaining, bits))
            child_count, child_log = visit(i + 1, next_remaining)
            if child_count:
                count += child_count
                log_partition = logadd(log_partition, child_log + log_weight)
        return count, log_partition

    count, log_partition = visit(0, columns)
    return count, log_partition, visit.cache_info().currsize


def cross_marginals(rows, columns, probability):
    probability = np.asarray(probability, dtype=float)
    require(np.all((probability > 0) & (probability < 1)),
            "oracle-support identity requires strictly interior block probabilities")
    logs = np.log(probability) - np.log1p(-probability)
    count, log_partition, states = cross_partition(rows, columns, logs)
    require(count > 0 and math.isfinite(log_partition), "empty cross-group fibre")
    q_uniform = np.zeros_like(probability)
    q_oracle = np.zeros_like(probability)
    forced_zero = np.zeros_like(probability, dtype=bool)
    forced_one = np.zeros_like(probability, dtype=bool)
    incidence_counts = []
    for i in range(probability.shape[0]):
        incidence_row = []
        for j in range(probability.shape[1]):
            count_one, log_one, used = cross_partition(rows, columns, logs, (i, j, 1))
            states += used
            count_zero, log_zero, used = cross_partition(rows, columns, logs, (i, j, 0))
            states += used
            require(count_one + count_zero == count, "edge fibres do not partition support")
            require((count_one > 0) == math.isfinite(log_one),
                    "edge-one support differs between count and weighted partition")
            require((count_zero > 0) == math.isfinite(log_zero),
                    "edge-zero support differs between count and weighted partition")
            require(abs(logadd(log_one, log_zero) - log_partition) < MARGINAL_TOL,
                    "edge-fixed log partitions do not sum to total")
            q_uniform[i, j] = count_one / count
            forced_zero[i, j] = count_one == 0
            forced_one[i, j] = count_zero == 0
            q_oracle[i, j] = (0.0 if count_one == 0 else 1.0 if count_zero == 0
                              else math.exp(log_one - log_partition))
            incidence_row.append(count_one)
        incidence_counts.append(incidence_row)
    return {
        "q_uniform": q_uniform, "q_oracle": q_oracle,
        "forced_zero": forced_zero, "forced_one": forced_one,
        "support_count": count, "edge_one_counts": incidence_counts,
        "log_partition_odds": log_partition, "dp_states_total": states,
    }


def within_marginals(degrees, probability):
    """Exhaust undirected edge masks; evaluate complete Bernoulli likelihoods."""
    degrees = integer_vector(degrees, "within-group degrees")
    n = len(degrees)
    probability = np.asarray(probability, dtype=float)
    require(probability.shape == (n, n), "within-group probability shape")
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    p_edges = np.asarray([probability[i, j] for i, j in edges])
    require(np.all((p_edges > 0) & (p_edges < 1)),
            "oracle-support identity requires strictly interior edge probabilities")
    log_one = np.log(p_edges)
    log_zero = np.log1p(-p_edges)
    count = 0
    edge_counts = [0] * len(edges)
    log_partition = -math.inf
    log_numerators = [-math.inf] * len(edges)
    for mask in range(1 << len(edges)):
        trial_degrees = [0] * n
        for e, (i, j) in enumerate(edges):
            if (mask >> e) & 1:
                trial_degrees[i] += 1
                trial_degrees[j] += 1
        if tuple(trial_degrees) != degrees:
            continue
        count += 1
        log_weight = math.fsum(log_one[e] if (mask >> e) & 1 else log_zero[e]
                               for e in range(len(edges)))
        log_partition = logadd(log_partition, log_weight)
        for e in range(len(edges)):
            if (mask >> e) & 1:
                edge_counts[e] += 1
                log_numerators[e] = logadd(log_numerators[e], log_weight)
    require(count > 0 and math.isfinite(log_partition), "empty within-group fibre")
    q_uniform = np.zeros((n, n))
    q_oracle = np.zeros((n, n))
    forced_zero = np.zeros((n, n), dtype=bool)
    forced_one = np.zeros((n, n), dtype=bool)
    incidence_counts = [[0] * n for _ in range(n)]
    for e, (i, j) in enumerate(edges):
        incidence_counts[i][j] = incidence_counts[j][i] = edge_counts[e]
        require((edge_counts[e] > 0) == math.isfinite(log_numerators[e]),
                "within-edge support differs between count and weighted partition")
        q_uniform[i, j] = q_uniform[j, i] = edge_counts[e] / count
        oracle = (0.0 if edge_counts[e] == 0 else 1.0 if edge_counts[e] == count
                  else math.exp(log_numerators[e] - log_partition))
        q_oracle[i, j] = q_oracle[j, i] = oracle
        forced_zero[i, j] = forced_zero[j, i] = edge_counts[e] == 0
        forced_one[i, j] = forced_one[j, i] = edge_counts[e] == count
    return {
        "q_uniform": q_uniform, "q_oracle": q_oracle,
        "forced_zero": forced_zero, "forced_one": forced_one,
        "support_count": count, "edge_one_counts": incidence_counts,
        "log_partition_bernoulli": log_partition, "masks_examined": 1 << len(edges),
    }


def assert_graph_inputs(data):
    required = {"Y", "B", "hidden_A", "true_P", "bayes_P", "nuclear_P"}
    require(required.issubset(data.files), f"missing source fields: {required - set(data.files)}")
    y, b, a, p = (np.asarray(data[name]) for name in ("Y", "B", "hidden_A", "true_P"))
    require(y.shape == (40, 8) and b.shape == (40, 8), "scope requires n=40, K=8")
    require(a.shape == p.shape == (40, 40), "source matrix dimensions")
    require(np.all(np.isfinite(y)) and np.all(np.isfinite(p)), "source has nonfinite values")
    require(np.all((b == 0) | (b == 1)) and np.all(np.sum(b, axis=1) == 1),
            "B must be a one-hot, full-response partition")
    require(np.all(np.sum(b, axis=0) == 5), "scope requires balanced group size 5")
    require(np.all((a == 0) | (a == 1)) and np.array_equal(a, a.T), "A must be simple binary symmetric")
    require(np.all(np.diag(a) == 0) and np.all(np.diag(p) == 0), "source diagonal must be zero")
    require(np.array_equal(a @ b, y), "saved A is not in F(Y,B)")
    require(np.array_equal(p, p.T), "true_P is not symmetric")
    offdiag = ~np.eye(40, dtype=bool)
    require(np.all((p[offdiag] > 0) & (p[offdiag] < 1)),
            "strictly interior true_P is required for identical uniform/oracle support")
    return y, b, a, p


def matrix_checks(matrix, b, y, name):
    require(matrix.shape == (40, 40) and np.all(np.isfinite(matrix)), f"{name}: shape/nonfinite")
    errors = {
        "symmetry": float(np.max(np.abs(matrix - matrix.T))),
        "diagonal": float(np.max(np.abs(np.diag(matrix)))),
        "constraint_QB_minus_Y": float(np.max(np.abs(matrix @ b - y))),
        "range_violation": float(max(0.0, -float(matrix.min()), float(matrix.max()) - 1.0)),
    }
    require(errors["symmetry"] <= MARGINAL_TOL, f"{name}: symmetry")
    require(errors["diagonal"] <= MARGINAL_TOL, f"{name}: zero diagonal")
    require(errors["range_violation"] <= MARGINAL_TOL, f"{name}: range")
    require(errors["constraint_QB_minus_Y"] <= CONSTRAINT_TOL, f"{name}: Q B = Y")
    return errors


def verify_graph(input_path: Path, output_path: Path):
    before_input, before_output = sha256(input_path), sha256(output_path)
    with np.load(input_path, allow_pickle=False) as source:
        y, b, a, p = assert_graph_inputs(source)
        groups = [np.flatnonzero(b[:, g]) for g in range(8)]
        independent = {name: np.zeros((40, 40), dtype=dtype) for name, dtype in (
            ("q_uniform", float), ("q_oracle", float),
            ("forced_zero", bool), ("forced_one", bool))}
        numerator = np.zeros((40, 40), dtype=object)
        denominator = np.zeros((40, 40), dtype=object)
        blocks = []
        full_support = 1
        for g in range(8):
            for h in range(g, 8):
                ii, jj = groups[g], groups[h]
                ix = np.ix_(ii, jj)
                if g == h:
                    block = within_marginals(y[ii, g], p[ix])
                else:
                    block = cross_marginals(y[ii, h], y[jj, g], p[ix])
                for name, full in independent.items():
                    full[ix] = block[name]
                    if g != h:
                        full[np.ix_(jj, ii)] = block[name].T
                numerator[ix] = block["edge_one_counts"]
                denominator[ix] = block["support_count"]
                if g != h:
                    numerator[np.ix_(jj, ii)] = np.asarray(block["edge_one_counts"], dtype=object).T
                    denominator[np.ix_(jj, ii)] = block["support_count"]
                full_support *= block["support_count"]
                record = {key: value for key, value in block.items()
                          if key not in independent}
                record["groups"] = [g, h]
                record["support_count"] = str(record["support_count"])
                blocks.append(record)
        offdiag = ~np.eye(40, dtype=bool)
        np.fill_diagonal(denominator, 0)
        independent["unresolved"] = offdiag & ~independent["forced_zero"] & ~independent["forced_one"]
        errors = {}
        with np.load(output_path, allow_pickle=False) as output:
            require(set(independent).issubset(output.files), "output missing required marginal/mask fields")
            for name in ("q_uniform", "q_oracle"):
                actual = np.asarray(output[name], dtype=float)
                errors[f"independent_{name}"] = matrix_checks(independent[name], b, y, name)
                errors[f"saved_{name}"] = matrix_checks(actual, b, y, name)
                error = float(np.max(np.abs(actual - independent[name])))
                errors[f"max_abs_{name}_difference"] = error
                require(error <= MARGINAL_TOL, f"{name} does not agree with independent result: {error}")
                require(np.all(actual[independent["forced_zero"]] == 0),
                        f"{name}: forced-zero values must be exactly zero")
                require(np.all(actual[independent["forced_one"]] == 1),
                        f"{name}: forced-one values must be exactly one")
                require(np.all(actual[independent["unresolved"]] > 0)
                        and np.all(actual[independent["unresolved"]] < 1),
                        f"{name}: unresolved edges lost both-state support numerically")
            for name in ("forced_zero", "forced_one", "unresolved"):
                actual = output[name]
                require(actual.shape == (40, 40) and actual.dtype.kind == "b", f"{name}: bool full matrix required")
                require(np.array_equal(actual, independent[name]), f"{name}: differs from exact support counts")
            for name, expected in (("support_numerator", numerator), ("support_denominator", denominator)):
                if name in output.files:
                    actual = output[name]
                    require(actual.dtype.kind in "iu", f"{name}: integer array required")
                    require(np.array_equal(actual, expected), f"{name}: differs from exact independent counts")
            require(not np.any(independent["forced_zero"] & independent["forced_one"]),
                    "overlapping forced sets")
            require(np.array_equal(independent["forced_zero"] | independent["forced_one"]
                                   | independent["unresolved"], offdiag),
                    "forced/unresolved sets do not partition off-diagonal pairs")
            require(np.all(a[independent["forced_zero"]] == 0)
                    and np.all(a[independent["forced_one"]] == 1),
                    "forced sets conflict with saved feasible adjacency")
            # If the primary output retains source arrays, bind them to their original values.
            for name in ("Y", "B", "hidden_A", "true_P", "bayes_P", "nuclear_P"):
                if name in output.files:
                    require(np.array_equal(output[name], source[name]), f"source field changed: {name}")
            if "block_counts" in output.files:
                primary_counts = np.asarray(output["block_counts"]).ravel()
                require(len(primary_counts) == len(blocks), "block_counts dimension")
                require([int(x) for x in primary_counts] == [int(x["support_count"]) for x in blocks],
                        "block_counts differ from independent exact counts")
        require(sha256(input_path) == before_input and sha256(output_path) == before_output,
                "source/output changed during verification")
        upper = np.triu(np.ones((40, 40), dtype=bool), 1)
        return {
            "file": input_path.name, "status": "passed",
            "input_sha256": before_input, "output_sha256": before_output,
            "blocks": blocks, "full_fibre_support_count": str(full_support),
            "uniform_oracle_support_identical": True,
            "support_basis": "Exact integer edge-one/edge-zero counts; true_P strictly interior off diagonal.",
            "pair_counts": {name: int(np.sum(independent[name] & upper))
                            for name in ("forced_zero", "forced_one", "unresolved")},
            "errors": errors,
        }


def verify_metadata_seed_order(source_seeds, primary_metadata, filename_seeds):
    """Preserve the source metadata order, independently of filename sorting."""
    expected = [int(seed) for seed in source_seeds]
    require(len(expected) == len(set(expected)) == 40, "source metadata requires 40 unique seeds")
    require(set(expected) == set(filename_seeds), "source metadata seed set differs from original files")
    require([int(seed) for seed in primary_metadata["seeds"]] == expected,
            "primary seeds order differs from source metadata seeds")
    runs = primary_metadata["runs"]
    require([int(run["seed"]) for run in runs] == expected,
            "primary runs order differs from source metadata seeds")
    return {int(run["seed"]): run for run in runs}


class IndependentTests(unittest.TestCase):
    def test_4_by_4_four_solution_fibre(self):
        result = cross_marginals([3, 3, 1, 1], [3, 3, 1, 1], np.full((4, 4), 0.5))
        self.assertEqual(result["support_count"], 4)
        expected = np.array([[1, 1, .5, .5], [1, 1, .5, .5], [.5, .5, 0, 0], [.5, .5, 0, 0]])
        np.testing.assert_array_equal(result["q_uniform"], expected)
        np.testing.assert_allclose(result["q_oracle"], expected, rtol=0, atol=MARGINAL_TOL)
        self.assertEqual(int(result["forced_one"].sum()), 4)
        self.assertEqual(int(result["forced_zero"].sum()), 4)

    def test_unequal_weighted_two_by_two(self):
        odds = np.array([[1., 2.], [3., 4.]])
        result = cross_marginals([1, 1], [1, 1], odds / (1 + odds))
        self.assertEqual(result["support_count"], 2)
        np.testing.assert_allclose(result["q_uniform"], .5, rtol=0, atol=0)
        np.testing.assert_allclose(result["q_oracle"], [[.4, .6], [.6, .4]], rtol=0, atol=MARGINAL_TOL)

    def test_row_column_odds_cancellation(self):
        odds = np.array([[.2, 2., .8, 1.5], [1.2, .7, 3., .4],
                         [4., .3, .6, 2.], [1., 5., .25, .9]])
        tilted = odds * np.array([.5, 2., 3., 1.5])[:, None] * np.array([4., .4, 1.3, .7])[None, :]
        original = cross_marginals([3, 3, 1, 1], [3, 3, 1, 1], odds / (1 + odds))
        changed = cross_marginals([3, 3, 1, 1], [3, 3, 1, 1], tilted / (1 + tilted))
        self.assertEqual(original["support_count"], changed["support_count"])
        np.testing.assert_allclose(original["q_oracle"], changed["q_oracle"], rtol=0, atol=MARGINAL_TOL)

    def test_within_perfect_matchings(self):
        p = np.full((4, 4), .5)
        np.fill_diagonal(p, 0)
        result = within_marginals([1, 1, 1, 1], p)
        self.assertEqual(result["support_count"], 3)
        expected = (np.ones((4, 4)) - np.eye(4)) / 3
        np.testing.assert_allclose(result["q_uniform"], expected, rtol=0, atol=0)
        np.testing.assert_allclose(result["q_oracle"], expected, rtol=0, atol=MARGINAL_TOL)

    def test_within_weighted_matchings(self):
        odds = np.array([[0., 1., 2., 3.], [1., 0., 4., 5.],
                         [2., 4., 0., 6.], [3., 5., 6., 0.]])
        result = within_marginals([1, 1, 1, 1], odds / (1 + odds))
        expected = np.array([[0., 6., 10., 12.], [6., 0., 12., 10.],
                             [10., 12., 0., 6.], [12., 10., 6., 0.]]) / 28
        np.testing.assert_allclose(result["q_oracle"], expected, rtol=0, atol=MARGINAL_TOL)

    def test_all_two_by_three_fibres_against_direct_bernoulli(self):
        p = np.array([[.13, .27, .48], [.73, .62, .31]])
        fibres = {}
        for mask in range(1 << 6):
            matrix = np.array([(mask >> j) & 1 for j in range(6)]).reshape(2, 3)
            key = (tuple(matrix.sum(axis=1)), tuple(matrix.sum(axis=0)))
            weight = float(np.prod(np.where(matrix, p, 1 - p)))
            count, numerator, partition, weighted = fibres.get(
                key, (0, np.zeros((2, 3)), 0.0, np.zeros((2, 3))))
            fibres[key] = (count + 1, numerator + matrix, partition + weight,
                           weighted + matrix * weight)
        for (rows, columns), (count, numerator, partition, weighted) in fibres.items():
            result = cross_marginals(rows, columns, p)
            self.assertEqual(result["support_count"], count)
            np.testing.assert_allclose(result["q_uniform"], numerator / count, rtol=0, atol=0)
            np.testing.assert_allclose(result["q_oracle"], weighted / partition, rtol=0, atol=MARGINAL_TOL)

    def test_infeasible_cross_support_and_endpoint_rejection(self):
        count, log_partition, _ = cross_partition([2, 0], [2, 0], np.zeros((2, 2)))
        self.assertEqual(count, 0)
        self.assertEqual(log_partition, -math.inf)
        with self.assertRaises(ValueError):
            cross_marginals([1, 1], [1, 1], np.array([[0., .5], [.5, .5]]))

    def test_source_seed_order_is_not_filename_order(self):
        source_seeds = [101, 202, 303, 404] + list(range(1001, 1037))
        filename_seeds = sorted(source_seeds, key=str)
        self.assertNotEqual(source_seeds, filename_seeds)
        metadata = {"seeds": source_seeds, "runs": [{"seed": seed} for seed in source_seeds]}
        by_seed = verify_metadata_seed_order(source_seeds, metadata, filename_seeds)
        self.assertEqual(list(by_seed), source_seeds)
        with self.assertRaises(ValueError):
            verify_metadata_seed_order(source_seeds, {**metadata, "seeds": filename_seeds}, filename_seeds)
        with self.assertRaises(ValueError):
            verify_metadata_seed_order(source_seeds, {**metadata, "runs": metadata["runs"][::-1]}, filename_seeds)


def run_tests():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(IndependentTests)
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(result.wasSuccessful(), "independent verifier self-tests failed")
    return {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--outputs", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--self-test", action="store_true", help="run built-in independent toy tests only")
    args = parser.parse_args()
    if args.self_test:
        require(args.inputs is None and args.outputs is None and args.report is None,
                "--self-test does not accept input/output/report paths")
        run_tests()
        return 0
    require(all(path is not None for path in (args.inputs, args.outputs, args.report)),
            "--inputs, --outputs, --report are required")
    require(not args.report.exists(), "refusing to overwrite report")
    inputs, outputs, report = args.inputs.resolve(), args.outputs.resolve(), args.report.resolve()
    require(inputs.is_dir() and outputs.is_dir(), "input/output directories must already exist")
    files = sorted(inputs.glob("run_k8_seed*.npz"))
    require(len(files) == 40, f"scope requires exactly 40 original K=8 files, found {len(files)}")
    require({p.name for p in files} == {p.name for p in outputs.glob("run_k8_seed*.npz")},
            "output K=8 file set differs from original 40 graphs")
    started = time.monotonic()
    script_hash = sha256(Path(__file__))
    input_hashes = {path.name: sha256(path) for path in files}
    output_hashes = {path.name: sha256(outputs / path.name) for path in files}
    primary_json = outputs / "results.json"
    primary_json_hash = sha256(primary_json) if primary_json.exists() else None
    primary_metadata = json.loads(primary_json.read_text(encoding="utf-8")) if primary_json.exists() else None
    source_json = inputs / "results.json"
    source_json_hash = sha256(source_json)
    source_metadata = json.loads(source_json.read_text(encoding="utf-8"))
    tests = run_tests()
    result = {
        "schema": "ard-k8-independent-verification-v1", "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": str(inputs), "outputs": str(outputs),
        "verifier_sha256": script_hash,
        "input_sha256_before": input_hashes, "output_sha256_before": output_hashes,
        "primary_results_json_sha256": primary_json_hash,
        "source_results_json_sha256": source_json_hash,
        "source_metadata_seed_order": source_metadata["metadata"]["seeds"],
        "algorithms": {
            "cross": "Remaining-column-margin DP with separate edge-fixed partitions; Python integer counts and log-sum-exp weights.",
            "within": "All undirected edge bitmasks, direct degrees, complete Bernoulli likelihoods.",
        },
        "tolerances": {"marginal_abs": MARGINAL_TOL, "constraint_abs": CONSTRAINT_TOL},
        "self_tests": tests, "graphs": [],
    }
    try:
        for index, path in enumerate(files, 1):
            result["graphs"].append(verify_graph(path, outputs / path.name))
            print(f"Verified {index}/40: {path.name}", flush=True)
        if primary_metadata is not None and isinstance(primary_metadata.get("runs"), list):
            ordered_seeds = [int(path.stem.removeprefix("run_k8_seed")) for path in files]
            by_seed = verify_metadata_seed_order(source_metadata["metadata"]["seeds"],
                                                  primary_metadata, ordered_seeds)
            for graph, seed in zip(result["graphs"], ordered_seeds):
                primary_blocks = by_seed[seed]["blocks"]
                expected = [(block["groups"][0], block["groups"][1], int(block["support_count"]))
                            for block in graph["blocks"]]
                actual = [(int(block["g"]), int(block["h"]), int(block["support_count"]))
                          for block in primary_blocks]
                require(actual == expected, f"seed {seed}: primary block support counts differ")
            result["primary_metadata_block_counts_verified"] = True
            result["primary_metadata_seed_order_verified"] = True
        result["input_sha256_after"] = {path.name: sha256(path) for path in files}
        result["output_sha256_after"] = {path.name: sha256(outputs / path.name) for path in files}
        require(result["input_sha256_after"] == input_hashes, "original input files changed during full run")
        require(result["output_sha256_after"] == output_hashes, "primary output files changed during full run")
        if primary_json_hash is not None:
            require(primary_json_hash == sha256(primary_json), "primary metadata changed during verification")
        require(source_json_hash == sha256(source_json), "source metadata changed during verification")
        require(script_hash == sha256(Path(__file__)), "verifier changed during execution")
        result["status"] = "passed"
    except Exception as error:
        result["status"] = "failed"
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
    result["elapsed_seconds"] = time.monotonic() - started
    result["graphs_verified"] = len(result["graphs"])
    result["max_abs_q_uniform_difference"] = max(
        (g["errors"]["max_abs_q_uniform_difference"] for g in result["graphs"]), default=None)
    result["max_abs_q_oracle_difference"] = max(
        (g["errors"]["max_abs_q_oracle_difference"] for g in result["graphs"]), default=None)
    result["max_abs_constraint_error"] = max(
        (values["constraint_QB_minus_Y"] for g in result["graphs"] for values in g["errors"].values()
         if isinstance(values, dict)), default=None)
    with report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "graphs_verified": result["graphs_verified"],
                      "report": str(report), "elapsed_seconds": result["elapsed_seconds"]}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
