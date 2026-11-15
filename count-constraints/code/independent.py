from __future__ import annotations
from functools import lru_cache
import hashlib
import math
from pathlib import Path
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
    required = {"Y", "B", "hidden_A", "true_P"}
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
            for name in ("Y", "B", "hidden_A", "true_P"):
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
