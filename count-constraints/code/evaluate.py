"""Read-only, block-paired Stage B evaluation. No estimation or data generation.

The independent unit is a seed block, shared across densities and K. Missing
artifacts invalidate a primary mean; undefined AUC domains instead receive an
explicit conditional-on-defined-domain mean and denominator. Solver caps and
failed diagnostic screens never select which available estimates are scored.
"""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import math

import numpy as np

HERE = Path(__file__).resolve().parent
RANK_METRICS = ("auc_raw", "auc_zero_adjusted", "auc_U")
COUNT_NAMES = (
    "pairs", "edges", "nonedges", "mean_true_probability", "realized_density",
    "mean_degree", "mean_count", "zero_count_fraction", "zero_excluded", "q0",
    "zero_excluded_pair_fraction", "remaining_U", "remaining_U_fraction",
    "edges_U", "nonedges_U", "forced_zero", "forced_one", "unresolved_V",
    "edges_V", "nonedges_V", "extra_forced_zero_inside_U",
    "forced_zero_nonedge_fraction", "forced_one_edge_fraction",
    "forced_zero_pair_fraction", "forced_one_pair_fraction", "unresolved_V_fraction",
    "full_auc_unresolved_weight", "forced_only_auc",
)
CONDITIONAL_COUNTS = set(COUNT_NAMES[15:])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def auc(labels, scores, tolerance=0.0):
    """Exact numerical ties receive half credit; undefined domains return None."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    require(labels.shape == scores.shape, "AUC dimensions")
    require(np.isin(labels, (0, 1)).all(), "AUC labels must be binary")
    require(np.isfinite(scores).all(), "AUC scores must be finite")
    positive, negative = int(np.sum(labels == 1)), int(np.sum(labels == 0))
    if not positive or not negative:
        return None
    require(tolerance >= 0, "Negative tie tolerance")
    if tolerance:
        # Cast before subtracting: uint8 count-product differences otherwise wrap.
        tolerance_scores = scores.astype(np.float64, copy=False)
        differences = tolerance_scores[labels == 1, None] - tolerance_scores[labels == 0][None, :]
        return float((np.sum(differences > tolerance) + .5 * np.sum(np.abs(differences) <= tolerance)) / differences.size)
    order = np.argsort(scores, kind="stable")
    values, labs = scores[order], labels[order]
    boundaries = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
    wins = 0.0
    lower_negatives = 0
    for first, last in zip(boundaries[:-1], boundaries[1:]):
        pos = int(labs[first:last].sum())
        neg = int(last - first - pos)
        wins += pos * (lower_negatives + 0.5 * neg)
        lower_negatives += neg
    return float(wins / (positive * negative))


def fixed_scores(y, b):
    """Observable baselines use only Y and B, never hidden_A or true_P."""
    y, b = np.asarray(y), np.asarray(b)
    n = len(b)
    require(n > 1 and y.shape == b.shape, "Y/B dimensions")
    require(np.isin(b, (0, 1)).all() and np.all(b.sum(axis=1) == 1), "B must be a partition")
    require(np.isfinite(y).all() and np.all(y >= 0) and np.all(y == np.floor(y)), "Invalid counts")
    group = b.argmax(axis=1)
    directed = y[:, group]
    product = directed * directed.T
    np.fill_diagonal(product, 0)
    density = np.full((n, n), float(y.sum() / (n * (n - 1))))
    np.fill_diagonal(density, 0)
    return {"Density": density, "Product": product, "ZeroOnly": (product > 0).astype(float)}


def validate_probability(p, n, name):
    p = np.asarray(p)
    require(p.shape == (n, n) and np.isfinite(p).all(), f"{name}: dimensions/nonfinite")
    require(np.max(np.abs(p - p.T)) < 1e-10 and np.max(np.abs(np.diag(p))) < 1e-10,
            f"{name}: symmetry/diagonal")
    require(np.all((p >= 0) & (p <= 1)), f"{name}: outside [0,1]")


def context(arrays, benchmark=None):
    y, b, a, truth = (np.asarray(arrays[x]) for x in ("Y", "B", "hidden_A", "true_P"))
    n = len(a)
    fixed = fixed_scores(y, b)
    validate_probability(a, n, "hidden_A")
    validate_probability(truth, n, "true_P")
    require(np.isin(a, (0, 1)).all() and np.array_equal(a @ b, y), "hidden_A B != Y")
    i, j = np.triu_indices(n, 1)
    labels = a[i, j].astype(int)
    u = fixed["ZeroOnly"][i, j].astype(bool)
    require(np.all(labels[~u] == 0), "Zero exclusion contradicts A")
    m, nn, pairs = int(labels.sum()), int((labels == 0).sum()), len(i)
    counts = {
        "pairs": pairs, "edges": m, "nonedges": nn,
        "mean_true_probability": float(truth[i, j].mean()), "realized_density": m / pairs,
        "mean_degree": 2 * m / n, "mean_count": float(y.mean()),
        "zero_count_fraction": float(np.mean(y == 0)), "zero_excluded": int((~u).sum()),
        "q0": float((~u).sum() / nn) if nn else None,
        "zero_excluded_pair_fraction": float((~u).mean()),
        "remaining_U": int(u.sum()), "remaining_U_fraction": float(u.mean()),
        "edges_U": int(labels[u].sum()), "nonedges_U": int((labels[u] == 0).sum()),
    }
    ctx = dict(y=y, b=b, truth=truth, n=n, i=i, j=j, labels=labels, u=u,
               domains={"": np.ones(pairs, dtype=bool), "_U": u}, counts=counts, fixed=fixed)
    if benchmark is not None:
        qu, qo = (np.asarray(benchmark[key]) for key in ("q_uniform", "q_oracle"))
        for name, q in (("q_uniform", qu), ("q_oracle", qo)):
            validate_probability(q, n, name)
            require(np.max(np.abs(q @ b - y)) < 2e-10, f"{name} B != Y")
        for key in ("forced_zero", "forced_one", "unresolved"):
            mask = np.asarray(benchmark[key])
            require(mask.shape == (n, n) and np.isin(mask, (0, 1)).all()
                    and np.array_equal(mask, mask.T) and not np.diag(mask).any(), f"Invalid {key}")
        f0, f1, v = (np.asarray(benchmark[key], dtype=bool)[i, j]
                     for key in ("forced_zero", "forced_one", "unresolved"))
        require(np.all(f0.astype(int) + f1 + v == 1), "Forced sets must partition pairs")
        require(np.all(labels[f0] == 0) and np.all(labels[f1] == 1) and np.all(f0[~u]),
                "Forced sets contradict A or zero exclusion")
        num, den = (np.asarray(benchmark[key])[i, j] for key in ("support_numerator", "support_denominator"))
        require(np.all(num == np.floor(num)) and np.all(den == np.floor(den)), "Noninteger support counts")
        require(np.all(den > 0) and np.all(num >= 0) and np.all(num <= den), "Invalid support counts")
        require(np.array_equal(f0, num == 0) and np.array_equal(f1, num == den), "Support/forced-set mismatch")
        require(np.allclose(qu[i, j], num / den, atol=2e-15, rtol=0), "Uniform/support mismatch")
        for q in (qu[i, j], qo[i, j]):
            require(np.all(q[f0] == 0) and np.all(q[f1] == 1) and np.all((q[v] > 0) & (q[v] < 1)),
                    "Conditional support lost")
        mv, nv = int(labels[v].sum()), int((labels[v] == 0).sum())
        weight = mv * nv / (m * nn) if m and nn else None
        counts.update({"forced_zero": int(f0.sum()), "forced_one": int(f1.sum()),
                       "unresolved_V": int(v.sum()), "edges_V": mv, "nonedges_V": nv,
                       "extra_forced_zero_inside_U": int((f0 & u).sum()),
                       "forced_zero_nonedge_fraction": float(f0.sum() / nn) if nn else None,
                       "forced_one_edge_fraction": float(f1.sum() / m) if m else None,
                       "forced_zero_pair_fraction": float(f0.mean()), "forced_one_pair_fraction": float(f1.mean()),
                       "unresolved_V_fraction": float(v.mean()), "full_auc_unresolved_weight": weight,
                       "forced_only_auc": 1 - weight / 2 if weight is not None else None})
        ctx.update(f0=f0, f1=f1, v=v, oracle=qo[i, j])
        ctx["domains"]["_V"] = v
    return ctx


def forced_rank_score(score, f0, f1, free):
    """Ranking-only F0 < V < F1 adjustment preserving all ties within V."""
    result = np.zeros(len(score), dtype=float)
    result[f1] = 1.0
    if free.any():
        distinct = np.unique(score[free])
        result[free] = (np.searchsorted(distinct, score[free]) + 1) / (len(distinct) + 1)
    return result


def metric_names(name, k):
    names = list(RANK_METRICS) + ["zero_edge_score_fraction_r0", "zero_identity_rhs", "zero_identity_error"]
    names += [f"{base}_tie_{eps:g}" for base in RANK_METRICS for eps in (1e-12, 1e-10)]
    if k == 8:
        names += ["auc_V", "auc_full_forced_layer", "full_forced_identity_rhs", "full_forced_identity_error"]
        names += [f"auc_V_tie_{eps:g}" for eps in (1e-12, 1e-10)]
    if name not in ("Product", "ZeroOnly"):
        names += ["ard_rmse", "brier_A", "brier_A_U"]
        if not name.startswith("Q_"):
            names += ["probability_rmse"]
        if k == 8:
            names += ["brier_A_V"]
            for suffix in ("", "_U", "_V"):
                names += ["conditional_excess_brier" + suffix, "conditional_expected_brier" + suffix]
    return names


def score_model(p, ctx, name, k, conditional_missing_reason="missing:conditional"):
    labels, u, i, j = (ctx[key] for key in ("labels", "u", "i", "j"))
    p = np.asarray(p)
    if name not in ("Product", "ZeroOnly"):
        validate_probability(p, ctx["n"], name)
    require(p.shape == (ctx["n"], ctx["n"]) and np.isfinite(p).all() and np.all(p >= 0), "Invalid score")
    s = p[i, j]
    values = dict.fromkeys(metric_names(name, k))
    reasons = {key: conditional_missing_reason for key in values}

    def put(key, value, reason="undefined:domain_has_fewer_than_two_classes"):
        values[key] = value
        if value is None:
            reasons[key] = reason
        else:
            reasons.pop(key, None)

    put("auc_raw", auc(labels, s))
    put("auc_U", auc(labels[u], s[u]))
    put("auc_zero_adjusted", auc(labels, s * u))
    for eps in (1e-12, 1e-10):
        put(f"auc_raw_tie_{eps:g}", auc(labels, s, eps))
        put(f"auc_U_tie_{eps:g}", auc(labels[u], s[u], eps))
        put(f"auc_zero_adjusted_tie_{eps:g}", auc(labels, s * u, eps))
    r0 = float(np.mean(s[labels == 1] == 0)) if np.any(labels == 1) else None
    put("zero_edge_score_fraction_r0", r0, "undefined:no_edges")
    q0 = ctx["counts"]["q0"]
    rhs = None
    if values["auc_raw"] is not None:
        # If U has no nonedges, its AUC is undefined but its coefficient is zero.
        rhs = q0 * (1 - r0 / 2)
        if q0 < 1:
            rhs += (1 - q0) * values["auc_U"]
    put("zero_identity_rhs", rhs)
    put("zero_identity_error", abs(values["auc_zero_adjusted"] - rhs) if rhs is not None else None)
    if rhs is not None:
        require(values["zero_identity_error"] < 2e-12, "Zero-decomposition identity failed")
    if "v" in ctx:
        v = ctx["v"]
        put("auc_V", auc(labels[v], s[v]))
        for eps in (1e-12, 1e-10):
            put(f"auc_V_tie_{eps:g}", auc(labels[v], s[v], eps))
        layered = auc(labels, forced_rank_score(s, ctx["f0"], ctx["f1"], v))
        weight = ctx["counts"]["full_auc_unresolved_weight"]
        full_rhs = (1.0 if weight == 0 else 1 - weight * (1 - values["auc_V"])) if weight is not None else None
        put("auc_full_forced_layer", layered)
        put("full_forced_identity_rhs", full_rhs)
        put("full_forced_identity_error", abs(layered - full_rhs) if full_rhs is not None else None)
        if full_rhs is not None:
            require(abs(layered - full_rhs) < 2e-12, "Full-forced identity failed")
            if name.startswith("Q_"):
                require(abs(values["auc_raw"] - full_rhs) < 2e-12, "Conditional raw/layer identity failed")
    if name not in ("Product", "ZeroOnly"):
        put("ard_rmse", float(np.sqrt(np.mean((p @ ctx["b"] - ctx["y"]) ** 2))))
        if not name.startswith("Q_"):
            put("probability_rmse", float(np.sqrt(np.mean((s - ctx["truth"][i, j]) ** 2))))
        for suffix, mask in ctx["domains"].items():
            put("brier_A" + suffix, float(np.mean((s[mask] - labels[mask]) ** 2)) if mask.any() else None,
                "undefined:empty_domain")
            if "oracle" in ctx:
                q = ctx["oracle"][mask]
                excess = float(np.mean((s[mask] - q) ** 2)) if mask.any() else None
                put("conditional_excess_brier" + suffix, excess, "undefined:empty_domain")
                put("conditional_expected_brier" + suffix,
                    excess + float(np.mean(q * (1 - q))) if mask.any() else None, "undefined:empty_domain")
    return values, reasons


def model_names(k, penalties):
    names = ["Density", "Product", "ZeroOnly", "TrueP", "Circular", "Circular_first_half",
             "Circular_last_half", "Circular_group12", "Circular_group34"]
    names += [f"Circular_chain{x}" for x in range(1, 5)]
    for penalty in penalties:
        names += [f"Nuclear_lambda{penalty:g}", f"Nuclear_early_lambda{penalty:g}"]
    if k == 8:
        names += ["Q_uniform", "Q_oracle"]
    return names


def evaluate_cell(root, density, k, seed):
    """Return one cell, retaining every missing artifact, cap and screen failure."""
    root = Path(root)
    contract = json.loads((root / "RUN_CONTRACT.json").read_text())
    penalties = contract["nuclear"]["penalties"]
    names = model_names(k, penalties)
    cell = {"density": density, "k": k, "seed": seed, "status": "complete", "metrics": {},
            "missing_reasons": {}, "counts": {}, "count_missing_reasons": {}, "failures": [],
            "provenance": {}, "circular": {}, "nuclear": {}, "conditional": {}}

    def fail(lane, exc):
        reason = f"unavailable:{lane}:{type(exc).__name__}:{exc}"
        cell["failures"].append({"lane": lane, "reason": reason})
        cell["status"] = "incomplete"
        return reason

    def load(path, kind="npz"):
        path = Path(path)
        if kind == "json":
            result = json.loads(path.read_text())
            require(isinstance(result, dict), f"JSON object required: {path.name}")
        else:
            with np.load(path, allow_pickle=False) as archive:
                result = {key: archive[key] for key in archive.files}
        cell["provenance"][str(path.relative_to(root))] = sha(path)
        return result

    missing = {}
    stem = f"run_k{k}_seed{seed}"
    try:
        arrays = load(root / "inputs" / density / (stem + ".npz"))
        ctx = context(arrays)
        require(ctx["n"] == contract["n"] and arrays["B"].shape[1] == k, "Input contract dimensions")
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        reason = fail("inputs", exc)
        for name in names:
            cell["metrics"][name] = dict.fromkeys(metric_names(name, k))
            cell["missing_reasons"][name] = {key: reason for key in metric_names(name, k)}
        cell["counts"] = dict.fromkeys(key for key in COUNT_NAMES if k == 8 or key not in CONDITIONAL_COUNTS)
        cell["count_missing_reasons"] = dict.fromkeys(cell["counts"], reason)
        return cell
    estimates = {**ctx["fixed"], "TrueP": arrays["true_P"]}
    conditional_reason = "missing:conditional"
    if k == 8:
        try:
            bp = root / "conditional" / density / (stem + ".npz")
            meta = load(bp.with_suffix(".json"), "json")
            cell["conditional"] = meta
            require(meta.get("status") not in ("failed", "failure", "error", "incomplete"), "Conditional output status failed")
            benchmark = load(bp)
            if "npz_sha256" in meta:
                require(sha(bp) == meta["npz_sha256"], "Conditional hash mismatch")
            ctx = context(arrays, benchmark)
            estimates.update(Q_uniform=benchmark["q_uniform"], Q_oracle=benchmark["q_oracle"])
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            conditional_reason = fail("conditional", exc)
            missing.update(Q_uniform=conditional_reason, Q_oracle=conditional_reason)
    cdir = root / "circular" / density / stem
    circular_names = [name for name in names if name.startswith("Circular")]
    try:
        cm = load(cdir / "summary.npz")
        cell["circular"] = load(cdir / "complete.json", "json")
        require(all(isinstance(cell["circular"].get(key), dict) for key in ("matrix_diagnostics", "scalar_diagnostics")),
                "Missing circular diagnostic summary")
        load(cdir / "diagnostics.npz")
        for name, key in (("Circular", "pooled_P"), ("Circular_first_half", "first_half_P"),
                          ("Circular_last_half", "last_half_P"), ("Circular_group12", "group12_P"),
                          ("Circular_group34", "group34_P")):
            estimates[name] = cm[key]
        require(len(cm["chain_mean_P"]) == 4, "Four circular chains required")
        for chain in range(4):
            estimates[f"Circular_chain{chain + 1}"] = cm["chain_mean_P"][chain]
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        reason = fail("circular", exc)
        missing.update(dict.fromkeys(circular_names, reason))
        for name in circular_names:
            estimates.pop(name, None)
    for penalty in penalties:
        names_here = [f"Nuclear_lambda{penalty:g}", f"Nuclear_early_lambda{penalty:g}"]
        try:
            npz = root / "nuclear" / density / f"{stem}_lambda{penalty:g}.npz"
            meta = load(npz.with_suffix(".json"), "json")
            cell["nuclear"][f"{penalty:g}"] = meta
            require(meta.get("status") not in ("failed", "failure", "error", "incomplete"), "Nuclear output status failed")
            require(all(isinstance(meta.get(key), dict) for key in ("early", "final")), "Missing nuclear checkpoint metadata")
            fit = load(npz)
            if "npz_sha256" in meta:
                require(sha(npz) == meta["npz_sha256"], "Nuclear hash mismatch")
            estimates[names_here[0]], estimates[names_here[1]] = fit["final_P"], fit["early_P"]
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            missing.update(dict.fromkeys(names_here, fail(f"nuclear_lambda{penalty:g}", exc)))
    for name in names:
        if name in estimates:
            try:
                values, reasons = score_model(estimates[name], ctx, name, k, conditional_reason)
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                missing[name] = fail(name, exc)
            else:
                cell["metrics"][name], cell["missing_reasons"][name] = values, reasons
                continue
        reason = missing.get(name, "unavailable:estimate")
        cell["metrics"][name] = dict.fromkeys(metric_names(name, k))
        cell["missing_reasons"][name] = dict.fromkeys(metric_names(name, k), reason)
    cell["counts"] = ctx["counts"]
    for key in COUNT_NAMES:
        if key in CONDITIONAL_COUNTS and k != 8:
            continue
        if key not in cell["counts"]:
            cell["counts"][key] = None
            cell["count_missing_reasons"][key] = conditional_reason
        elif cell["counts"][key] is None:
            cell["count_missing_reasons"][key] = "undefined:zero_class_denominator"
    return cell


def stats(values, reasons=None):
    """Never disguise available-case means as complete independent-block means."""
    reasons = reasons if reasons is not None else [None if x is not None else "unavailable:unspecified" for x in values]
    require(len(values) == len(reasons), "Summary value/reason lengths")
    kept = []
    misses = []
    for value, reason in zip(values, reasons):
        if value is not None and math.isfinite(float(value)):
            kept.append(float(value))
        else:
            misses.append(reason or "unavailable:nonfinite")
    vals = np.asarray(kept)
    infrastructure = sum(not reason.startswith("undefined:") for reason in misses)
    n = len(vals)
    conditional = float(vals.mean()) if n else None
    sd = float(vals.std(ddof=1)) if n > 1 else None
    mcse = sd / math.sqrt(n) if n > 1 else None
    return {"n_expected": len(values), "n_valid": n, "n_missing": len(misses),
            "n_infrastructure_missing": infrastructure, "n_undefined": len(misses) - infrastructure,
            "status": "incomplete" if infrastructure else "conditional_on_defined_domain" if misses else "complete",
            "mean": conditional if not infrastructure else None, "mcse": mcse if not infrastructure else None,
            "conditional_mean": conditional, "conditional_mcse": mcse, "sd": sd,
            "min": float(vals.min()) if n else None, "max": float(vals.max()) if n else None,
            "positive": int(np.sum(vals > 0)), "negative": int(np.sum(vals < 0)), "zero": int(np.sum(vals == 0)),
            "missing_reasons": dict(Counter(misses))}


def block_stats(values, reasons, block_ids):
    """Attach the exact block intersection underlying each reported statistic."""
    require(len(values) == len(block_ids), "Summary value/block-ID lengths")
    result = stats(values, reasons)
    valid, missing = [], []
    for seed, value, reason in zip(block_ids, values, reasons):
        if value is not None and math.isfinite(float(value)):
            valid.append(seed)
        else:
            missing.append({"seed": seed, "reason": reason or "unavailable:nonfinite"})
    return {**result, "valid_block_ids": valid, "missing_blocks": missing}


def paired_difference(cell, left, right, metric):
    a, b = (cell["metrics"][name][metric] for name in (left, right))
    if a is not None and b is not None:
        return a - b, None
    reasons = [cell["missing_reasons"][name].get(metric, "unavailable:unspecified")
               for name, val in ((left, a), (right, b)) if val is None]
    # Infrastructure has precedence even if the other member has undefined AUC.
    return None, next((reason for reason in reasons if not reason.startswith("undefined:")), reasons[0])


def contrast_specs(k, penalties):
    pairs = [("Circular", "Density"), ("Circular", "Nuclear_lambda1"),
             ("Circular_last_half", "Circular_first_half"), ("Circular_group34", "Circular_group12")]
    for penalty in penalties:
        nu = f"Nuclear_lambda{penalty:g}"
        pairs += [(nu, "Density"), (nu, "Product"), (nu, f"Nuclear_early_lambda{penalty:g}")]
        if k == 8:
            pairs += [("Q_uniform", nu), (nu, "Q_uniform")]
    if k == 8:
        pairs += [("Q_uniform", "Circular"), ("Q_uniform", "Product"), ("Q_oracle", "Q_uniform")]
    return [(left, right, metric) for left, right in pairs
            for metric in sorted(set(metric_names(left, k)) & set(metric_names(right, k)))
            if "identity" not in metric and "_tie_" not in metric and metric != "zero_edge_score_fraction_r0"]


def summarize(root, destination=None):
    """Write one fresh readout; refuse to overwrite any prior summary artifact."""
    root = Path(root)
    destination = root if destination is None else Path(destination)
    destinations = [destination / name for name in ("SUMMARY.json", "cell-metrics.json", "summary.csv", "paired-contrasts.csv")]
    for path in destinations:
        if path.exists():
            raise FileExistsError(f"Preserving existing readout: {path}")
    contract_path = root / "RUN_CONTRACT.json"
    contract = json.loads(contract_path.read_text())
    densities = [x["id"] for x in contract["densities"]]
    seeds, ks, penalties = contract["seeds"], contract["k"], contract["nuclear"]["penalties"]
    require(len(set(seeds)) == len(seeds) and len(set(densities)) == len(densities), "Duplicate design cells")
    cells = [evaluate_cell(root, density, k, seed) for density in densities for k in ks for seed in seeds]
    lookup = {(x["density"], x["k"], x["seed"]): x for x in cells}
    summaries, contrasts, reviews = [], [], []
    for density in densities:
        for k in ks:
            group = [lookup[density, k, seed] for seed in seeds]
            for name in model_names(k, penalties):
                for metric in metric_names(name, k):
                    summaries.append({"density": density, "k": k, "model": name, "metric": metric,
                                      **block_stats([x["metrics"][name][metric] for x in group],
                                                    [x["missing_reasons"][name].get(metric) for x in group], seeds)})
            for metric in COUNT_NAMES:
                if metric in CONDITIONAL_COUNTS and k != 8:
                    continue
                summaries.append({"density": density, "k": k, "model": "Design", "metric": metric,
                                  **block_stats([x["counts"][metric] for x in group],
                                                [x["count_missing_reasons"].get(metric) for x in group], seeds)})
            for left, right, metric in contrast_specs(k, penalties):
                values, reasons = zip(*(paired_difference(x, left, right, metric) for x in group))
                contrasts.append({"type": "within_density", "density": density, "reference_density": None,
                                  "k": k, "left": left, "right": right, "metric": metric,
                                  **block_stats(values, reasons, seeds)})
            # Descriptive finite-computation flags, never fit/chain exclusion rules.
            comparisons = [("Circular_last_half", "Circular_first_half"), ("Circular_group34", "Circular_group12")]
            comparisons += [(f"Nuclear_lambda{lam:g}", f"Nuclear_early_lambda{lam:g}") for lam in penalties]
            for left, right in comparisons:
                for metric in ("auc_raw", "auc_U", "probability_rmse", "ard_rmse") + (("auc_V",) if k == 8 else ()):
                    reference = "Q_uniform" if metric == "auc_V" else "Product" if metric.startswith("auc") else "Density"
                    before_v, before_r = zip(*(paired_difference(x, right, reference, metric) for x in group))
                    after_v, after_r = zip(*(paired_difference(x, left, reference, metric) for x in group))
                    before, after = block_stats(before_v, before_r, seeds), block_stats(after_v, after_r, seeds)
                    if any(x["mean"] is None or x["mcse"] is None for x in (before, after)):
                        continue
                    delta = after["mean"] - before["mean"]
                    threshold = .5 * max(before["mcse"], after["mcse"])
                    reversal = abs(before["mean"]) > 2 * before["mcse"] and before["mean"] * after["mean"] < 0
                    if (abs(delta) > threshold and abs(delta) > 1e-14) or reversal:
                        reviews.append({"density": density, "k": k, "left": left, "right": right, "metric": metric,
                                        "reference": reference, "mean_change": delta, "half_max_block_mcse": threshold,
                                        "previous_resolved_direction_reversed": reversal, "before": before, "after": after})
    # Density changes are differences of within-block differences, never unpaired means.
    for d_index, density in enumerate(densities):
        for reference in densities[:d_index]:
            for k in ks:
                for left, right, metric in contrast_specs(k, penalties):
                    values, reasons = [], []
                    for seed in seeds:
                        now, rn = paired_difference(lookup[density, k, seed], left, right, metric)
                        old, ro = paired_difference(lookup[reference, k, seed], left, right, metric)
                        values.append(now - old if now is not None and old is not None else None)
                        failures = [r for r in (rn, ro) if r is not None]
                        reasons.append(next((r for r in failures if not r.startswith("undefined:")),
                                            failures[0] if failures else None))
                    contrasts.append({"type": "density_change", "density": density, "reference_density": reference,
                                      "k": k, "left": left, "right": right, "metric": metric, **block_stats(values, reasons, seeds)})
    caps, screens = [], []
    for cell in cells:
        tag = {key: cell[key] for key in ("density", "k", "seed")}
        for penalty, meta in cell["nuclear"].items():
            final = meta.get("final") if isinstance(meta.get("final"), dict) else {}
            if "cap" in str(meta.get("status", "")).lower() or final.get("converged") is False:
                caps.append({**tag, "penalty": penalty, "metadata": meta})
        for scope in ("matrix_diagnostics", "scalar_diagnostics"):
            diagnostic = cell["circular"].get(scope)
            if not isinstance(diagnostic, dict) or diagnostic.get("screen_pass") is not True:
                screens.append({**tag, "scope": scope, "diagnostics": cell["circular"].get(scope),
                                "reason": "screen_failed_or_unavailable"})
    report = {"status": "complete" if all(x["status"] == "complete" for x in cells) else "incomplete",
              "contract_sha256": sha(contract_path), "independent_unit": "seed block shared across densities and K",
              "evaluation_source": {
                  "file": "evaluate.py", "sha256": sha(__file__),
                  "correction_note": "The frozen original is unchanged. This exact copy casts tolerance-AUC score operands to float64 before subtraction to prevent unsigned integer wraparound; primary exact-tie AUC and all other metric formulas are unchanged."},
              "n_independent_blocks": len(seeds), "n_expected_cells": len(densities) * len(ks) * len(seeds),
              "n_complete_cells": sum(x["status"] == "complete" for x in cells), "seeds": seeds,
              "densities": densities, "k": ks, "summary": summaries, "paired_contrasts": contrasts,
              "failures": [{**{key: x[key] for key in ("density", "k", "seed")}, **failure}
                           for x in cells for failure in x["failures"]],
              "nuclear_final_caps": caps, "circular_screen_flags": screens, "computation_review_flags": reviews,
              "predeclared_numerical_screens_cleared": not caps and not screens and all(x["status"] == "complete" for x in cells),
              "predeclared_numerical_review_flags_cleared": not reviews and all(x["status"] == "complete" for x in cells),
              "interpretation": ["All available fixed-penalty fits and all chains are retained, including capped fits.",
                                 "A numerical screen or identity check does not certify calibration, convergence, or a scientific claim.",
                                 "Numerical screens cover solver caps and Circular diagnostics; computation-review flags have a separate clearance flag.",
                                 "Incomplete infrastructure makes the primary mean/MCSE null; conditional_mean reports only the explicit available count.",
                                 "Undefined-domain means condition on evaluable seed blocks; n_valid is not silently replaced by 40.",
                                 "Density changes are paired sensitivity contrasts, not causal zero-exclusion effects.",
                                 "Q references target conditional graph prediction; their P-star RMSE is deliberately not defined.",
                                 "Intervals and MCSEs are descriptive, pointwise, and not multiplicity-adjusted."]}
    for path, data in ((destinations[0], report), (destinations[1], {"contract_sha256": sha(contract_path), "cells": cells})):
        with path.open("x") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
    for path, rows in ((destinations[2], summaries), (destinations[3], contrasts)):
        with path.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()} for row in rows)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE)
    args = parser.parse_args()
    result = summarize(args.root)
    print(json.dumps({key: result[key] for key in ("status", "n_expected_cells", "n_complete_cells")}, allow_nan=False))
