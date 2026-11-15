#!/usr/bin/env python3
"""Rebuild Table 6 and Tables A1–A3 from saved per-graph measurements.

Usage: python exploratory-tables/rebuild.py --out NEW_DIRECTORY
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJECT = "exploratory/project/"
SOURCE_PATHS = {
    "stage_a": PROJECT + "analysis/ard-reconstruction/compstat-stability-2026-11-04/SUMMARY.json",
    "method_contrasts": PROJECT + "analysis/ard-reconstruction/compstat-stability-2026-11-04/method-contrasts.json",
    "nuclear": PROJECT + "analysis/ard-reconstruction/compstat-stability-2026-11-04/nuclear/SUMMARY.json",
    "conditional": PROJECT + "analysis/ard-reconstruction/compstat-conditional-2026-11-04/results/results.json",
    "k4_zero": "exploratory-tables/k4-zero.json",
}
LABELS = {
    "Circular_long": "Circular",
    "Nuclear_lambda1": r"Nuclear ($\lambda=1$)",
    "Density": "Density",
    "Product": "Count product",
    "TrueP": r"True $P^\star$",
    "Q_uniform": r"$Q^{\mathrm{unif}}$",
    "Q_oracle": r"$Q^{\mathrm{or}}$",
}
COND_KEYS = {
    "auc_raw": ("auc", "all_pairs"),
    "auc_zero_adjusted": ("zero_adjusted_full_auc",),
    "auc_U": ("auc", "zero_nonexcluded_U"),
    "auc_V": ("auc", "full_fibre_unresolved_V"),
    "brier_A_V": ("brier", "full_fibre_unresolved_V"),
    "conditional_excess_brier_V": ("conditional_excess_brier", "full_fibre_unresolved_V"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    values = [float(x) for x in values]
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("Every summary requires finite saved observations")
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"n": len(values), "mean": statistics.mean(values), "sd": sd,
            "mcse": sd / math.sqrt(len(values)), "min": min(values), "max": max(values),
            "positive": sum(x > 0 for x in values), "negative": sum(x < 0 for x in values),
            "zero": sum(x == 0 for x in values)}


def get_path(value, keys):
    for key in keys:
        value = value[key]
    return value


def pointer(*parts):
    return "/" + "/".join(str(x).replace("~", "~0").replace("/", "~1") for x in parts)


class Builder:
    def __init__(self, output):
        self.output = output
        self.data = {key: json.loads((ROOT / path).read_text()) for key, path in SOURCE_PATHS.items()}
        self.sources = {key: {"path": path, "sha256": digest(ROOT / path)}
                        for key, path in SOURCE_PATHS.items()}
        self.checks = []
        self.tables = {}

    def source(self, key, json_pointer):
        return {**self.sources[key], "json_pointer": json_pointer}

    def check_summary(self, name, values, saved, source, saved_pointer):
        computed = stats(values)
        differences = {}
        for key in computed:
            if key in saved:
                difference = abs(computed[key] - saved[key])
                differences[key] = difference
                if not math.isclose(computed[key], saved[key], rel_tol=1e-11, abs_tol=1e-12):
                    raise ValueError(f"{name}: {key}: computed {computed[key]} != saved {saved[key]}")
        self.checks.append({"name": name, "passed": True, "source": self.source(source, saved_pointer),
                            "computed": computed, "max_absolute_difference": max(differences.values(), default=0)})
        return computed

    def stage_values(self, k, model, metric):
        return [{"seed": run["seed"], "value": run["metrics"][model][metric],
                 "json_pointer": pointer("runs", i, "metrics", model, metric)}
                for i, run in enumerate(self.data["stage_a"]["runs"]) if run["k"] == k]

    def cond_values(self, model, metric):
        keys = COND_KEYS[metric]
        return [{"seed": run["seed"], "value": get_path(run["metrics"][model], keys),
                 "json_pointer": pointer("runs", i, "metrics", model, *keys)}
                for i, run in enumerate(self.data["conditional"]["runs"])]

    def mean_cell(self, k, model, metric):
        if model in ("TrueP", "Q_uniform", "Q_oracle"):
            source = "conditional"
            observations = self.cond_values(model, metric)
            keys = COND_KEYS[metric]
            saved = get_path(self.data[source]["summary"][model], keys)
            saved_pointer = pointer("summary", model, *keys)
        else:
            source = "stage_a"
            observations = self.stage_values(k, model, metric)
            matches = [(i, row) for i, row in enumerate(self.data[source]["summary"])
                       if (row["k"], row["model"], row["metric"]) == (k, model, metric)]
            if len(matches) != 1:
                raise ValueError(f"Expected one summary for {(k, model, metric)}")
            index, saved = matches[0]
            saved_pointer = pointer("summary", index)
        summary = self.check_summary(f"cell/{k}/{model}/{metric}", [o["value"] for o in observations],
                                     saved, source, saved_pointer)
        if summary["n"] != 40:
            raise ValueError("Manuscript cell does not contain exactly 40 graphs")
        mean_dp, mcse_dp = (3, 4) if metric.startswith("auc_") else (4, 5)
        rendered = f"{summary['mean']:.{mean_dp}f} ({summary['mcse']:.{mcse_dp}f})"
        return {"metric": metric, **summary, "rendered": rendered,
                "display_precision": {"mean_decimal_places": mean_dp, "mcse_decimal_places": mcse_dp},
                "source": self.source(source, saved_pointer), "observations": observations}

    def cap_cell(self, k, penalty):
        matches = [(i, row) for i, row in enumerate(self.data["nuclear"]["summary"])
                   if (row["k"], row["penalty"]) == (k, penalty)]
        index, saved = matches[0]
        caps = [{"value": cap, "json_pointer": pointer("final_caps", i)}
                for i, cap in enumerate(self.data["nuclear"]["final_caps"])
                if (cap["k"], cap["penalty"]) == (k, penalty)]
        if saved["final_capped"] != len(caps):
            raise ValueError("Cap count differs from individual saved caps")
        return {"value": len(caps), "rendered": str(len(caps)), "observations": caps,
                "source": self.source("nuclear", pointer("summary", index, "final_capped"))}

    def validate_sources(self):
        stage, conditional, nuclear = (self.data[key] for key in ("stage_a", "conditional", "nuclear"))
        if stage["status"] != "complete" or conditional["pilot"]:
            raise ValueError("Expected completed Stage A and full conditional benchmark")
        if not (len(stage["seeds"]) == 40 and stage["seeds"] == conditional["seeds"]):
            raise ValueError("The two sources must use the same 40 seeds")
        expected = {(k, seed) for k in (4, 8) for seed in stage["seeds"]}
        if {(run["k"], run["seed"]) for run in stage["runs"]} != expected or len(stage["runs"]) != 80:
            raise ValueError("Stage A must retain every graph/K cell exactly once")
        if len(conditional["runs"]) != 40 or {run["seed"] for run in conditional["runs"]} != set(stage["seeds"]):
            raise ValueError("Conditional benchmark must retain all 40 graphs")
        if nuclear["expected_fits"] != 240 or nuclear["saved_fits"] != 240 or nuclear["missing"] or nuclear["failed"]:
            raise ValueError("Nuclear fit accounting is incomplete")
        if stage["nuclear_final_caps"] != nuclear["final_caps"] or len(nuclear["final_caps"]) != 5:
            raise ValueError("The five cap records must be preserved across sources")
        # The original and released files differ by an internal status sentence.
        exploratory = ROOT / "exploratory"
        manifest = json.loads((exploratory / "MANIFEST.json").read_bytes())["files"]
        originals_path = exploratory / "ORIGINAL_RUN_HASHES.json"
        if digest(originals_path) != manifest[originals_path.name]["sha256"]:
            raise ValueError("Original run hash record differs from the release manifest")
        originals = json.loads(originals_path.read_bytes())
        if self.data["method_contrasts"]["summary_input_sha256"] != originals["stability_summary"]:
            raise ValueError("Method contrasts do not identify the archived summary")
        for label in ("stage_a", "method_contrasts", "conditional", "nuclear"):
            record = self.sources[label]
            relative = Path(record["path"]).relative_to("exploratory").as_posix()
            if record["sha256"] != manifest[relative]["sha256"]:
                raise ValueError(f"Released {label} differs from the manifest")
        numerical_record = {k: v for k, v in stage.items() if k != "limitations"}
        canonical = json.dumps(numerical_record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if hashlib.sha256(canonical.encode()).hexdigest() != "f24564a0e8d35a8f7d0108a0b2232eae8c3293bbf1c8c48411928425e5c91e94":
            raise ValueError("Stage A fields differ from the archived scientific record")
        for i, row in enumerate(stage["summary"]):
            values = self.stage_values(row["k"], row["model"], row["metric"])
            self.check_summary(f"stage_summary/{i}", [v["value"] for v in values], row,
                               "stage_a", pointer("summary", i))
        for i, row in enumerate(stage["paired_contrasts"]):
            values = [run["metrics"][row["left"]][row["metric"]] - run["metrics"][row["right"]][row["metric"]]
                      for run in stage["runs"] if run["k"] == row["k"]]
            self.check_summary(f"stage_paired/{i}", values, row, "stage_a", pointer("paired_contrasts", i))
        for i, row in enumerate(self.data["method_contrasts"]["contrasts"]):
            model = f"Nuclear_lambda{row['lambda']:g}"
            values = [run["metrics"][model][row["metric"]] - run["metrics"]["Circular_long"][row["metric"]]
                      for run in stage["runs"] if run["k"] == row["k"]]
            self.check_summary(f"method_paired/{i}", values, row, "method_contrasts", pointer("contrasts", i))
        for key, saved in conditional["paired_contrasts"].items():
            contrast, metric, subset = key.split("/")
            left, right = contrast.split("_minus_")
            values = [run["metrics"][left][metric][subset] - run["metrics"][right][metric][subset]
                      for run in conditional["runs"]]
            self.check_summary(f"conditional_paired/{key}", values, saved,
                               "conditional", pointer("paired_contrasts", key))
        metrics = {"final_auc": "auc_raw", "final_probability_rmse": "probability_rmse",
                   "final_ard_rmse": "ard_rmse", "final_within_u_auc": "auc_U",
                   "final_zero_adjusted_auc": "auc_zero_adjusted"}
        for i, row in enumerate(nuclear["summary"]):
            model = f"Nuclear_lambda{row['penalty']:g}"
            for key, metric in metrics.items():
                values = self.stage_values(row["k"], model, metric)
                self.check_summary(f"nuclear_final/{i}/{metric}", [v["value"] for v in values], row[key],
                                   "nuclear", pointer("summary", i, key))
        # Conditional Qs are unchanged exact benchmarks; compare every corresponding metric.
        for model in ("Q_uniform", "Q_oracle"):
            for metric in COND_KEYS:
                stage_values = self.stage_values(8, model, metric)
                exact_values = self.cond_values(model, metric)
                if [(x["seed"], x["value"]) for x in stage_values] != [(x["seed"], x["value"]) for x in exact_values]:
                    raise ValueError(f"Stage A changed exact benchmark {model}/{metric}")

    def auxiliary_counts(self):
        conditional = self.data["conditional"]
        result = {"k8": {}}
        for metric, saved in conditional["count_summary"].items():
            observations = [{"seed": run["seed"], "value": run["counts"][metric],
                             "json_pointer": pointer("runs", i, "counts", metric)}
                            for i, run in enumerate(conditional["runs"])]
            summary = self.check_summary(f"conditional_counts/{metric}", [x["value"] for x in observations],
                                         saved, "conditional", pointer("count_summary", metric))
            result["k8"][metric] = {**summary, "observations": observations,
                                      "source": self.source("conditional", pointer("count_summary", metric))}
        # The old diagnostics are used only for K4's zero-count exclusion summary.
        zero = self.data["k4_zero"]
        observations = [{"seed": run["seed"], "value": run["zero_only_auc"],
                         "json_pointer": pointer("runs", i, "zero_only_auc")}
                        for i, run in enumerate(zero["runs"]) if run["k"] == 4]
        summary = self.check_summary("k4_zero_only_auc", [x["value"] for x in observations],
                                     zero["zero_only_summary"]["4"], "k4_zero", pointer("zero_only_summary", "4"))
        result["k4_zero_only_auc"] = {**summary, "observations": observations,
                                       "source": self.source("k4_zero", pointer("zero_only_summary", "4"))}
        exclusion = [{"seed": run["seed"], "value": run["nonedges_excluded_by_zero_fraction"],
                      "json_pointer": pointer("runs", i, "nonedges_excluded_by_zero_fraction")}
                     for i, run in enumerate(zero["runs"]) if run["k"] == 4]
        result["k4_zero_excluded_nonedge_fraction"] = {**stats([x["value"] for x in exclusion]),
            "observations": exclusion, "source": self.source("k4_zero", pointer("runs"))}
        return result

    def table(self, name, colspec, columns, rows, group_key=None):
        lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule",
                 " & ".join(label for _, label in columns) + r" \\", r"\midrule"]
        last_group = None
        for row in rows:
            if group_key and last_group is not None and row[group_key]["value"] != last_group:
                lines.append(r"\midrule")
            lines.append(" & ".join(row[key]["rendered"] for key, _ in columns) + r" \\")
            if group_key:
                last_group = row[group_key]["value"]
        lines += [r"\bottomrule", r"\end{tabular}", ""]
        content = "\n".join(lines)
        self.tables[name] = {"path": name, "column_spec": colspec,
                             "columns": [{"id": key, "label_tex": label} for key, label in columns], "rows": rows,
                             "sha256": hashlib.sha256(content.encode()).hexdigest()}
        return content

    def row(self, k, model, metrics):
        source = "conditional" if model in ("TrueP", "Q_uniform", "Q_oracle") else "stage_a"
        source_runs = [(i, r) for i, r in enumerate(self.data[source]["runs"]) if r["k"] == k]
        index = source_runs[0][0]
        return {"k": {"value": k, "rendered": str(k), "source": self.source(source, pointer("runs", index, "k"))},
                "model": {"value": model, "rendered": LABELS[model],
                          "source": self.source(source, pointer("runs", index, "metrics", model))},
                **{metric: self.mean_cell(k, model, metric) for metric in metrics}}

    def build(self):
        self.validate_sources()
        counts = self.auxiliary_counts()
        products = {}
        metrics = ("probability_rmse", "auc_raw", "ard_rmse")
        rows = [self.row(k, model, metrics) for k in (4, 8)
                for model in ("Circular_long", "Nuclear_lambda1", "Density")]
        products["estimator-results.tex"] = self.table("estimator-results.tex", "llrrr",
            [("k", "$K$"), ("model", "Estimator"), ("probability_rmse", r"$P^\star$ RMSE"),
             ("auc_raw", "Raw AUC"), ("ard_rmse", "Count RMSE")], rows, "k")
        metrics = ("auc_raw", "auc_zero_adjusted", "auc_U", "auc_V")
        rows = [self.row(8, model, metrics) for model in
                ("Product", "TrueP", "Circular_long", "Nuclear_lambda1", "Q_uniform", "Q_oracle")]
        products["ranking.tex"] = self.table("ranking.tex", "lrrrr",
            [("model", "Score"), ("auc_raw", "Raw AUC"), ("auc_zero_adjusted", "Zero-adjusted AUC"),
             ("auc_U", r"AUC on $\mathcal{U}$"), ("auc_V", r"AUC on $\mathcal{V}$")], rows)
        metrics = ("brier_A_V", "conditional_excess_brier_V")
        rows = [self.row(8, model, metrics) for model in
                ("Q_uniform", "Q_oracle", "TrueP", "Circular_long", "Nuclear_lambda1")]
        products["conditional-risk.tex"] = self.table("conditional-risk.tex", "lrr",
            [("model", "Probability matrix"), ("brier_A_V", r"Realized Brier on $\mathcal{V}$"),
             ("conditional_excess_brier_V", r"Conditional excess Brier on $\mathcal{V}$")], rows)
        rows = []
        for k in (4, 8):
            for penalty in (0.25, 1, 4):
                model = f"Nuclear_lambda{penalty:g}"
                matches = [(i, r) for i, r in enumerate(self.data["nuclear"]["summary"])
                           if (r["k"], r["penalty"]) == (k, penalty)]
                index = matches[0][0]
                row = {"k": {"value": k, "rendered": str(k),
                             "source": self.source("nuclear", pointer("summary", index, "k"))},
                       "penalty": {"value": penalty, "rendered": f"{penalty:g}",
                                   "source": self.source("nuclear", pointer("summary", index, "penalty"))}}
                for metric in ("probability_rmse", "auc_raw", "auc_U", "auc_V"):
                    row[metric] = self.mean_cell(k, model, metric) if k == 8 or metric != "auc_V" else {
                        "value": None, "rendered": "--", "reason": "Exact finite-fibre V evaluated only at K=8"}
                row["final_caps"] = self.cap_cell(k, penalty)
                rows.append(row)
        products["penalty.tex"] = self.table("penalty.tex", "rrrrrrr",
            [("k", "$K$"), ("penalty", r"$\lambda$"), ("probability_rmse", r"$P^\star$ RMSE"),
             ("auc_raw", "Raw AUC"), ("auc_U", r"AUC on $\mathcal{U}$"), ("auc_V", r"AUC on $\mathcal{V}$"),
             ("final_caps", "Final caps")], rows, "k")
        # Refuse source drift between input loading and output writing.
        for key, source in self.sources.items():
            if digest(ROOT / source["path"]) != source["sha256"]:
                raise ValueError(f"Source changed while building tables: {key}")
        for name, content in products.items():
            reference = HERE / "reference" / name
            if reference.read_bytes() != content.encode():
                raise ValueError(f"Regenerated table differs from submitted source: {name}")
        self.output.mkdir(parents=True, exist_ok=False)
        for name, content in products.items():
            (self.output / name).write_text(content)
        validation = {"schema_version": 1, "status": "passed", "generator": "exploratory-tables/rebuild.py",
            "generator_sha256": digest(Path(__file__)), "sources": self.sources,
            "aggregation": "Unweighted mean of 40 saved per-graph metrics; MCSE = sample SD / sqrt(40).",
            "no_fitting_or_sampling": True, "tables": self.tables, "auxiliary_counts": counts,
            "final_caps": self.data["nuclear"]["final_caps"],
            "final_caps_source": self.source("nuclear", "/final_caps"),
            "checks": self.checks, "check_count": len(self.checks),
            "scope": {"circular": "Stage A four-chain long-run mean", "nuclear": "Stage A final ADMM iterates",
                      "conditional": "Exact K8 benchmarks; no generating-P RMSE assigned to conditional Q",
                      "k4_v": "Not evaluated", "penalty_selection": "None; all three prespecified penalties retained"}}
        (self.output / "table-validation.json").write_text(json.dumps(validation, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": "passed", "tables": list(products), "checks": len(self.checks),
                          "final_caps": len(validation["final_caps"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New directory for four TeX files and the validation report")
    Builder(parser.parse_args().out.resolve()).build()
