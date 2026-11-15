"""Rescore the archived fitted-weight marginals using the corrected probability-tie rule."""
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import numpy as np
import plugin

BASE = Path(__file__).resolve().parents[1]
DESIGN = json.loads((BASE/"reference/plugin/design.json").read_text())

def routes(studies=None):
    studies = studies or {}
    for cohort in DESIGN["scope"]["cohorts"]:
        if cohort == "exploratory":
            lane = studies.get("exploratory",BASE/"exploratory/project/analysis/ard-reconstruction")
            stage = lane/"compstat-stability-2026-11-04"
            inp = lane/"jspi-rerun-2026-10-14/fresh-results"
            ref = lane/"compstat-conditional-2026-11-04/results"
            circ, nuc = stage/"circular",stage/"nuclear/fits"
            seeds = [101,202,303,404,*range(1001,1037)]
        else:
            tag = "sbm" if cohort=="sbm" else "fresh"
            condition = cohort.removeprefix("fresh_")
            lane = studies.get(tag, BASE/"studies"/tag)
            seeds = json.loads((lane/"RUN_CONTRACT.json").read_text())["seeds"]
            inp, ref, circ, nuc = [lane / d / condition for d in ("inputs","conditional","circular","nuclear")]
        for seed in seeds:
            stem = f"run_k8_seed{seed}"
            yield dict(cohort=cohort,seed=seed,input=inp/(stem+".npz"),reference=ref/(stem+".npz"),
                       circular=circ/stem/"summary.npz",circular_meta=circ/stem/"complete.json",
                       nuclear={str(lam):nuc/(stem+f"_lambda{lam}.npz") for lam in (0.25,1,4)})

def corrected_scores(matrix, arrays, reference, logits):
    metrics = plugin.score_matrix(matrix, arrays, reference, logits)
    for label in ("auc_V","auc_full"):
        metrics[label+"_logit_diagnostic"] = metrics[label]
        metrics[label] = metrics[label+"_probability"]
    return metrics

def save_csv(path,rows):
    with path.open("x",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in row.items()})

def rescore(directory, matrices=None, studies=None):
    rows = []
    matrices = matrices or BASE/"plugin-matrices"
    for case in routes(studies):
        arrays = plugin.read_npz(case["input"], ("Y","B","hidden_A","true_P"))
        reference = plugin.read_npz(case["reference"],("q_oracle","unresolved","forced_zero","forced_one"))
        with np.load(matrices / case["cohort"] / case["input"].name, allow_pickle=False) as z:
            metrics = {}
            for key in z.files:
                if key not in ("Q_uniform","Q_oracle","Circular") and not key.startswith(("Nuclear_lambda","Clipped_","Q_plugin_")):
                    continue
                if key.endswith("_logit_score"):
                    continue
                logits=z[key+"_logit_score"] if key.startswith("Q_plugin_") else None
                metrics[key] = corrected_scores(z[key], arrays, reference, logits)
            if len(metrics)!=25:
                raise ValueError("Expected 25 scores per graph")
        rows.append(dict(cohort=case["cohort"],seed=case["seed"],metrics=metrics))
    summary, contrasts = plugin.summarize(rows, DESIGN)
    directory.mkdir(parents=True, exist_ok=False)
    save_csv(directory/"summary.csv",summary)
    save_csv(directory/"paired-contrasts.csv",contrasts)
    return rows

def reweight_case(args):
    case,directory,compare = args
    plugin.evaluate_case(case, DESIGN, directory)
    if compare:
        local=directory/case["cohort"]/case["input"].name
        saved=BASE/"plugin-matrices"/case["cohort"]/case["input"].name
        with np.load(local,allow_pickle=False) as a,np.load(saved,allow_pickle=False) as b:
            if set(a.files)!=set(b.files): raise ValueError("Plug-in matrix schema changed")
            for key in a.files:
                if a[key].dtype.kind in "fc":
                    np.testing.assert_allclose(a[key],b[key],rtol=2e-12,atol=2e-13,err_msg=key)
                else:
                    np.testing.assert_array_equal(a[key],b[key])
    return case["cohort"],case["seed"]

def reweight(directory,workers,compare,studies=None):
    directory.mkdir(parents=True,exist_ok=False)
    jobs = [(case,directory,compare) for case in routes(studies)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,_ in enumerate(pool.map(reweight_case,jobs),1):
            if i % 40 == 0: print(f"Reweighted {i}/200 graphs",flush=True)
