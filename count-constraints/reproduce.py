"""Recompute manuscript results using only files in this archive."""
import argparse
import csv
import hashlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import subprocess
import sys

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/"code"))
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import evaluate
import paper
import plugin_readout

def compare_csv(found,expected):
    with found.open(newline="") as a,expected.open(newline="") as b:
        left,right=csv.DictReader(a),csv.DictReader(b)
        if left.fieldnames!=right.fieldnames:
            raise ValueError(f"Column mismatch in {found}")
        rows=0
        for x,y in zip_longest(left,right):
            if x is None or y is None: raise ValueError(f"Row count mismatch in {found}")
            rows+=1
            for k in x:
                if x[k]==y[k]: continue
                try: equal=math.isclose(float(x[k]),float(y[k]),rel_tol=2e-12,abs_tol=2e-13)
                except ValueError: equal=False
                if not equal: raise ValueError(f"{found.name}, row {rows}, {k}: {x[k]} vs {y[k]}")
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["verify","rescore","reweight"])
    parser.add_argument("--out",type=Path,help="Create a new output directory")
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--compile-figures",action="store_true")
    parser.add_argument("--matrices",type=Path,help="Read newly regenerated plug-in matrices for rescore")
    parser.add_argument("--fresh", type=Path, help="Read regenerated fresh study results")
    parser.add_argument("--sbm", type=Path, help="Read regenerated SBM results")
    parser.add_argument("--exploratory",type=Path,help="Read regenerated exploratory analysis tree")
    args=parser.parse_args()
    studies={key:getattr(args,key).resolve() for key in ("fresh","sbm","exploratory") if getattr(args,key)}
    if args.action=="verify":
        sums=json.loads((BASE/"MANIFEST.json").read_text())
        for name,digest in sums.items():
            if hashlib.sha256((BASE/name).read_bytes()).hexdigest()!=digest: raise ValueError(name)
        print(f"Verified all {len(sums)} archived files")
        return
    if args.out is None: parser.error("--out is required")
    directory=args.out.resolve()
    if args.action=="reweight":
        plugin_readout.reweight(directory,args.workers,not studies,studies)
        return
    directory.mkdir(parents=True,exist_ok=False)
    result={}
    for study in ("fresh","sbm"):
        output=directory/study
        output.mkdir()
        summary=evaluate.summarize(studies.get(study,BASE/"studies"/study),output)
        if summary["status"]!="complete": raise ValueError(summary["failures"])
        result[study]={"complete_cases":summary["n_complete_cells"]}
    plugin_rows=plugin_readout.rescore(directory/"plugin",args.matrices,studies)
    result["plugin"]={"complete_cases":len(plugin_rows)}
    for study in ("fresh","sbm","plugin"):
        for name in ("summary.csv","paired-contrasts.csv"):
            result[study][name]=compare_csv(directory/study/name,BASE/"reference"/study/name)
    result["matched_table_rows"]=paper.paper_tables(directory,plugin_rows)
    table_out=directory/"exploratory-tables"
    subprocess.run([sys.executable,"-B",str(BASE/"exploratory-tables/rebuild.py"),"--out",str(table_out)],check=True)
    result["exploratory_tables"]=str((table_out/"table-validation.json").relative_to(directory))
    checked=directory/"exploratory-verification.json"
    subprocess.run([sys.executable,"-B",str(BASE/"exploratory/verify_saved.py"),"--report",str(checked)],check=True)
    result["exploratory_verification"]=checked.name
    paper.figures(directory)
    if args.compile_figures:
        for stem in ("positive-margins","density-unresolved-auc"):
            subprocess.run(["pdflatex","-interaction=batchmode","-halt-on-error",stem+".tex"],cwd=directory,check=True)
    (directory/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
