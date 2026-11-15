"""Produce manuscript table bodies and both figure sources from replicate-level results."""
import json
import math
from pathlib import Path
import re
import statistics
import numpy as np
import density_figure
from conditional import enumerate_bipartite

REFERENCE = Path(__file__).resolve().parents[1] / "reference"
DENSITIES = ("baseline","d15","d25")
METHODS = ("Product","Circular","Nuclear_lambda0.25","Nuclear_lambda1","Nuclear_lambda4","Q_uniform","Q_oracle")

def texrow(parts):
    return " & ".join(parts) + r" \\"

def write_table(directory, name, entries):
    """Reuse manuscript labels and formatting, replace all numerical data fields."""
    reference = (REFERENCE/(name+".tex")).read_text()
    lines = reference.splitlines()
    used = []
    for idx, line in enumerate(lines):
        if not entries: break
        labels, numbers = entries[0]
        if not line.startswith(labels): continue
        if not labels.endswith(" & "):  # spanning cohort title with a computed uniform AUC
            suffix = numbers[0]+"} "+r"\\"
            lines[idx] = labels + suffix
        else:
            lines[idx] = labels + texrow(numbers)
        used.append(idx)
        entries.pop(0)
    if entries: raise ValueError(f"{name}: manuscript row missing: {entries[0]}")
    result="\n".join(lines)+"\n"
    (directory/(name+".tex")).write_text(result)
    if result != reference:
        import difflib
        difference="\n".join(difflib.unified_diff(reference.splitlines(),result.splitlines()))
        raise ValueError(f"{name} disagrees with the manuscript\n{difference}")
    return len(used)

def paper_tables(directory, plug_rows):
    cells = {}
    for study in ("fresh","sbm"):
        for c in json.loads((directory/study/"cell-metrics.json").read_text())["cells"]:
            if c["status"] != "complete": raise ValueError(c["failures"])
            cells.setdefault((c["density"],c["k"]),[]).append(c)
    def stat(d,k,model,metric,right=None):
        group=cells[(d,k)]
        if len(group)!=40: raise ValueError("Expected forty independent blocks")
        def val(c,name):
            return c["counts"][metric] if name=="Design" else c["metrics"][name][metric]
        v=[val(c,model)-(val(c,right) if right else 0) for c in group]
        return statistics.mean(v),statistics.stdev(v)/math.sqrt(len(v))
    def fmt(values,digits=3,err=4,scale=1,signed=False):
        mean,se=values
        sign="+" if signed else ""
        result=f"{mean*scale:{sign}.{digits}f}"
        if result.startswith("-"): result="$"+result+"$"
        return f"{result} ({se*scale:.{err}f})"
    def labels(name):
        lines=(REFERENCE/(name+".tex")).read_text().splitlines()
        return [line for line in lines if r"8.53\%" not in line and re.search(r" & (?:\$?[+-]?\d+\.\d+|---)",line)]
    def panel(name, outputs):
        ss=labels(name)
        if len(ss)!=len(outputs): raise ValueError(f"{name}: {len(ss)} labels for {len(outputs)} output rows")
        entries=[]
        for line, row in zip(ss,outputs):
            prefix=line.split(" & ")[:-len(row)]
            entries.append((" & ".join(prefix)+" & ",row))
        return write_table(directory,name,entries)
    report={}
    design=[(4,"q0",1,1,100),(4,"remaining_U",2,2,1),(8,"q0",1,1,100),
            (8,"forced_zero_nonedge_fraction",1,1,100),(8,"remaining_U",2,2,1),
            (8,"unresolved_V",2,2,1),(8,"forced_one_edge_fraction",1,1,100),(8,"forced_only_auc",3,4,1)]
    rows=[[fmt(stat(d,k,"Design",m),dp,se,sc) for d in DENSITIES] for k,m,dp,se,sc in design]
    report["1"]=panel("density-design",rows)
    rows=[[fmt(stat(d,8,m,"auc_V")) for d in DENSITIES] for m in METHODS]
    for m,r in [("Q_uniform","Nuclear_lambda0.25"),("Q_uniform","Nuclear_lambda1"),("Q_uniform","Circular"),("Q_oracle","Q_uniform")]:
        rows.append([fmt(stat(d,8,m,"auc_V",r),4,signed=True) for d in DENSITIES])
    report["2"]=panel("density-ranking",rows)
    rows=[]
    for k in (4,8):
        for m in ("Density","Circular","reduction","Nuclear_lambda1","difference"):
            row=[]
            for d in DENSITIES:
                if m=="reduction":
                    pct=100*(1-stat(d,k,"Circular","probability_rmse")[0]/stat(d,k,"Density","probability_rmse")[0])
                    row.append(f"{pct:.2f}")
                elif m=="difference":
                    row.append(fmt(stat(d,k,"Nuclear_lambda1","auc_U","Product"),4,4,signed=True))
                else:
                    row.append(fmt(stat(d,k,m,"probability_rmse"),4,5))
            rows.append(row)
    report["3"]=panel("density-estimation",rows)
    rows=[]
    for m,dp,se,sc in [("q0",1,1,100),("forced_zero_nonedge_fraction",1,1,100),("forced_one_edge_fraction",1,1,100),("unresolved_V",2,2,1)]:
        rows.append([fmt(stat(d,8,"Design",m),dp,se,sc) for d in ("d15","sbm")])
    for m,r in [("Q_uniform","Product"),("Q_oracle","Q_uniform"),("Q_uniform","Nuclear_lambda0.25"),("Q_uniform","Nuclear_lambda1")]:
        rows.append([fmt(stat("sbm",8,m,"auc_V",r),4,4,signed=True)])
    report["4"]=panel("sbm-context",rows)
    rows=[]
    for model in (*METHODS,"TrueP","Density"):
        row=[]
        for metric,dp,se in [("auc_raw",3,4),("auc_V",3,4),("probability_rmse",4,5),("conditional_excess_brier_V",4,4)]:
            if ((metric=="probability_rmse" and model in ("Product","Q_uniform","Q_oracle")) or
                (metric=="conditional_excess_brier_V" and model=="Product")): row.append("---")
            else: row.append(fmt(stat("sbm",8,model,metric),dp,se))
        rows.append(row)
    report["5"]=panel("sbm-comparison",rows)
    lines=(REFERENCE/"plugin-comparison.tex").read_text().splitlines()
    rows=[]
    headings=[line.split("uniform AUC ")[0]+"uniform AUC " for line in lines if "uniform AUC " in line]
    for cohort,h in zip(("fresh_baseline","fresh_d15","fresh_d25","sbm","exploratory"),headings):
        group=[r for r in plug_rows if r["cohort"]==cohort]
        if len(group)!=40: raise ValueError("Missing plug-in cases")
        def scores(m,metric):
            return np.array([r["metrics"][m][metric] for r in group])
        rows.append((h,[f'{scores("Q_uniform","auc_V").mean():.3f}']))
        for model in METHODS[1:5]:
            plug="Q_plugin_"+model+("_eps1e-06" if model!="Circular" else "")
            vals=[f'{scores(m,"auc_V").mean():.3f}' for m in (model,plug)]
            for met in ("auc_V","conditional_excess_brier_V"):
                diff=scores(plug,met)-scores("Q_uniform",met)
                vals.append(f"${diff.mean():+.4f}$ ({diff.std(ddof=1)/math.sqrt(40):.4f})")
            label="Circular" if model=="Circular" else r"Nuclear ($\lambda="+model.split("lambda")[1]+"$)"
            rows.append((label+" & ",vals))
    report["7"]=write_table(directory,"plugin-comparison",rows)
    return report

def figures(directory):
    margins=(3,3,1,1)
    configs=enumerate_bipartite(margins,margins)
    if len(configs)!=4: raise ValueError("Expected four configurations")
    low, high = configs.min(axis=0), configs.max(axis=0)
    cells=np.where(low==high, low.astype(str),r"\ast")
    rows=[]
    for i,row in enumerate(cells):
        rows.append("&".join(row)+(r"\\" if i<3 else "")+(r"\hline" if i==1 else ""))
    tex = "\n".join([r"\documentclass[border=3pt]{standalone}",r"\usepackage{amsmath}",r"\begin{document}",r"$\displaystyle",
        r"\left[\begin{array}{cc|cc}", *rows, r"\end{array}\right]\qquad\begin{array}{l}",
        r"\text{Row sums: }(3,3,1,1)\\[3pt]",r"\text{Column sums: }(3,3,1,1)",
        r"\end{array}$",r"\end{document}",""])
    (directory/"positive-margins.tex").write_text(tex)
    (directory/"positive-margins-configurations.json").write_text(json.dumps(configs.tolist(),indent=2)+"\n")
    density_figure.render(directory,directory/"fresh/SUMMARY.json",directory/"density-ranking.tex")
    expected=json.loads((REFERENCE/"figure-2-points.json").read_text())
    points=json.loads((directory/"density-unresolved-auc-data.json").read_text())["points"]
    for a,b in zip(expected,points,strict=True):
        if (a["density"],a["model"])!=(b["density"],b["model"]): raise ValueError("Figure panel or method differs")
        np.testing.assert_allclose([a["mean"],a["mcse"]],[b["mean"],b["mcse"]],rtol=0,atol=2e-14)
